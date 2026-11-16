"""Hybrid retrieval: BM25 + dense (RRF fusion) + reranker.

Dense backends
  * OpenAIEmbedder   - the task's embeddings endpoint, with an on-disk cache so each
                       chunk is embedded once (the old code re-embedded every chunk per query).
  * LsaEmbedder      - offline fallback (TF-IDF + truncated SVD). Not a neural model; used
                       for tests / when no endpoint is configured. Reported as such in evals.
Rerankers
  * LexicalReranker  - cheap, deterministic cross-feature scorer (default)
  * LLMReranker      - pointwise relevance scoring through the shared <=8B model (optional)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .chunking import Chunk, load_corpus
from .text_utils import tokenize


@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float
    sources: Dict[str, float] = field(default_factory=dict)  # per-stage scores for tracing

    @property
    def file_path(self) -> str:
        return self.chunk.path


# ----------------------------------------------------------------------------- BM25
class BM25Index:
    def __init__(self, texts: Sequence[str], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf: List[Counter] = [Counter(tokenize(t)) for t in texts]
        self.dl = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.avgdl = float(self.dl.mean()) if len(self.dl) else 1.0
        self.postings: Dict[str, List[int]] = defaultdict(list)
        for i, c in enumerate(self.tf):
            for term in c:
                self.postings[term].append(i)
        n = len(texts)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query: str) -> Dict[int, float]:
        out: Dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            if term not in self.idf:
                continue
            for i in self.postings[term]:
                tf = self.tf[i][term]
                denom = tf + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                out[i] += self.idf[term] * tf * (self.k1 + 1) / denom
        return out


# ------------------------------------------------------------------------ embedders
class LsaEmbedder:
    name = "lsa-offline"

    def __init__(self, dim: int = 192):
        self.dim, self.vec, self.svd = dim, None, None

    def fit(self, texts: Sequence[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, ngram_range=(1, 2),
                                   min_df=1, sublinear_tf=True, token_pattern=None)
        x = self.vec.fit_transform(texts)
        k = max(2, min(self.dim, x.shape[0] - 1, x.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0)
        return _l2(self.svd.fit_transform(x))

    def encode_query(self, q: str) -> np.ndarray:
        return _l2(self.svd.transform(self.vec.transform([q])))[0]


class OpenAIEmbedder:
    """OpenAI-compatible /embeddings with a persistent cache (keyed by model+text hash)."""

    def __init__(self, base_url: str, api_key: str, model: str, cache_dir: str = ".cache/embeddings",
                 batch: int = 32):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=60)
        self.model, self.batch = model, batch
        self.name = f"api:{model}"
        self.cache_path = Path(cache_dir) / f"{hashlib.sha1(model.encode()).hexdigest()[:10]}.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, List[float]] = (
            json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {})

    def _key(self, t: str) -> str:
        return hashlib.sha1(t.encode()).hexdigest()

    def _embed(self, texts: List[str]) -> List[List[float]]:
        import time
        missing = [t for t in dict.fromkeys(texts) if self._key(t) not in self.cache]
        for i in range(0, len(missing), self.batch):
            part = missing[i : i + self.batch]
            for attempt in range(4):  # transient errors: exponential backoff; give up -> caller falls back
                try:
                    resp = self.client.embeddings.create(model=self.model, input=part)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(0.5 * 2 ** attempt)
            if len(resp.data) != len(part):
                raise ValueError(f"embeddings endpoint returned {len(resp.data)} vectors for {len(part)} inputs")
            for t, d in zip(part, sorted(resp.data, key=lambda x: getattr(x, "index", 0))):
                self.cache[self._key(t)] = d.embedding
            self.cache_path.write_text(json.dumps(self.cache))  # checkpoint: a crash does not lose progress
        if missing:
            self.cache_path.write_text(json.dumps(self.cache))
        return [self.cache[self._key(t)] for t in texts]

    def fit(self, texts: Sequence[str]) -> np.ndarray:
        return _l2(np.array(self._embed(list(texts)), dtype=float))

    def encode_query(self, q: str) -> np.ndarray:
        return _l2(np.array(self._embed([q]), dtype=float))[0]


def _l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(n == 0, 1, n)


def rrf(rankings: List[List[int]], k: int = 60, weights: Optional[List[float]] = None) -> Dict[int, float]:
    weights = weights or [1.0] * len(rankings)
    fused: Dict[int, float] = defaultdict(float)
    for w, ranking in zip(weights, rankings):
        for r, idx in enumerate(ranking):
            fused[idx] += w / (k + r + 1)
    return fused


# ------------------------------------------------------------------------ rerankers
class LexicalReranker:
    """idf-weighted query coverage + bigram phrase hits + heading/title overlap + fused prior."""
    name = "lexical"

    def __init__(self, bm25: BM25Index):
        self.idf = bm25.idf

    def rerank(self, query: str, hits: List[RetrievalHit]) -> List[RetrievalHit]:
        q = [t for t in dict.fromkeys(tokenize(query))]
        if not q:
            return hits
        qbi = {(a, b) for a, b in zip(q, q[1:])}
        tot_idf = sum(self.idf.get(t, 1.0) for t in q) or 1.0
        max_prior = max((h.score for h in hits), default=1.0) or 1.0
        for h in hits:
            body = tokenize(h.chunk.text)
            bset, head = set(body), set(tokenize(h.chunk.section + " " + h.chunk.doc_title))
            cover = sum(self.idf.get(t, 1.0) for t in q if t in bset) / tot_idf
            hcover = sum(self.idf.get(t, 1.0) for t in q if t in head) / tot_idf
            bigrams = {(a, b) for a, b in zip(body, body[1:])}
            phrase = len(qbi & bigrams) / max(1, len(qbi))
            h.sources["prior"] = h.score
            h.score = 0.45 * cover + 0.25 * hcover + 0.15 * phrase + 0.15 * (h.score / max_prior)
        return sorted(hits, key=lambda x: x.score, reverse=True)


class CrossEncoderReranker:
    """Neural cross-encoder (e.g. BAAI/bge-reranker-base) via sentence-transformers, scored on
    (query, "title > section: text") pairs. Optional dependency; PolicyCorpus falls back to the
    lexical reranker (and records why) if the model cannot be loaded."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", model=None, max_chars: int = 1200):
        if model is None:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(model_name, max_length=512)
        self.model, self.max_chars = model, max_chars
        self.name = f"cross-encoder:{model_name}"

    def rerank(self, query: str, hits: List[RetrievalHit]) -> List[RetrievalHit]:
        if not hits:
            return hits
        pairs = [(query, f"{h.chunk.doc_title} > {h.chunk.section}: {h.chunk.text[: self.max_chars]}") for h in hits]
        scores = self.model.predict(pairs)
        for h, sc in zip(hits, scores):
            h.sources["prior"], h.score = h.score, float(sc)
        return sorted(hits, key=lambda x: x.score, reverse=True)


class LLMReranker:
    """Pointwise rerank with the shared small LLM: 0-3 relevance grade via JSON. Optional."""
    name = "llm-pointwise"

    def __init__(self, llm, max_candidates: int = 10):
        self.llm, self.max_candidates = llm, max_candidates

    def rerank(self, query: str, hits: List[RetrievalHit]) -> List[RetrievalHit]:
        head, tail = hits[: self.max_candidates], hits[self.max_candidates :]
        for h in head:
            try:
                out = self.llm.json_call(
                    system="Grade how well the passage answers the question. Reply JSON {\"grade\": 0|1|2|3}.",
                    user=f"Question: {query}\n\nPassage:\n{h.chunk.text[:900]}",
                    schema={"type": "object", "properties": {"grade": {"type": "integer", "minimum": 0, "maximum": 3}},
                            "required": ["grade"]},
                    stage="rerank")
                g = int(out.get("grade", 0))
            except Exception:
                g = 0
            h.sources["prior"] = h.score
            h.score = g + 0.01 * h.score
        return sorted(head, key=lambda x: x.score, reverse=True) + tail


# --------------------------------------------------------------------------- corpus
class PolicyCorpus:
    def __init__(self, root: str | Path, chunking: str = "heading_cap", embedder="auto",
                 reranker: str = "lexical", llm=None, use_dense: bool = True, reranker_model=None, **chunk_kw):
        self.root = Path(root)
        self.chunking = chunking
        self.chunks, self.docs = load_corpus(self.root, chunking, **chunk_kw)
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.bm25 = BM25Index([c.indexed_text for c in self.chunks])
        self.embedder = None
        self.matrix: Optional[np.ndarray] = None
        self.notes: List[str] = []          # degradations (visible in the UI / eval output, never silent)
        if use_dense and self.chunks:
            self.embedder = self._make_embedder(embedder)
            if self.embedder is not None:
                try:
                    self.matrix = self.embedder.fit([c.indexed_text for c in self.chunks])
                except Exception as e:  # embeddings endpoint down/misconfigured -> offline fallback
                    self.notes.append(f"embedder {getattr(self.embedder, 'name', '?')} failed ({type(e).__name__}: {str(e)[:80]}); using offline LSA")
                    try:
                        self.embedder = LsaEmbedder()
                        self.matrix = self.embedder.fit([c.indexed_text for c in self.chunks])
                    except Exception:
                        self.embedder, self.matrix = None, None
        self.reranker = None
        self.reranker_name = reranker
        if reranker in ("lexical", "none"):
            self.reranker = LexicalReranker(self.bm25) if reranker == "lexical" else None
        elif reranker == "cross_encoder":
            try:
                self.reranker = reranker_model if reranker_model is not None else CrossEncoderReranker(
                    os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-base"))
            except Exception as e:
                self.notes.append(f"cross-encoder unavailable ({type(e).__name__}); using lexical reranker")
                self.reranker, self.reranker_name = LexicalReranker(self.bm25), "lexical"
        elif reranker == "llm" and llm is not None:
            self.reranker = LLMReranker(llm)

    @staticmethod
    def _make_embedder(spec):
        if spec is None or spec is False:
            return None
        if spec != "auto":
            return spec
        base, key = os.getenv("OPENAI_BASE_URL", ""), os.getenv("OPENAI_API_KEY", "")
        if base and key and os.getenv("EMBEDDING_MODEL"):
            try:
                return OpenAIEmbedder(base, key, os.environ["EMBEDDING_MODEL"])
            except Exception:
                pass
        try:
            return LsaEmbedder()
        except Exception:
            return None

    # -- stages exposed separately so evals can ablate them
    def bm25_rank(self, query: str, n: int) -> List[int]:
        s = self.bm25.scores(query)
        return [i for i, _ in sorted(s.items(), key=lambda kv: kv[1], reverse=True)[:n]]

    def dense_rank(self, query: str, n: int) -> List[int]:
        if self.matrix is None:
            return []
        sims = self.matrix @ self.embedder.encode_query(query)
        return [int(i) for i in np.argsort(-sims)[:n]]

    def search(self, query: str, top_k: int = 6, mode: str = "hybrid_rerank", pool: int = 30,
               max_per_doc: int = 3) -> List[RetrievalHit]:
        """mode: bm25 | dense | hybrid | hybrid_rerank"""
        b = self.bm25_rank(query, pool) if mode != "dense" else []
        d = self.dense_rank(query, pool) if mode != "bm25" else []
        if mode == "bm25":
            hits = [RetrievalHit(self.chunks[i], 1.0 / (r + 1), {"bm25_rank": r}) for r, i in enumerate(b)]
        elif mode == "dense":
            hits = [RetrievalHit(self.chunks[i], 1.0 / (r + 1), {"dense_rank": r}) for r, i in enumerate(d)]
        else:
            fused = rrf([b, d] if d else [b])
            brank, drank = {i: r for r, i in enumerate(b)}, {i: r for r, i in enumerate(d)}
            hits = [RetrievalHit(self.chunks[i], s, {"bm25_rank": brank.get(i, -1), "dense_rank": drank.get(i, -1)})
                    for i, s in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)]
            if mode == "hybrid_rerank" and self.reranker is not None:
                hits = self.reranker.rerank(query, hits[: pool])
        out, per_doc = [], Counter()
        for h in hits:  # cap chunks per document so one long file cannot crowd out the rest
            if per_doc[h.file_path] >= max_per_doc:
                continue
            per_doc[h.file_path] += 1
            out.append(h)
            if len(out) == top_k:
                break
        return out

    def read_section(self, path: str, section: str = "") -> str:
        parts = [c for c in self.chunks if c.path == path and (not section or section.lower() in c.section.lower())]
        return "\n\n".join(f"[{c.section}]\n{c.text}" for c in parts[:4])
