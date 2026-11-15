import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from openai import OpenAI

from .config import DATA_ROOT, EMBEDDING_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, has_model_config


TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class RetrievalHit:
    file_path: str
    title: str
    content: str
    score: float
    section: str = ""
    chunk_index: int = 0
    source: str = ""


class BM25Index:
    def __init__(self, documents: Sequence[Tuple[str, str, str]]):
        self.documents = list(documents)
        self.doc_freq: Dict[str, int] = {}
        self.doc_len: List[int] = []
        self.total_docs = len(documents)
        self.avgdl = 0.0
        self.tf: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

        if not documents:
            return

        for _, _, text in documents:
            tokens = self._tokenize(text)
            self.doc_len.append(len(tokens))
            self.tf.append(self._term_frequency(tokens))
            for term in set(tokens):
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        for term, df in self.doc_freq.items():
            self.idf[term] = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return TOKEN_RE.findall(text.lower())

    @staticmethod
    def _term_frequency(tokens: Sequence[str]) -> Dict[str, int]:
        stats: Dict[str, int] = {}
        for token in tokens:
            stats[token] = stats.get(token, 0) + 1
        return stats

    def _score(self, query_tokens: Sequence[str], doc_index: int) -> float:
        if not query_tokens:
            return 0.0
        doc_tokens = self.documents[doc_index][2]
        tokens = self._tokenize(doc_tokens)
        doc_len = max(1, len(tokens))
        k1 = 1.5
        b = 0.75
        score = 0.0
        qf = {}
        for token in query_tokens:
            qf[token] = qf.get(token, 0) + 1
        for token, freq in qf.items():
            if token not in self.idf:
                continue
            tf = self.tf[doc_index].get(token, 0)
            numerator = self.idf[token] * tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / max(1.0, self.avgdl))
            score += numerator / denominator * freq
        return score

    def search(self, query: str, top_k: int = 8) -> List[Tuple[str, str, str, float]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        ranked: List[Tuple[float, Tuple[str, str, str, float]]] = []
        for i, (path, title, text) in enumerate(self.documents):
            score = self._score(query_tokens, i)
            if score > 0:
                ranked.append((score, (path, title, text, score)))
        ranked.sort(key=lambda x: x[0], reverse=True)
        top = [item[1] for item in ranked[:top_k]]
        return top


class PolicyCorpus:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.documents: List[Dict[str, Any]] = []
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: BM25Index = BM25Index([])
        self.embeddings_client: OpenAI | None = None
        self._load_documents()

    def _load_documents(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"Corpus directory not found: {self.root}")

        for file_path in sorted(self.root.rglob("*.md")):
            if file_path.name.startswith("."):
                continue
            relative_path = file_path.relative_to(self.root.parent).as_posix()
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            chunks = self._chunk_markdown(file_path, text)
            for chunk in chunks:
                self.chunks.append(chunk)
                self.documents.append({
                    "path": relative_path,
                    "title": chunk["title"],
                    "text": chunk["text"],
                })

        bm25_docs = [(entry["path"], entry["title"], entry["text"]) for entry in self.documents]
        self.bm25 = BM25Index(bm25_docs)
        self._initialize_embeddings_client()

    @staticmethod
    def _chunk_markdown(file_path: Path, text: str) -> List[Dict[str, Any]]:
        normalized = text.strip()
        if not normalized:
            return []
        relative = file_path.relative_to(file_path.parents[2]).as_posix() if len(file_path.parents) >= 2 else file_path.name
        title = file_path.stem.replace("-", " ").replace("_", " ").title()
        sections: List[Dict[str, Any]] = []
        lines = text.splitlines()
        buffer: List[str] = []
        current_heading = title

        for raw_line in lines:
            line = raw_line.strip()
            if line.startswith("#"):
                if buffer:
                    sections.append({"title": current_heading, "text": "\n".join(buffer).strip()})
                    buffer = []
                current_heading = line.lstrip("#").strip() or current_heading
                buffer.append(line)
            else:
                buffer.append(line)

        if buffer:
            sections.append({"title": current_heading, "text": "\n".join(buffer).strip()})

        if not sections:
            sections.append({"title": title, "text": normalized[:1500]})

        chunk_list: List[Dict[str, Any]] = []
        for idx, section in enumerate(sections):
            content = section["text"].strip()
            if not content:
                continue
            chunk_list.append({
                "title": section["title"],
                "text": content,
                "section": section["title"],
                "path": relative,
                "chunk_index": idx,
            })
        return chunk_list

    def _initialize_embeddings_client(self) -> None:
        if not has_model_config():
            self.embeddings_client = None
            return
        self.embeddings_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    def _get_embedding(self, text: str) -> List[float] | None:
        if self.embeddings_client is None:
            return None
        try:
            response = self.embeddings_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
            return response.data[0].embedding
        except Exception:
            return None

    def search(self, query: str, top_k: int = 8) -> List[RetrievalHit]:
        bm25_hits = self.bm25.search(query, top_k=top_k)
        q_embedding = self._get_embedding(query)
        dense_hits: List[Tuple[float, RetrievalHit]] = []

        if q_embedding is not None:
            for chunk in self.chunks:
                candidate_text = chunk["text"]
                embed = self._get_embedding(candidate_text)
                if embed is None:
                    continue
                sim = self._cosine_similarity(q_embedding, embed)
                if sim > 0:
                    dense_hits.append((sim, RetrievalHit(
                        file_path=chunk["path"],
                        title=chunk["title"],
                        content=candidate_text[:1200],
                        score=sim,
                        section=chunk.get("section", ""),
                        chunk_index=chunk.get("chunk_index", 0),
                        source=chunk["path"],
                    )))

        combined: Dict[str, RetrievalHit] = {}
        for path, title, text, score in bm25_hits:
            key = f"{path}:{title}"
            combined[key] = RetrievalHit(file_path=path, title=title, content=text[:1200], score=float(score), source=path)

        for sim, hit in dense_hits:
            key = f"{hit.file_path}:{hit.title}"
            if key in combined:
                combined[key].score = max(combined[key].score, float(sim))
            else:
                combined[key] = hit

        final_hits = sorted(combined.values(), key=lambda hit: hit.score, reverse=True)[:top_k]
        if not final_hits:
            return [RetrievalHit(file_path="", title="No direct match", content=query, score=0.0)]
        return final_hits

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
