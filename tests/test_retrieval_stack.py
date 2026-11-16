import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hr_policy_navigator.retrieval import CrossEncoderReranker, OpenAIEmbedder, PolicyCorpus
from run_retrieval_eval import bootstrap_ci


class FakeEmbClient:
    """OpenAI-like embeddings client: deterministic bag-of-letters vectors; can fail N times first."""
    def __init__(self, fail_first=0, wrong_count=False):
        self.calls, self.fail, self.wrong = 0, fail_first, wrong_count
        self.embeddings = SimpleNamespace(create=self.create)

    def create(self, model, input):
        self.calls += 1
        if self.calls <= self.fail:
            raise RuntimeError("503 overloaded")
        vecs = [[float(t.lower().count(c)) for c in "abcdefghijklmnopqrstuvwxyz"] for t in input]
        data = [SimpleNamespace(embedding=v, index=i) for i, v in enumerate(vecs)]
        return SimpleNamespace(data=data[:-1] if self.wrong else data[::-1])  # reversed: must be re-sorted by .index


def make_embedder(tmp_path, client):
    e = OpenAIEmbedder.__new__(OpenAIEmbedder)
    e.client, e.model, e.batch, e.name = client, "m", 2, "api:m"
    e.cache_path = tmp_path / "c.json"; e.cache = {}
    return e


def test_embedder_batches_caches_and_respects_index_order(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    c = FakeEmbClient()
    e = make_embedder(tmp_path, c)
    m = e.fit(["aaa", "bbb", "ccc"])
    assert m.shape[0] == 3 and c.calls == 2                      # 3 texts, batch size 2
    assert m[0].argmax() == 0 and m[1].argmax() == 1             # row order preserved despite reversed response
    e.fit(["aaa", "bbb", "ccc"]); assert c.calls == 2            # fully cached the second time
    assert json.loads(e.cache_path.read_text())                  # persisted


def test_embedder_retries_transient_errors(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    c = FakeEmbClient(fail_first=2); e = make_embedder(tmp_path, c)
    assert e.fit(["aaa"]).shape == (1, 26) and c.calls == 3


def test_embedder_rejects_wrong_vector_count(tmp_path):
    with pytest.raises(ValueError):
        make_embedder(tmp_path, FakeEmbClient(wrong_count=True)).fit(["a", "b"])


def test_corpus_falls_back_to_lsa_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    root = tmp_path / "c"; root.mkdir()
    (root / "a.md").write_text("# Leave\n" + "parental leave weeks " * 30)
    (root / "b.md").write_text("# Travel\n" + "flight hotel booking " * 30)
    broken = make_embedder(tmp_path, FakeEmbClient(fail_first=99))
    c = PolicyCorpus(root, embedder=broken)
    assert c.embedder.name == "lsa-offline" and any("failed" in n for n in c.notes)
    assert c.search("parental leave", 1)[0].file_path == "a.md"   # still works


def test_cross_encoder_reorders_by_model_scores(corpus):
    class Fake:
        def predict(self, pairs):             # prefers the passage mentioning "Sequoia"
            return [10.0 if "sequoia" in p[1].lower() else 0.0 for p in pairs]
    ce = CrossEncoderReranker("fake", model=Fake())
    c = PolicyCorpus(corpus.root, reranker="cross_encoder", reranker_model=ce)
    hits = c.search("health insurance provider", 3)
    assert c.reranker_name == "cross_encoder" and "sequoia" in hits[0].chunk.text.lower()


def test_cross_encoder_unavailable_falls_back_visibly(corpus, monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)    # simulate missing dependency
    c = PolicyCorpus(corpus.root, reranker="cross_encoder", use_dense=False)
    assert c.reranker_name == "lexical" and any("cross-encoder unavailable" in n for n in c.notes)


def test_bootstrap_ci_clusters_by_theme_and_paired_delta():
    per_q = [1, 1, 1, 1, 0, 0, 0, 0]
    themes = ["a", "a", "a", "a", "b", "b", "b", "b"]
    lo, hi = bootstrap_ci(per_q, themes)
    assert lo == 0.0 and hi == 1.0                  # two clusters -> maximally wide, as it should be
    lo2, hi2 = bootstrap_ci([1, 1, 1, 1], ["a", "b", "c", "d"], base=[1, 1, 1, 1])
    assert (lo2, hi2) == (0.0, 0.0)                 # identical systems: delta CI collapses to 0
