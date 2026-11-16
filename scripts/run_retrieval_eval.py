"""Retrieval-only evaluation: recall@k / MRR at document level, chunking x mode ablation.

    python scripts/run_retrieval_eval.py [--eval evaluation_set.json] [--out eval_runs/retrieval.json]
Run this *before* judging the generator: with a weak 8B model, retrieval misses are the
first thing to rule out.
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hr_policy_navigator.config import DATA_ROOT
from hr_policy_navigator.retrieval import PolicyCorpus

KS = (1, 3, 5, 8)


def doc_ranking(hits):
    seen, order = set(), []
    for h in hits:
        if h.file_path not in seen:
            seen.add(h.file_path); order.append(h.file_path)
    return order


def bootstrap_ci(per_q, themes, base=None, iters=2000, seed=0):
    """95% CI of mean(per_q) (or of mean(per_q - base) when `base` is given), resampling *themes*
    (clusters) rather than questions: paraphrases of one topic are not independent evidence."""
    import random
    rng = random.Random(seed)
    groups = {}
    for i, t in enumerate(themes):
        groups.setdefault(t, []).append(i)
    keys = list(groups)
    vals = [x - (base[i] if base else 0) for i, x in enumerate(per_q)]
    means = []
    for _ in range(iters):
        idx = [i for k in (rng.choice(keys) for _ in keys) for i in groups[k]]
        means.append(sum(vals[i] for i in idx) / len(idx))
    means.sort()
    return [round(means[int(.025 * iters)], 3), round(means[int(.975 * iters)], 3)]


def evaluate(corpus, items, mode, pool=30):
    rec = {k: 0.0 for k in KS}; all_rec = {k: 0.0 for k in KS}; mrr = 0.0; n = 0; misses = []; per_q5 = []; themes = []
    for it in items:
        gold = set(it["expected_sources"])
        if not gold:
            continue
        n += 1
        order = doc_ranking(corpus.search(it["question"], top_k=40, mode=mode, pool=pool, max_per_doc=40))
        first = next((r for r, p in enumerate(order) if p in gold), None)
        mrr += 1 / (first + 1) if first is not None else 0
        per_q5.append(1.0 if first is not None and first < 5 else 0.0); themes.append(it.get("theme", it["id"]))
        for k in KS:
            got = gold & set(order[:k])
            rec[k] += 1.0 if got else 0.0          # any-hit recall (>=1 gold doc in top-k)
            all_rec[k] += len(got) / len(gold)      # fraction of gold docs retrieved
        if first is None or first >= 5:
            misses.append({"id": it["id"], "q": it["question"], "gold": sorted(gold), "top3": order[:3]})
    f = lambda d: {f"@{k}": round(v / n, 3) for k, v in d.items()}
    return {"n": n, "n_themes": len(set(themes)), "hit_rate": f(rec), "doc_recall": f(all_rec), "mrr": round(mrr / n, 3),
            "hit5_ci95": bootstrap_ci(per_q5, themes), "_per_q5": per_q5, "_themes": themes, "misses": misses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="evaluation_set.json")
    ap.add_argument("--out", default="eval_runs/retrieval.json")
    ap.add_argument("--chunkings", default="fixed_200,fixed_500,heading,heading_cap")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--reranker", default="lexical", help="lexical | cross_encoder (needs sentence-transformers + model download)")
    a = ap.parse_args()
    items = json.loads(Path(a.eval).read_text(encoding="utf-8"))
    items = [i for i in items if i.get("expected_sources")]
    results = {}
    for ch in a.chunkings.split(","):
        t = time.time()
        corpus = PolicyCorpus(DATA_ROOT, chunking=ch, use_dense=not a.no_dense, reranker=a.reranker)
        for note in corpus.notes:
            print('  NOTE:', note)
        build = round(time.time() - t, 1)
        modes = ["bm25"] + ([] if corpus.matrix is None else ["dense"]) + ["hybrid"] + ["hybrid_rerank"]
        for m in modes:
            r = evaluate(corpus, items, m)
            if m != "bm25" and f"{ch}|bm25" in results:   # paired difference vs BM25 on the same questions
                b = results[f"{ch}|bm25"]
                r["hit5_delta_vs_bm25_ci95"] = bootstrap_ci(r["_per_q5"], r["_themes"], base=b["_per_q5"])
            r.update(reranker=corpus.reranker_name, notes=corpus.notes,chunks=len(corpus.chunks), docs=len(corpus.docs), build_s=build,
                     embedder=corpus.embedder.name if corpus.embedder else None)
            results[f"{ch}|{m}"] = r
            print(f"{ch:12s} {m:14s} hit@1={r['hit_rate']['@1']:.2f} hit@3={r['hit_rate']['@3']:.2f} "
                  f"hit@5={r['hit_rate']['@5']:.2f} docrec@5={r['doc_recall']['@5']:.2f} mrr={r['mrr']:.2f}"
                  f"  hit@5 95%CI={r['hit5_ci95']}" + (f" Δvs bm25={r['hit5_delta_vs_bm25_ci95']}" if "hit5_delta_vs_bm25_ci95" in r else "") + f"  ({r['chunks']} chunks)")
    for r in results.values():
        r.pop("_per_q5", None); r.pop("_themes", None)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
