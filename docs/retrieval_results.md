> **Stale:** measured on the pre-cleanup 297-file corpus. See docs/synthetic_corpus.md for current numbers.

# Retrieval results (measured, full corpus)

**Setup:** all 297 markdown files (chunk counts per strategy in the table), dense retriever = offline LSA stand-in
(192-d TF-IDF + SVD), **not** a neural embedder; reranker = lexical feature scorer; n=60 questions from 12 topics.
Hit rate = at least one expected source document in the top-k documents. Only 95 of the 297 files are core HR policy
(`docs/corpus_classification.csv`), so ~half the index is distractor material, which is realistic for a handbook.
Intervals are cluster-bootstrap over **topics** (paraphrases of one topic are not independent). Reproduce: `python scripts/run_retrieval_eval.py`.

| chunking | mode | chunks | hit@1 | hit@3 | hit@5 | hit@5 95% CI | Δ hit@5 vs BM25 (95% CI) | MRR |
|---|---|---|---|---|---|---|---|---|
| fixed_200 | bm25 | 1989 | 0.6 | 0.8 | 0.85 | [0.75, 0.933] | - | 0.723 |
| fixed_200 | dense | 1989 | 0.517 | 0.783 | 0.8 | [0.633, 0.933] | [-0.117, 0.0] | 0.647 |
| fixed_200 | hybrid | 1989 | 0.55 | 0.85 | 0.867 | [0.75, 0.967] | [-0.033, 0.067] | 0.694 |
| fixed_200 | hybrid_rerank | 1989 | 0.7 | 0.85 | 0.867 | [0.767, 0.967] | [-0.033, 0.067] | 0.783 |
| fixed_500 | bm25 | 885 | 0.583 | 0.817 | 0.85 | [0.75, 0.95] | - | 0.703 |
| fixed_500 | dense | 885 | 0.5 | 0.75 | 0.867 | [0.733, 0.967] | [-0.05, 0.083] | 0.653 |
| fixed_500 | hybrid | 885 | 0.55 | 0.833 | 0.9 | [0.8, 0.983] | [0.0, 0.1] | 0.706 |
| fixed_500 | hybrid_rerank | 885 | 0.733 | 0.85 | 0.883 | [0.767, 0.983] | [0.0, 0.083] | 0.799 |
| heading | bm25 | 2650 | 0.633 | 0.783 | 0.833 | [0.7, 0.933] | - | 0.728 |
| heading | dense | 2650 | 0.4 | 0.633 | 0.733 | [0.583, 0.883] | [-0.183, -0.017] | 0.544 |
| heading | hybrid | 2650 | 0.533 | 0.8 | 0.833 | [0.667, 0.967] | [-0.067, 0.067] | 0.665 |
| heading | hybrid_rerank | 2650 | 0.65 | 0.85 | 0.883 | [0.783, 0.967] | [0.0, 0.1] | 0.755 |
| heading_cap | bm25 | 2668 | 0.6 | 0.8 | 0.867 | [0.767, 0.95] | - | 0.719 |
| heading_cap | dense | 2668 | 0.45 | 0.717 | 0.783 | [0.617, 0.933] | [-0.2, 0.033] | 0.587 |
| heading_cap | hybrid | 2668 | 0.55 | 0.767 | 0.833 | [0.667, 0.967] | [-0.133, 0.067] | 0.672 |
| heading_cap | hybrid_rerank | 2668 | 0.683 | 0.85 | 0.883 | [0.783, 0.967] | [-0.033, 0.067] | 0.78 |

## What this supports, and what it does not
- **Reranking helps ranking quality.** hit@1 rises from 0.58–0.63 (BM25) to 0.65–0.73 (hybrid+rerank), and MRR from 0.70–0.73 to 0.76–0.80 in every chunking.
  hit@1/MRR have no interval here (only hit@5 does), so treat the size of this gain as indicative.
- **hit@5 gains are small and borderline:** paired Δ vs BM25 for hybrid+rerank is [0.0, 0.1] on `heading` and `fixed_500`, [-0.033, 0.067] on `heading_cap` and `fixed_200`.
- **The LSA stand-in dense retriever hurts:** alone it is worse than BM25 (heading: Δ=[-0.183, -0.017]), and plain RRF fusion with it does not reliably help. A neural embedder may change this; it is untested.
- **No chunking strategy is shown to be better** at hit@5 (intervals overlap heavily). Choose on cost and citation granularity, or re-test with the real embedder.
- Not yet tested: neural embedder (`EMBEDDING_MODEL`), cross-encoder (`RERANKER=cross_encoder`).
