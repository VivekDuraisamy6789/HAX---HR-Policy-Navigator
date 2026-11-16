# Technical report outline (12 pages) — what is measured vs what still needs a live endpoint

| § | Pages | Content | Evidence status |
|---|---|---|---|
| 1 | 0.5 | Problem, constraints (≤8B, hosted, no weights) | — |
| 2 | 1 | Task spec, refusal policy, eval-set design (`docs/specification.md`) | written; **per-question golds still templated (12 distinct)** |
| 3 | 1.5 | Prompting baseline + 3 strategies; how the small model breaks | harness ready; **needs `run_e2e_eval.py` on real endpoint** |
| 4 | 2 | RAG: chunking × retrieval ablation, reranker, recall@k vs answer quality | **measured on full corpus with offline LSA dense (`docs/retrieval_results.md`); re-run with real embedder + cross-encoder** |
| 5 | 1.5 | Agent architecture (`docs/agent-design.md`), tool schema, loop/budget guards | implemented, 50 tests pass |
| 6 | 2 | Evaluation: accuracy, completion, tool calls, cost/latency, failure taxonomy | **pending real run** |
| 7 | 1 | Observability: trace dashboard screenshots | pending real queries |
| 8 | 1.5 | Responsible AI: injection suite, PII, refusals, HITL | guard unit tests done; **model-side injection results pending** |
| 9 | 1 | Limitations, threats to validity, future work | see README "Read this before trusting any number" |
