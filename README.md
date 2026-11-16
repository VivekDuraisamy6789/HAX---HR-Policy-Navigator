# HR Policy Navigator — multi-agent copilot on a hosted ≤8B model

Grounded Q&A over the Sourcegraph handbook/policy corpus using only an OpenAI-compatible chat endpoint
(shared ≤8B instruct model) plus an embeddings endpoint. All quality comes from retrieval, prompting,
tool design, orchestration and evaluation — no weights, logits or bigger model.

```
user ─▶ guardrails (injection / consequential / private / legal, PII redaction)
      ─▶ Planner  (LLM, schema-validated; heuristic fallback)  intent · rewritten query · sub-queries
      ─▶ Executor (tool loop: search_policy · read_section · escalate_to_human; step cap, loop guard, reformulate-on-weak-retrieval)
      ─▶ Generator (baseline | structured few-shot | self-consistency | decomposition) → claims with [S#] source ids
      ─▶ Critic   (schema, source ids, per-claim lexical support, numbers must appear in evidence; 1 revision with expanded context;
                   drops unsupported claims; abstains if nothing grounded remains)
      ─▶ answer + citations + trace        state: per-session memory, human-review queue
```

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                                      # fill OPENAI_BASE_URL / OPENAI_API_KEY / models
python -m pytest -q tests                                 # 50 tests, no endpoint needed (scripted fake model)
streamlit run streamlit_app.py                            # or: docker compose up --build
```

## Using a hosted Qwen-8B (Qwen2.5/Qwen3-8B-Instruct)
1. Fill `.env` (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`). Use a **separate embedding model** — a chat model cannot embed.
2. Run `python scripts/check_endpoint.py`: it tests chat, `<think>` leakage, each constrained-decoding mode and embeddings, and tells you which `CONSTRAINED_MODE` to set.
3. Qwen3's `<think>` blocks are stripped before JSON parsing, and `/no_think` is added automatically when the model name contains `qwen3` (`DISABLE_THINKING=true|false` to override).
If the host rejects structured output, the client downgrades `json_schema → guided_json → json_object → none` and the validate-and-repair loop covers the rest.

## Real embedder and cross-encoder reranker
- Dense: set `EMBEDDING_MODEL` (a real embedding model, not the chat model). Embeddings are batched, retried with backoff, cached on disk, and if the endpoint fails the system falls back to offline LSA **and says so** (`corpus.notes`, shown as a warning in the app sidebar).
- Reranker: `pip install sentence-transformers`, set `RERANKER=cross_encoder` (model: `CROSS_ENCODER_MODEL`, default `BAAI/bge-reranker-base`, downloaded from Hugging Face on first use). If it can't load, the lexical reranker is used and the fallback is reported.

## Evaluation commands (the numbers for the report)
| What | Command | Needs endpoint |
|---|---|---|
| Corpus size vs the 500–2,000 requirement | `python scripts/corpus_stats.py [--exclude trivy]` | no |
| Retrieval recall@k with bootstrap CIs, chunking × mode ablation (add `--reranker cross_encoder` for the neural reranker) | `python scripts/run_retrieval_eval.py` | no (uses offline LSA unless `EMBEDDING_MODEL` set) |
| Prompt-strategy comparison, accuracy, task completion, tool calls, latency, cost, failure taxonomy | `python scripts/run_e2e_eval.py --strategies baseline,structured,selfconsistency,decomposition` | **yes** |
| Indirect prompt-injection suite (poisoned docs) | `python scripts/run_injection_suite.py` | **yes** (`--mock` = plumbing only) |
| Strategy-comparison + failure-characterization write-ups (generated from the run above; refuses to run without real results) | `python scripts/make_report.py` → `docs/strategy_comparison.md`, `docs/failure_characterization.md` | no (reads `eval_runs/`) |
| Trace statistics | `python scripts/trace_summary.py` (or the app's *Traces* tab) | — |

`--mock` on the e2e/injection scripts runs a scripted fake model to prove the harness works; **those outputs are not model results.**

## Layout
`hr_policy_navigator/` — `chunking.py` · `retrieval.py` (BM25 + dense + RRF + reranker) · `llm.py` (constrained decoding, JSON repair, budgets) ·
`prompts.py` · `tools.py` · `agent.py` · `guardrails.py` · `state.py` · `tracing.py` · `serving.py` (rate limit, cache).
`docs/` — specification, agent design, report outline. `evaluation_set.json` (60) + `evaluation_set_adversarial.json` (29).

## Read this before trusting any number
* Retrieval results in `docs/retrieval_results.md` are on the **full 297-file corpus** but with the **offline LSA** stand-in for dense retrieval (not neural); bootstrap CIs show reranking improves hit@1/MRR but no configuration reliably beats BM25 at hit@5. Re-run with a real embedder and the cross-encoder.
* The original 60-question set has only **12 distinct gold answers** (5 paraphrases share one gold). Gold-token recall therefore cannot separate those paraphrases; the team should write per-question gold answers and key facts (owner: Domain spec & eval design).
* No model endpoint was available while this was built, so **no LLM accuracy, failure-taxonomy, or strategy-comparison numbers exist yet**. The harness is in place and tested.
* Corpus: **297 markdown files** (`scripts/corpus_stats.py`): 1,260 H1/H2 sections, 2,668 chunks. Files are *below* the brief's 500–2,000 documents; sections and chunks are within or above it. Say in the report which unit you count. Only **95 files are core HR policy, 56 HR-adjacent, 146 not HR** (85 security docs, 48 of them Trivy release notes; see `docs/corpus_classification.csv`, `scripts/classify_corpus.py`).


## Corpus update
Non-HR files were removed. Corpus = 151 real docs (95 core HR, 56 HR-adjacent) + **1,000 synthetic** HR docs (`sourcegraph_files/synthetic/`, see `docs/synthetic_corpus.md`). Synthetic docs are generated, consistent with real policy facts, and NOT real Sourcegraph policy; disclose this in the report. Use `HR_EXCLUDE=synthetic/` for real-only runs. Retrieval results: `eval_runs/retrieval_real_only.json`, `eval_runs/retrieval_with_synthetic.json`. Older numbers in `docs/retrieval_results.md` predate this change.
