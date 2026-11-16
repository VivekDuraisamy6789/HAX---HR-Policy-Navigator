# Agent design

## Control flow (`agent.py`)
1. **Guard** – `screen_query` (regex intent patterns) → refuse / ticket without spending tokens. PII redacted first.
2. **Planner** – one schema-validated LLM call: `intent ∈ {policy_lookup, multi_part, needs_clarification, out_of_scope, consequential_action}`, standalone `rewritten_query` (resolves follow-ups from session memory), ≤3 `sub_queries`. If JSON fails after repairs → heuristic plan (the system degrades, it does not fail).
3. **Executor** – loop over (sub-)queries calling `search_policy`. Guards: `max_steps`; duplicate-call detection on the normalised query (infinite-loop guard); one reformulation when the top reranked score < 0.22; one simplified retry on tool error. Evidence is numbered S1..S8, de-duplicated, and every chunk is passed through `sanitize_chunk` (indirect-injection defence).
4. **Generator** – strategies: `baseline` (zero-shot, no planner/critic), `structured` (2 few-shot turns + schema), `selfconsistency` (3 samples @T=0.7, "reasoning" field, vote on answerability then source agreement then claim support), `decomposition` (answer each sub-question, then compose from notes).
5. **Critic** – per claim: source ids exist; ≥60 % of claim content tokens occur in cited text; every number in the claim occurs in the evidence. Failure → one revision using `read_section` to expand context, keep the better attempt; remaining unsupported claims are dropped; nothing left → abstain.
6. **State** – `SessionStore` JSON per session (last 6 turns, last sources); `handoffs.jsonl` review queue.

## Why these choices (small-model specifics)
* **Cite by id (`S2`), not path** – 8B models mangle long paths; ids are validated mechanically and mapped to paths afterwards.
* **Claims array with per-claim sources** – makes hallucination *checkable*; free-text answers are not.
* **`answerable` flag + few-shot abstention example** – cheapest reliable abstention signal for a weak model.
* **Constrained decoding with graceful downgrade** (`json_schema → guided_json → json_object → none`) plus validate → repair-with-error-message loop. Tool arguments are JSON-schema validated; tool exceptions become `ToolResult(ok=False)`.
* **Budgets** – `MAX_LLM_CALLS`, `MAX_QUERY_TOKENS` (hard `BudgetExceeded`), `MAX_STEPS`, per-chunk char cap.

## Failure taxonomy (assigned in `scripts/run_e2e_eval.py::judge`)
| Class | Rule |
|---|---|
| retrieval_miss | expected source not among retrieved evidence |
| reasoning_error | source retrieved but answer wrong / abstained / low gold recall |
| tool_call_error | `error_*` status, tool errors, unrecoverable JSON |
| hallucination | critic found unsupported claims, or answered an unanswerable question |
| refusal_error | refused a legitimate question, or answered one that must be refused |

## Observability
Every query writes one JSON trace line (`traces/traces.jsonl`): guard, planner, each tool call (args, ms, ok), each LLM call (stage, tokens, mode), critic, counters, result. Mirrored to Langfuse when keys are set. The Streamlit *Traces* tab and `scripts/trace_summary.py` aggregate latency percentiles, tool calls/query, repairs, status mix.
