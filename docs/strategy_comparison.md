# Prompting-strategy comparison

Model: `Qwen/Qwen3-8B` · n = 8 queries per strategy (evaluation_set.json + evaluation_set_adversarial.json). Correctness definition: see docstring of `scripts/run_e2e_eval.py` (lexical gold recall is a proxy; spot-check by hand).

| strategy | accuracy | task_completion_rate | retrieval_hit_in_context | avg_tool_calls | avg_llm_calls | avg_tokens | avg_cost_usd | p50_latency_ms | p95_latency_ms | json_repairs |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.625 | 1.0 | 1.0 | 1.0 | 1.0 | 1799 | 9.6e-05 | 8353.8 | 10950.4 | 0 |
| decomposition | 1.0 | 1.0 | 1.0 | 1.25 | 2.88 | 3474 | 0.000191 | 5940.6 | 34331.1 | 5 |
| selfconsistency | 0.75 | 1.0 | 1.0 | 1.0 | 5.38 | 6985 | 0.000389 | 22190.9 | 28310.7 | 13 |
| structured | 0.75 | 1.0 | 1.0 | 1.0 | 2.62 | 2596 | 0.000148 | 9757.0 | 21954.7 | 7 |

## Accuracy by query type (correct/total)

| type | baseline | decomposition | selfconsistency | structured |
|---|---|---|---|---|
| answerable | 5/8 | 8/8 | 6/8 | 6/8 |

## Failure taxonomy (count of incorrect queries)

| strategy | retrieval_miss | reasoning_error | tool_call_error | hallucination | refusal_error |
|---|---|---|---|---|---|
| baseline | 0 | 3 | 0 | 0 | 0 |
| decomposition | 0 | 0 | 0 | 0 | 0 |
| selfconsistency | 0 | 2 | 0 | 0 | 0 |
| structured | 0 | 2 | 0 | 0 | 0 |

Highest accuracy: **decomposition**. Differences of a few queries are within noise at this n; re-run with `temperature` fixed and report the spread before claiming a winner.

## Interpretation (team to write after reading the tables)
- Did few-shot structure beat baseline, and by how much?
- Did self-consistency's extra cost buy accuracy, or only latency?
- Did decomposition help multi-part questions or add errors?
