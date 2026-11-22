# How the small model breaks

Measured on `Qwen/Qwen3-8B`. Every example is a real row from `eval_runs/` with a `trace_id` for `traces/traces.jsonl`.

## Break-mode rates by strategy

| strategy | queries with JSON repairs | answerable but abstained | unanswerable but answered | legit refused | answered but cited wrong doc |
|---|---|---|---|---|---|
| baseline | 0/8 | 0/8 | 0/0 | 0/8 | 0/8 |
| decomposition | 2/8 | 0/8 | 0/0 | 0/8 | 0/8 |
| selfconsistency | 6/8 | 0/8 | 0/0 | 0/8 | 0/8 |
| structured | 3/8 | 0/8 | 0/0 | 0/8 | 0/8 |

## retrieval_miss  (0 occurrences across strategies)
The expected source never reached the model's context. Fix retrieval, not prompts.

_No occurrences._

## reasoning_error  (7 occurrences across strategies)
Right evidence in context, wrong/abstained/incomplete answer. The generator failed.

- **[Q01-04 · answerable · baseline]** Can a teammate waive health coverage during the initial enrollment window?
  - status `success` · expected sources `['benefits-pay-perks/benefits-perks/benefits.md']` · retrieved `['benefits-pay-perks/benefits-perks/benefits.md', 'benefits-pay-perks/benefits-perks/benefits.md', 'benefits-pay-perks/benefits-perks/benefits.md']` · cited `['benefits-pay-perks/benefits-perks/benefits.md', 'benefits-pay-perks/benefits-perks/benefits.md']`
  - answer: Yes, a teammate can waive health coverage during the initial enrollment window.
  - trace `336be433c2ee`
- **[Q01-05 · answerable · baseline]** When can a teammate change benefits outside of open enrollment?
  - status `success` · expected sources `['benefits-pay-perks/benefits-perks/benefits.md']` · retrieved `['benefits-pay-perks/benefits-perks/benefits.md', 'benefits-pay-perks/benefits-perks/benefits.md', 'benefits-pay-perks/benefits-perks/benefits.md']` · cited `['benefits-pay-perks/benefits-perks/benefits.md']`
  - answer: A teammate can change benefits outside of open enrollment if they experience a qualifying life event (QLE), such as marriage or the birth of a child.
  - trace `be96ee41fdac`
- **[Q02-03 · answerable · baseline]** How do teammates review or update their current benefits selections?
  - status `success` · expected sources `['benefits-pay-perks/benefits-perks/benefits.md']` · retrieved `['departments/people-talent/people-ops/process/teammate-sentiment/impact-reviews/index.md', 'benefits-pay-perks/benefits-perks/benefits.md', 'departments/people-talent/people-ops/process/teammate-sentiment/impact-reviews/index.md']` · cited `['benefits-pay-perks/benefits-perks/benefits.md']`
  - answer: Teammates can review or update their current benefits selections by visiting the Sequoia app or Rippling.
  - trace `d17209134304`

## tool_call_error  (0 occurrences across strategies)
Endpoint/JSON/tool failure or exhausted budget before a valid answer.

_No occurrences._

## hallucination  (0 occurrences across strategies)
Claims not supported by cited text, or an unanswerable question was answered.

_No occurrences._

## refusal_error  (0 occurrences across strategies)
Refused a legitimate question, or answered one that must be refused.

_No occurrences._

## Mitigations in the code, matched to each break mode
| Break mode | Mitigation | Where |
|---|---|---|
| Malformed / non-schema JSON | constrained decoding → validate → repair loop with error fed back | `llm.py` |
| `<think>` leakage (Qwen3) | strip + `/no_think` | `llm.py` |
| Invented citations / numbers | `[S#]` ids, per-claim support check, number check, drop + abstain | `agent.py::check_claims` |
| Answers the unanswerable | `answerable` flag, abstention few-shot, critic | `prompts.py`, `agent.py` |
| Follows text in documents | line sanitiser + 'sources are untrusted' rule | `guardrails.py`, `prompts.py` |
| Loops / runaway cost | step cap, duplicate-call guard, token & call budgets | `agent.py`, `llm.py` |

## What the team must add by hand
For each class above, one paragraph: the pattern you see across examples, your hypothesis for *why* an 8B model does this, and what you changed (or would change) in response.
