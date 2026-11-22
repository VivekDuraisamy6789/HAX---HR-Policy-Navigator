# Week 3 Deliverable: Mentor Progress Update Email

**Course:** Executive Post Graduate Programme in Generative AI and Agentic AI  
**Track:** Track B — API-Side Capstone (Hosted $\le$8B Inference API)  
**Project:** Multi-Agent Operations Copilot (HR Policy Navigator)  
**Target Milestone:** Week 3 — Multi-Agent Architecture, Error Recovery & Observability  

---

## 📧 Email Draft

**To:** `[Mentor Name / Email]`  
**Subject:** `[Capstone Track B] Week 3 Deliverables: Multi-Agent Architecture, State & Observability`

Dear [Mentor Name],

I hope this email finds you well.

Our team has completed the **Week 3 deliverables** for our Capstone project: **HR Policy Navigator (Track B — API-Side Capstone on Hosted $\le$8B API)**.

In Week 2, we established an enterprise hybrid retrieval layer achieving 89.1% Hit@5. This week, we focused on building the agentic control loop, self-healing error recovery mechanisms, and full execution observability tailored specifically to overcome the reasoning limitations of our hosted edge-class model (`Qwen/Qwen3-8B`).

Below is an executive summary of our architecture, empirical results, and failure taxonomy:

---

### 1. Key Accomplishments

#### A. Multi-Agent Control Loop (Planner $\to$ Executor $\to$ Critic)
* **Pre-Guard:** Intercepts regex patterns to screen queries for PII redaction and immediately deflect out-of-scope/consequential requests without spending LLM tokens.
* **Planner Agent:** Executes schema-constrained decoding to classify intent (`policy_lookup`, `multi_part`, `out_of_scope`, `consequential_action`), rewrite queries resolving multi-turn session history, and decompose complex multi-part queries into $\le 3$ targeted sub-queries. Includes graceful degradation to heuristic parsing if JSON schema fails.
* **Executor Agent:** Dispatches sub-queries to retrieval tools (`search_policy`, `read_section`, `escalate_to_human`), incorporating loop detection (blocking duplicate queries) and automated reformulation when reranked retrieval confidence falls below 0.22.
* **Critic Agent:** Verifies citation fidelity on every factual claim, enforcing $\ge 60\%$ lexical token overlap against cited chunks, verifying that all numbers and dates exist in the evidence, and pruning ungrounded claims or triggering context expansion.

#### B. Error Recovery & Resilience for Small Models ($\le$8B)
* **Constrained Decoding Cascade:** Automatically falls back across decoding modes (`json_schema` $\to$ `guided_json` $\to$ `json_object` $\to$ `none`) to handle API provider quirks.
* **Automated JSON Schema Repair:** Detects malformed JSON (trailing commas, unescaped quotes, Markdown backtick wrappers), parses line-by-line, and executes targeted repair passes (rescuing 49 runtime syntax errors during benchmark evaluation).
* **Execution & Budget Guardrails:** Hard caps enforced at runtime: `MAX_QUERY_TOKENS=14,000`, `MAX_STEPS=8`, per-chunk length clamps, and regex-based sanitization of Qwen3 `<think>...</think>` reasoning tokens.

#### C. Full Observability & Test Coverage
* **Comprehensive Test Suite:** **60 passing unit tests** across core guardrails, chunking, retrieval fusion, schema repair, rate limiting, and tool execution (`pytest tests`).
* **Persistent Execution Tracing:** End-to-end execution logs captured in `traces/traces.jsonl` recording per-span latency, token budgets, tool invocations, and JSON repairs, aggregated live via `scripts/trace_summary.py` and visualized in the Streamlit UI.

---

### 2. Failure Taxonomy & Root Cause Attribution
In compliance with the syllabus, we classified all agent failure events into 5 distinct categories:
1. `retrieval_miss`: Gold document not present in top-$k$ retrieved chunks.
2. `reasoning_error`: Gold document retrieved in context, but model deduced an incorrect answer or prematurely abstained.
3. `tool_call_error`: Unrecoverable JSON syntax errors or invalid tool parameter types.
4. `hallucination`: Model generated factual claims lacking source evidence or answered an unanswerable question.
5. `refusal_error`: Model improperly refused a valid in-scope query or complied with a forbidden consequential action.

---

### 3. Empirical Multi-Agent Benchmark Evaluation
*Evaluated on hosted `Qwen/Qwen3-8B` across our frozen benchmark suite:*

| Agent Strategy | Task Accuracy | Task Completion | $p_{50}$ Latency | Avg LLM Calls | Avg Cost / Query | Primary Failure Mode |
|---|---|---|---|---|---|---|
| **Simple RAG Baseline** | 62.5% | 100% | 8.35s | 1.00 | \$0.000096 | Reasoning error (3/8 missed) |
| **Structured Few-Shot Agent** | 75.0% | 100% | 9.76s | 2.62 | \$0.000148 | Reasoning error (2/8 missed) |
| **Self-Consistency Agent** | 75.0% | 100% | 22.19s | 5.38 | \$0.000389 | Latency explosion; marginal gain |
| **Decomposition Agent** | **100.0%** | **100%** | **5.94s** | **2.88** | **\$0.000191** | **Zero failures (8/8 correct)** |

*Key Empirical Insights:*
* **Decomposition Prevents Small-Model Saturation:** Small 8B models struggle when forced to plan, retrieve, and synthesize multi-part policies in a single context window. Breaking queries into targeted sub-questions achieves **100% accuracy** at lower median latency (5.94s) than standard structured prompting.
* **Self-Consistency is Inefficient on Small APIs:** Sampling 3 paths tripled token consumption and latency ($p_{50}=22.2$s) without exceeding the accuracy of single-turn structured prompting (75.0%).

---

### 4. Live Execution Tracing Summary
Telemetry aggregated across 55 real end-to-end queries (`traces/traces.jsonl` via `scripts/trace_summary.py`):
* **Median Latency ($p_{50}$):** 9.76 seconds (including multi-step retrieval and critic verification).
* **95th Percentile Latency ($p_{95}$):** 28.31 seconds (for complex multi-sub-query queries).
* **Average Tool Calls:** 0.96 per query.
* **Auto-Repaired JSON Schemas:** 49 syntax rescues (zero unhandled JSON parser crashes).
* **Execution Outcomes:** 48 successes, 3 policy refusals, 3 conservative abstentions, and only 1 upstream API timeout.

---

### 5. Deliverables & Repository Links
* **Repository (Branch: Amit):** [GitHub Repository](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/tree/Amit)
* **Agent Architecture Specification:** [`docs/agent-design.md`](agent-design.md) · [GitHub Link](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/blob/Amit/docs/agent-design.md)
* **Failure Characterization & Taxonomy:** [`docs/failure_characterization.md`](failure_characterization.md) · [GitHub Link](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/blob/Amit/docs/failure_characterization.md)
* **Strategy Comparison Report:** [`docs/strategy_comparison.md`](strategy_comparison.md) · [GitHub Link](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/blob/Amit/docs/strategy_comparison.md)
* **Evaluation Summary Telemetry:** [`eval_runs/summary.json`](../eval_runs/summary.json) · [GitHub Link](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/blob/Amit/eval_runs/summary.json)
* **Trace Analytics Script:** [`scripts/trace_summary.py`](../scripts/trace_summary.py) · [GitHub Link](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/blob/Amit/scripts/trace_summary.py)

We look forward to demonstrating the working agent loop during our upcoming sync.

Best regards,  
**Amit Bhardwaj** (and Team)  
*Capstone Track B — Group [Number]*  
*Executive Post Graduate Programme in Generative AI and Agentic AI*

---

## 📌 Appendix: Anticipated Mentor Questions & Answers (Defense Guide)

### Q1: Why did you choose a dedicated Critic agent instead of letting the generator evaluate itself in a single turn?
**Answer:**  
Small 8B parameter models exhibit high self-confirmation bias; when asked to generate and self-critique in the same context, they consistently validate their own hallucinations. Decoupling the Critic into a dedicated verification step allows us to run deterministic, non-LLM algorithmic checks: calculating lexical token overlap ($\ge 60\%$) between individual claims and cited passages, and verifying that every numerical figure or date exists literally in the retrieved text. If validation fails, the Critic triggers targeted context expansion (`read_section`) or strips the unsupported claim.

### Q2: Why did Decomposition Prompting outperform Self-Consistency while being substantially faster?
**Answer:**  
Self-consistency relies on majority voting across independent stochastic samples. On an 8B model, if the underlying reasoning prompt is too broad or contains distractor noise, all sampled paths often converge on the same plausible misconception, while tripling latency ($p_{50}=22.2$s) and token costs. Decomposition eliminates reasoning overload by transforming complex queries into focused atomic questions (e.g., *"What is parental leave duration?"* and *"What is parental leave pay percentage?"*). Each sub-query retrieves precise context, yielding 100% accuracy at a median latency of only 5.94s.

### Q3: How does your system guarantee that malformed JSON from an 8B model will not crash production workflows?
**Answer:**  
We implement a three-tier defense: First, we request structured outputs via API-level constrained decoding (`json_schema`). Second, if the API provider returns malformed syntax (e.g., conversational preambles or unescaped quotes), our custom parser executes a repair routine (`hr_policy_navigator/schema_repair.py`) that extracts JSON substrings, strips Markdown delimiters, and corrects trailing commas. Third, if schema parsing still fails after targeted retries, the Planner agent falls back to a deterministic heuristic plan rather than throwing an unhandled exception. Across our benchmark, 49 syntax errors were auto-repaired with zero server crashes.
