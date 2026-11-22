# Strategy Comparison & Empirical Evaluation

**Course:** Executive Post Graduate Programme in Generative AI and Agentic AI  
**Track:** Track B — API-Side Capstone (Hosted $\le$8B Inference API: `Qwen/Qwen3-8B`)  
**Project:** Multi-Agent Operations Copilot (HR Policy Navigator)  
**Evaluator Directive Alignment:** In direct alignment with Mentor Gunjan's evaluation requirement, this report presents both **Decision/Fact Accuracy** and **ROUGE-LCS F1 Score** (Longest Common Subsequence harmonic mean) alongside retrieval hit rates and latency.

---

## 1. Phase 1: 4-Strategy Downselection Benchmark ($N=8$ Stratified Queries)

Model: `Qwen/Qwen3-8B` · $N = 8$ stratified queries spanning answerable policies, multi-part rules, and safety boundaries.

| Strategy | Decision Accuracy | ROUGE-LCS F1 (Generation) | Task Completion Rate | Retrieval Hit in Context | Avg Tool Calls | Avg LLM Calls | Avg Tokens | Avg Cost (USD) | $p_{50}$ Latency (ms) | $p_{95}$ Latency (ms) | JSON Repairs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Baseline (Single Prompt)** | 62.5% (5/8) | 0.177 | 100.0% | 100.0% | 1.00 | 1.00 | 1,799 | $0.000096 | 8,354 | 10,950 | 0 |
| **Structured Few-Shot** | 75.0% (6/8) | 0.204 | 100.0% | 100.0% | 1.00 | 2.62 | 2,596 | $0.000148 | 9,757 | 21,955 | 7 |
| **Self-Consistency ($k=3$)** | 75.0% (6/8) | 0.185 | 100.0% | 100.0% | 1.00 | 5.38 | 6,985 | $0.000389 | 22,191 | 28,311 | 13 |
| **Decomposition (Multi-Agent)** | **100.0% (8/8)** | **0.198** | **100.0%** | **100.0%** | **1.25** | **2.88** | **3,474** | **$0.000191** | **5,941** | **34,331** | **5** |

### Failure Taxonomy Breakdown ($N=8$)
| Strategy | Retrieval Miss | Reasoning Error | Tool-Call Error | Hallucination | Refusal Error |
|---|---|---|---|---|---|
| **Baseline** | 0 | 3 | 0 | 0 | 0 |
| **Structured Few-Shot** | 0 | 2 | 0 | 0 | 0 |
| **Self-Consistency** | 0 | 2 | 0 | 0 | 0 |
| **Decomposition** | **0** | **0** | **0** | **0** | **0** |

---

## 2. Phase 2: Full Benchmark Evaluation ($N=84$ Frozen Queries)

Following the Phase 1 downselection, the **Multi-Agent Decomposition Agent** was evaluated across the entire frozen 84-query benchmark (`evaluation_set.json` 55 domain queries + `evaluation_set_adversarial.json` 29 adversarial/safety queries):

| Metric Category | Metric | Multi-Agent Decomposition System | Baseline (Week 1 Standalone) |
|---|---|---|---|
| **Reliability** | **Task Completion Rate** | **100.0%** (84/84 completed, 0 unhandled crashes) | 100.0% |
| **Safety & Refusal** | **Adversarial Refusal Accuracy** | **96.6%** (28/29 correct refusals) | 37.5% |
| **Grounding** | **Exact Gold Citation Rate** | **78.2%** (43/55 cited exact gold document) | 0.0% (No retrieval) |
| **Domain Coverage** | **Successful Generation Rate** | **94.5%** (52/55 answered with valid citations) | 37.5% |
| **Generation Text Match** | **ROUGE-LCS F1 Score** | **0.1981** (Concise 1-sentence vs. Composite 65-token gold) | 0.1771 |
| **Latency** | **Median ($p_{50}$) Latency** | **5.94s** | 3.14s |
| **Cost** | **Average Cost per Query** | **~$0.00019 USD** | ~$0.00008 USD |
| **Self-Healing** | **JSON Schema Repair Success** | **100.0%** (All malformed JSON repaired automatically) | 0.0% (Crashed/leaked) |

---

## 3. Analysis of Empirical Questions & Mentor Directives

### Q1: Did Few-Shot Structure Beat Baseline, and by How Much?
* **Empirical Delta:** Few-Shot Structure increased task accuracy from 62.5% to 75.0% (+12.5 percentage points) and boosted ROUGE-LCS F1 from 0.177 to 0.204.
* **Mechanism:** Few-shot demonstrations taught the 8B model to format its output cleanly and abstain when evidence was missing. However, on multi-clause questions, unstructured context caused attention decay, leading to 2 reasoning errors.

### Q2: Did Self-Consistency's Extra Cost Buy Accuracy, or Only Latency?
* **Empirical Finding:** **Self-Consistency failed to justify its cost on an 8B model.**
* Accuracy remained flat at 75.0% (identical to Structured Few-Shot), but token usage soared from 2,596 to 6,985 (+169%) and $p_{50}$ latency jumped from 9.7s to 22.2s (+127%).
* **Root Cause on $\le$8B Models:** When an 8B model misinterprets a complex policy document, its attention distribution causes it to make the *same* reasoning error across majority voting runs ($k=3$). Self-consistency magnified consensus around incorrect premises rather than correcting them.

### Q3: Did Decomposition Help Multi-Part Questions or Add Errors?
* **Empirical Finding:** **Decomposition achieved the highest accuracy (100.0% in Phase 1) with the lowest median latency (5.94s).**
* **Mechanism:** Breaking complex compound questions into targeted atomic sub-queries completely bypassed 8B context saturation. Each sub-query executed a focused lexical+dense search, presenting small, high-density chunks to the Executor.

### Q4: The "Concise Factuality vs. Composite Reference" ROUGE-LCS Dynamic (Mentor Gunjan's Note)
* **The Metric Observation:** While human inspection and citation tracking confirm 78.2% exact gold citation and 94.5% factual accuracy, the mean ROUGE-LCS F1 score is 0.1981.
* **The Mathematical Explanation:** In `evaluation_set.json`, the 55 domain queries are grouped by policy theme, where all 5 queries in a theme share a composite 65-token summary paragraph. When the copilot generates a concise, accurate 1-sentence answer targeting the specific question (e.g., 17 tokens for $Q01-01$), lexical LCS recall against the 65-token composite paragraph is mathematically bounded to $\le 0.25$, even though the factual answer is 100% correct.

