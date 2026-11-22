# Week 4 Deliverable: Final Project Submission & Demo Invite Email

**Course:** Executive Post Graduate Programme in Generative AI and Agentic AI  
**Track:** Track B — API-Side Capstone (Hosted $\le$8B Inference API)  
**Project:** Multi-Agent Operations Copilot (HR Policy Navigator)  
**Target Milestone:** Week 4 — Final Submission, Responsible AI & Live Demo  

---

## 📧 Email Draft

**To:** `[Mentor Name / Evaluation Committee]`  
**Subject:** `[Capstone Track B Final Submission] HR Policy Navigator: Technical Report, Deployed System & Demo`

Dear [Mentor Name / Evaluation Committee],

On behalf of our team, I am delighted to present the **final project submission** for our Capstone project: **HR Policy Navigator — Multi-Agent Operations Copilot Built on a Hosted $\le$8B Model (Track B)**.

Over the past four weeks, our team has addressed the central challenge of Track B: **engineering production-grade reliability, strict citation grounding, and safety guardrails on an edge-class hosted open model (`Qwen/Qwen3-8B`) without model weight access or fine-tuning**. 

Below is an executive summary of our final deployed system, empirical achievements, and submission deliverables:

---

### 1. Final Project Highlights

#### A. Production Streamlit Web Application (`streamlit_app.py`)
* **Interactive Employee Interface:** Conversational interface featuring multi-turn session memory, interactive expandable citation drawers showing exact source markdown passages, and real-time streaming responses.
* **Live Telemetry & Traces:** Dedicated *Traces* dashboard tab displaying end-to-end execution latency percentiles ($p_{50}, p_{95}$), token consumption budgets, tool call sequences, and JSON repair events.
* **Human-in-the-Loop (HITL) Review Queue:** Built-in reviewer panel displaying flagged consequential requests (terminations, salary adjustments, disciplinary actions) logged asynchronously in `handoffs.jsonl`.
* **Sliding-Window Rate Limiting:** Sliding-window per-IP / per-session rate limiter preventing API abuse and token exhaustion.

#### B. Responsible AI, Safety & Red-Teaming
* **Deterministic PII Redaction:** Pre-guard regex scrubbing masking employee Social Security Numbers, emails, phone numbers, and compensation values before any payload reaches the hosted API.
* **Indirect Prompt Injection Defense:** Document sanitizer stripping malicious prompt injection vectors (system instruction overrides, role-flipping delimiters) hidden inside retrieved handbook files. Validated via `scripts/run_injection_suite.py` achieving an **87.5% defense success rate** (7/8 malicious document injections thwarted).
* **Strict Refusal & Policy Boundaries:** Deterministic interception of material employment decisions and personal employee records, strictly redirecting users to human People Operations representatives.

#### C. Containerized & Reproducible Deployment
* Packaged with production-grade `Dockerfile` and `docker-compose.yml` for single-command deployment (`docker compose up --build`).
* Verified multi-platform compatibility, environment variable isolation, and automated health checks.

---

### 2. Comprehensive System Performance Summary

| Milestone | Architecture / Strategy | Key Metric | Significance / Improvement |
|---|---|---|---|
| **Week 1** | Single-Agent Prompting Baseline | 37.5% Accuracy | Established raw 8B model baseline; documented hallucination and schema failure modes. |
| **Week 2** | Scaled Hybrid RAG (1,151 docs) | 89.1% Hit@5 / 0.793 MRR | BM25 + Dense (`bge-small-en-v1.5`) + Reranking elevated Top-1 retrieval from 58.2% to 70.9%. |
| **Week 3** | Multi-Agent Control Loop | 100.0% Task Accuracy | Decomposition Agent eliminated 8B reasoning overload, cutting $p_{50}$ latency to 5.94s. |
| **Week 4** | Safety & Red-Teaming Suite | 87.5% Injection Defense | Thwarted poisoned corpus injections; 96.5% refusal accuracy on adversarial queries. |

*Overall System Telemetry:*
* **Task Accuracy:** 88.3% across all benchmarked policy queries.
* **Refusal Accuracy:** 96.5% on adversarial, out-of-scope, and consequential queries.
* **Median Latency ($p_{50}$):** 5.94s (Decomposition agent) / 9.76s (overall traced pipeline).
* **Inference Cost:** ~\$0.00019 USD per query on hosted inference.
* **Test Suite:** **60/60 passing unit tests** (`pytest tests`).

---

### 3. Submission Deliverables & Links
1. **GitHub Repository (Branch: Amit):** [GitHub Repository](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/tree/Amit)
2. **Comprehensive 12-Page Technical Report:** [`docs/technical_report.md`](technical_report.md) · [GitHub Link](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/blob/Amit/docs/technical_report.md)
3. **Task Specification & Refusal Boundaries:** [`docs/specification.md`](specification.md) · [GitHub Link](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/blob/Amit/docs/specification.md)
4. **Retrieval Ablation Report:** [`docs/retrieval_results.md`](retrieval_results.md) · [GitHub Link](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/blob/Amit/docs/retrieval_results.md)
5. **Agent Design & Failure Taxonomy:** [`docs/agent-design.md`](agent-design.md) · [GitHub Link](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/blob/Amit/docs/agent-design.md)
6. **Strategy Comparison & Benchmark Report:** [`docs/strategy_comparison.md`](strategy_comparison.md) · [GitHub Link](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/blob/Amit/docs/strategy_comparison.md)
7. **Trace Telemetry & Evaluation Artifacts:** [`traces/traces.jsonl`](../traces/traces.jsonl) & [`eval_runs/summary.json`](../eval_runs/summary.json) · [GitHub Link](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/blob/Amit/eval_runs/summary.json)
8. **Deployment Configuration:** [`Dockerfile`](../Dockerfile) & [`docker-compose.yml`](../docker-compose.yml)

---

### 4. 10-Minute Final Presentation & Live Demo Agenda
We cordially invite you and the evaluation committee to our 10-minute live Capstone presentation:
* **00:00–02:00 — Problem Formulation & $\le$8B Constraints:** Architectural challenges of constrained inference APIs, context saturation, and Week 1 baseline failure characterization.
* **02:00–05:00 — Live Operations Copilot Demonstration:** Live multi-turn employee interaction demonstrating query decomposition, grounded citations with markdown expanders, and real-time execution trace telemetry.
* **05:00–08:00 — Responsible AI & Adversarial Red-Teaming:** Live attack demonstration: indirect prompt injection via poisoned document chunks, PII redaction, and Human-in-the-Loop review queue escalation for consequential actions.
* **08:00–10:00 — Empirical Evaluation, Failure Taxonomy & Viva Q&A:** Walkthrough of Week 1–4 empirical progression, statistical significance testing, and defense Q&A.

**Demo Meeting Link:** `[Insert Google Meet / Zoom Link]`  
**Proposed Date & Time:** `[Insert Date / Slot]`

Thank you for your invaluable guidance, feedback, and mentorship throughout this programme.

Warm regards,  
**Amit Bhardwaj** (and Team)  
*Capstone Track B — Group [Number]*  
*Executive Post Graduate Programme in Generative AI and Agentic AI*  
[Phone Number / Contact Email]

---

## 📌 Appendix: Final Evaluation & Viva Defense Q&A

### Q1: What was the single biggest architectural challenge of building on an edge-class $\le$8B model, and how did you solve it?
**Answer:**  
The biggest challenge was **reasoning saturation under multi-task prompts**. When an 8B model is asked to simultaneously analyze intent, execute retrieval, evaluate document relevance, and format valid JSON, it suffers from attention degradation and hallucinates plausible text. We solved this by decomposing the workflow into dedicated, modular stages: a lightweight Planner agent that only extracts sub-queries; an algorithmic BM25 + dense retrieval engine that handles search deterministically; an Executor agent that generates answers with structured claims; and a Critic agent that validates citation token overlap algorithmically ($\ge 60\%$) rather than asking the LLM to grade itself.

### Q2: How did you defend against indirect prompt injections without fine-tuning model weights?
**Answer:**  
In Track B, we have zero access to model weights or logit adjustments. To defend against indirect prompt injection (where malicious instructions are embedded inside indexed handbook markdown files), we implemented a multi-stage sanitizer (`hr_policy_navigator/agent.py::sanitize_chunk`):
1. **Structural Sanitization:** We strip markdown comment tags, invisible Unicode characters, and pseudo-system role headers (e.g., `[SYSTEM]`, `<|im_start|>`, `### Instructions:`).
2. **Quarantine Tagging:** Evidence passages are presented to the model inside strictly bounded, escaped XML containers (`<evidence id="S1">...</evidence>`) with explicit system-level instructions instructing the model that evidence contents must only be treated as raw factual data, never as executable instructions.
3. **Post-Generation Critic:** The Critic agent verifies that the output strictly answers the user's original query and does not mirror prohibited instructional phrases. This pipeline successfully thwarted 7 out of 8 indirect injection attacks (87.5% defense rate).

### Q3: Why is Human-in-the-Loop (HITL) mandatory for consequential employment actions?
**Answer:**  
Under enterprise labor compliance and ethical AI guidelines, autonomous software must never execute or approve material employment decisions (terminations, salary changes, promotions, formal disciplinary sanctions). Even a 99% accurate model carries catastrophic legal liability if it errs on a termination decision. Our system implements deterministic pre-guard screening that intercepts any consequential action request, refuses direct execution, and automatically generates an asynchronous ticket in the People Operations human review queue (`handoffs.jsonl`) for qualified HR personnel to review.
