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
* **Multi-Persona Enterprise Authentication:** Secure login screen supporting 4 distinct roles & personas (`hr_admin`, `mgr_alice`, `emp_bob`, `emp_carol`).
* **Personalized Response Adaptation:** Dynamically tailors answer depth and style based on employee tenure and role (step-by-step onboarding for new hires vs. concise direct answers for 3+ yr seniors).
* **Offline Voice I/O & Accessibility:** Speech-to-text powered locally by Whisper `tiny` (39MB CPU model) and offline Read Aloud text-to-speech powered by `pyttsx3`.
* **Continuous RLHF Learning Loop:** In-app thumbs-up / thumbs-down ratings capturing real user signal and suggestions into `state/feedback_log.jsonl`, visualized on a real-time Feedback Analytics dashboard.
* **"Our Journey" Interactive Presentation Tab:** Built-in 4-week progression story arc and live side-by-side comparison arena (Week 1 raw LLM hallucination vs. Week 4 copilot).
* **Live Telemetry & Traces:** Dedicated *Traces* dashboard tab displaying end-to-end execution latency percentiles ($p_{50}, p_{95}$), token consumption budgets, tool call sequences, and JSON repair events.
* **Human-in-the-Loop (HITL) State Machine:** Built-in reviewer panel displaying flagged consequential requests with audit lifecycle tracking (`🔴 Pending` $\to$ `🟡 Acknowledged` $\to$ `🟢 Resolved / 🔺 Escalated`) and resolution notes.
* **Sliding-Window Rate Limiting & Response Cache:** Sliding-window rate limiter (10 req/min) and LRU ResponseCache (900s TTL) preventing API abuse and token exhaustion.

#### B. Responsible AI, Safety & Red-Teaming
* **Deterministic PII Redaction:** Pre-guard regex scrubbing masking employee Social Security Numbers, emails, phone numbers, and compensation values before any payload reaches the hosted API.
* **Indirect Prompt Injection Defense:** Document sanitizer stripping malicious prompt injection vectors hidden inside retrieved handbook files. Validated via `scripts/run_injection_suite.py` achieving an **87.5% defense success rate** (7/8 malicious document injections thwarted).
* **Strict Refusal & Policy Boundaries:** Deterministic interception of material employment decisions and personal employee records, strictly redirecting users to human People Operations representatives with automated ticket generation.

#### C. Containerized & Reproducible Deployment
* Packaged with production-grade `Dockerfile` and `docker-compose.yml` for single-command deployment (`docker compose up --build`).
* Verified multi-platform compatibility, pre-bundled SSL certificates (`certs/ca-bundle.crt`), environment variable isolation, and automated health checks.

---

### 2. Comprehensive System Performance Summary

| Milestone | Architecture / Strategy | Key Metric | Significance / Improvement |
|---|---|---|---|
| **Week 1** | Single-Agent Prompting Baseline | 37.5% Accuracy | Established raw 8B model baseline; documented hallucination and schema failure modes. |
| **Week 2** | Scaled Hybrid RAG (1,151 docs) | 89.1% Hit@5 / 0.793 MRR | BM25 + Dense (`bge-small-en-v1.5`) + Reranking elevated Top-1 retrieval from 58.2% to 70.9%. |
| **Week 3** | Multi-Agent Control Loop | 100.0% Completion / 96.6% Refusal | Decomposition Agent eliminated 8B reasoning overload, cutting $p_{50}$ latency to 5.94s. |
| **Week 4** | Amit-Enhanced Enterprise Platform | 87.5% Injection Def / 60 Tests | Multi-persona auth, voice I/O, RLHF loop, HITL state machine, 100% unit tests passed. |

*Overall System Telemetry:*
* **Task Completion Rate:** 100.0% (84 / 84 benchmark queries completed with 0 unhandled crashes).
* **Adversarial Refusal Accuracy:** 96.6% (28 / 29 safe refusals on dangerous/adversarial probes).
* **Exact Gold Document Citation Rate:** 78.2% (43 / 55 domain queries).
* **Indirect Injection Defense Rate:** 87.5% (7 / 8 attack vectors thwarted).
* **Median Latency ($p_{50}$):** 5.94s (Decomposition agent).
* **Inference Cost:** ~\$0.00019 USD per query on hosted inference.
* **Test Suite:** **60/60 passing unit tests** (`pytest tests`).

---

### 3. Submission Deliverables & Links
1. **GitHub Repository (Branch: Amit-Enhanced):** [GitHub Repository](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/tree/Amit-Enhanced)
2. **Comprehensive 12-Page Technical Report:** [`docs/technical_report.md`](technical_report.md) · [GitHub Link](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/blob/Amit-Enhanced/docs/technical_report.md)
3. **10-Minute Live Demo Cue Script:** [`docs/DEMO_SCRIPT.md`](DEMO_SCRIPT.md) · [GitHub Link](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/blob/Amit-Enhanced/docs/DEMO_SCRIPT.md)
4. **1-Page Startup Guide:** [`docs/STARTUP_GUIDE.md`](STARTUP_GUIDE.md) · [GitHub Link](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/blob/Amit-Enhanced/docs/STARTUP_GUIDE.md)
5. **Task Specification & Refusal Boundaries:** [`docs/specification.md`](specification.md) · [GitHub Link](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/blob/Amit-Enhanced/docs/specification.md)
6. **Retrieval Ablation Report:** [`docs/retrieval_results.md`](retrieval_results.md) · [GitHub Link](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/blob/Amit-Enhanced/docs/retrieval_results.md)
7. **Strategy Comparison & Benchmark Report:** [`docs/strategy_comparison.md`](strategy_comparison.md) · [GitHub Link](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/blob/Amit-Enhanced/docs/strategy_comparison.md)
8. **Trace Telemetry & Evaluation Artifacts:** [`traces/traces.jsonl`](../traces/traces.jsonl) & [`eval_runs/summary.json`](../eval_runs/summary.json) · [GitHub Link](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/blob/Amit-Enhanced/eval_runs/summary.json)
9. **Deployment Configuration:** [`Dockerfile`](../Dockerfile) & [`docker-compose.yml`](../docker-compose.yml)

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
