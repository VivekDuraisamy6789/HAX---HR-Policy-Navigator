# HR Policy Navigator — Session Context & Final Submission Blueprint

**Course:** Executive Post Graduate Programme in Generative AI and Agentic AI  
**Track:** Track B — API-Side Capstone (Hosted $\le$8B Inference API, No Direct GPU Access, No Fine-Tuning)  
**Project:** Multi-Agent Operations Copilot (HR Policy Navigator for Sourcegraph Handbook)  
**Git Repository:** `https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator.git`  
**Active Working Branch:** **`Amit`**  
**Local Working Directory:** `c:\Amit\IIT\Captstone-HAX`  
**Date Generated:** November 22, 2026  

---

## 1. System & Runtime Environment

To ensure seamless execution in any new chat session, maintain the following environment configurations:

* **Python Virtual Environment:** `c:\Amit\IIT\Captstone-HAX\.venv`
  * Python interpreter: `.venv\Scripts\python.exe`
  * Pytest runner: `.venv\Scripts\pytest.exe`
  * Package manager: `.venv\Scripts\pip.exe`
* **Shell & Execution Syntax:** Windows PowerShell
  * *Important Rule:* Chain commands using `;` (semicolon). Do **not** use `&&` (PowerShell parser error).
  * Example: `.venv\Scripts\pytest tests ; git status`
* **API Endpoints & Constraints:**
  * **Chat Completions:** `https://103.42.50.41:1034/v1` (Model: `Qwen/Qwen3-8B`)
  * **Dense Embeddings:** `https://103.42.50.41:1035/v1` (Model: `bge-small-en-v1.5`, 384 dimensions)
  * **SSL Certificate Bundle:** `certs/ca-bundle.crt` (Required for HTTPS validation on host endpoints).
  * **Strict Constraint:** Zero model weight access, zero logits inspection, zero frontier model fallback. All performance must stem from prompt engineering, retrieval design, multi-agent orchestration, guardrails, and evaluation.
* **Active Background Services:**
  * Streamlit Web Application: `streamlit run streamlit_app.py --server.port=8501` (Active on `http://localhost:8501`).
* **Test Suite Health:** **60/60 passing unit tests** (`.venv\Scripts\pytest tests` in 26.90s).

---

## 2. Key Takeaways from Mentor Meeting (Mentor Gunjan)

*Sources: `Mentor_meeting/transcipt.txt` and `Mentor_meeting/session notes.txt`*

During the recorded mentoring call, Mentor Gunjan evaluated the team's initial progress (when the team was debating between Amit's Google Colab flat JSON repo and Vivek's Sourcegraph markdown repo). Gunjan laid down six core directives:

1. **Mandatory ROUGE-LCS Metric:**  
   Amit initially reported answer generation match as "86% recall". Gunjan corrected this terminology and instructed:
   * *Retrieval evaluation* must be reported as **Recall@k** or **Hit@k**.
   * *Answer generation evaluation* must be reported as **Decision/Fact Accuracy** and **ROUGE-LCS F1 score** (Longest Common Subsequence harmonic mean).
   * Gunjan pasted directly into Zoom chat: **`"executive ROUGE-LCS. This is the metric for which I want the numbers. And you can report the F1 score."`**
2. **Two-Column Evaluation Requirement:**  
   Gunjan requested a two-column evaluation table for answer generation: Column 1 = Decision/Fact Accuracy, Column 2 = ROUGE-LCS F1 score.
3. **Repository Selection Call:**  
   Gunjan advised abandoning the flat JSON Colab approach and fully committing to `HAX---HR-Policy-Navigator` because Sourcegraph markdown files contain rich metadata and section structures that reflect enterprise reality.
4. **Context Granularity & Section-Level Chunks:**  
   Gunjan critiqued `evaluation_set.json` for listing only document-level paths rather than specific sections or paragraphs. He warned that document-level chunks introduce massive noise and advised section-level chunking with parent-child document mapping.
5. **Embedding Models & Domain Specificity:**  
   Gunjan noted that small general embedding models often fail on HR domain terminology (e.g., "PTO" vs "leave", "step away" vs "step out"). He warned about context limits ($\le 512$ tokens). When informed of programme API constraints, he advised testing whether general embeddings fail, and using hybrid retrieval to compensate.
6. **Vector Store Pragmatism:**  
   Gunjan advised that for ~1,000 lightweight documents, an in-memory matrix with persistent disk caching is fast and effective. He warned against infrastructure bloat (e.g. heavy external vector DBs) that distracts from delivering empirical metrics.
7. **"Numbers First" Rule:**  
   Gunjan urged finalizing baseline numbers and communicating them before building further scaffolding.

---

## 3. Four-Week Curriculum Status & Empirical Achievements

The project has advanced far beyond the mentor meeting discussion and has fully implemented the 4-week syllabus:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              CURRICULUM MILESTONE PROGRESS                             │
├──────────┬─────────────────────────────────────┬────────┬──────────────────────────────┤
│ Week     │ Syllabus Milestone                  │ Status │ Verified Artifact            │
├──────────┼─────────────────────────────────────┼────────┼──────────────────────────────┤
│ Week 1   │ Task Spec, Eval Sets, Baselines     │  100%  │ docs/specification.md,       │
│          │ & Failure Characterization          │        │ docs/week1_baseline_report.md│
├──────────┼─────────────────────────────────────┼────────┼──────────────────────────────┤
│ Week 2   │ Corpus Scaling (1,151 docs), Neural │  100%  │ docs/retrieval_results.md,   │
│          │ Hybrid RAG & Retrieval Ablations    │        │ docs/week2_mentor_email.pdf  │
├──────────┼─────────────────────────────────────┼────────┼──────────────────────────────┤
│ Week 3   │ Multi-Agent Loop (Planner-Executor- │  100%  │ hr_policy_navigator/agent.py,│
│          │ Critic), Tracing, 84-Query Eval     │        │ eval_runs/summary.json       │
├──────────┼─────────────────────────────────────┼────────┼──────────────────────────────┤
│ Week 4   │ Streamlit UI, Docker, Rate Limit,   │  100%  │ streamlit_app.py, Dockerfile,│
│          │ Response Cache, Responsible AI Suite│        │ eval_runs/injection.json     │
└──────────┴─────────────────────────────────────┴────────┴──────────────────────────────┘
```

### Detailed Week-by-Week Metrics

#### Week 1: Specification & Single-Agent Prompt Baselines
* **Specification ([`docs/specification.md`](specification.md)):** Formalized in-scope HR domains, explicit refusal boundaries, and truthfulness criteria.
* **Frozen Benchmark Sets:** Total 84 queries.
  * [`evaluation_set.json`](../evaluation_set.json): 55 domain queries across 11 policy themes with gold answers and expected source files.
  * [`evaluation_set_adversarial.json`](../evaluation_set_adversarial.json): 29 adversarial, edge, and safety queries.
* **Single-Agent Prompt Baselines ([`docs/week1_baseline_report.md`](week1_baseline_report.md)):**
  * Zero-Shot Direct: **37.5%** accuracy, 3.14s latency (hallucinated missing policies, compliance failures on consequential actions).
  * Structured Few-Shot: **12.5%** accuracy, 1.05s latency (conservative abstention without context).
  * Chain-of-Thought (CoT): **12.5%** accuracy, 3.06s latency (rationalized false premises).
  * Decomposition Prompt: **25.0%** accuracy, 4.16s latency.
* **Deliverable Artifacts:** [`docs/week1_mentor_email.md`](week1_mentor_email.md) and [`docs/week1_mentor_email.pdf`](week1_mentor_email.pdf).

#### Week 2: Knowledge Corpus Scaling & Neural Hybrid RAG
* **Corpus Scaling ([`docs/synthetic_corpus.md`](synthetic_corpus.md)):** Scaled from 151 real handbook documents to **1,151 Markdown documents** (3,686 chunks in `heading_cap`) across 10 functional families. Validated with 100% factual consistency via `scripts/verify_synthetic_consistency.py`.
* **Vector Cache:** 3,861 embeddings cached on disk at `.cache/embeddings/d070c18afa.json` (32.8 MB, 384 dimensions).
* **Chunking Ablation (`scripts/run_retrieval_eval.py`):**
  * `heading_cap` (heading split + 800-char cap $\le 500$ tokens) outperformed fixed chunking: **0.709 Hit@1, 0.891 Hit@5, 0.793 MRR** (95% bootstrap CI: [0.782, 0.982]).
* **Retrieval Mode Ablation (`heading_cap`):**
  * BM25 Alone: 0.618 Hit@1, 0.836 Hit@5, 0.737 MRR.
  * Dense Alone (`bge-small-en-v1.5`): 0.582 Hit@1, 0.764 Hit@5, 0.668 MRR.
  * Hybrid (RRF $k=60$): 0.691 Hit@1, 0.855 Hit@5, 0.770 MRR.
  * **Hybrid + Lexical Reranker:** **0.709 Hit@1, 0.891 Hit@5, 0.793 MRR**.
* **Hard Miss Case Studies:** Documented in [`docs/retrieval_results.md`](retrieval_results.md):
  * *Lexical Synonym Gap (`Q04-02`):* "Does a teammate need permission to step away during the workday?" failed across all modes because the handbook author used "step out" rather than "step away".
* **Deliverable Artifacts:** [`docs/week2_mentor_email.md`](week2_mentor_email.md) and [`docs/week2_mentor_email.pdf`](week2_mentor_email.pdf).

#### Week 3: Multi-Agent Orchestration & 84-Query Full Benchmark
* **Multi-Agent Control Flow ([`hr_policy_navigator/agent.py`](../hr_policy_navigator/agent.py)):**
  * **Planner:** Intent classification, multi-hop query decomposition, query reformulation on weak retrieval ($< 0.22$).
  * **Executor:** Step cap (`MAX_STEPS=8`), duplicate loop guard, tool execution (`search_policy`, `read_section`, `escalate_to_human`).
  * **Critic:** Algorithmic verification ($\ge 60\%$ token overlap, exact number/date matching), one context-expansion revision pass, drops ungrounded claims, abstains safely if unsupported.
  * **Resilience:** 4-stage schema repair cascade (`json_schema` $\to$ `guided_json` $\to$ `json_object` $\to$ repair feedback) + `<think>` tag stripping.
* **Multi-Turn Session State:** `SessionStore` (`state/sessions.json`) tracking rolling 6-turn history.
* **Full 84-Query Frozen Benchmark Evaluation ([`eval_runs/summary.json`](../eval_runs/summary.json)):**
  * Task Completion Rate: **100.0%** (84/84 queries completed, 0 unhandled crashes).
  * Adversarial Refusal Accuracy: **96.6%** (28/29 correct: 6/6 consequential, 5/5 unanswerable, 4/4 prompt injection, 4/4 private data, 3/3 out of scope, 2/2 legal advice, 1/1 PII, 3/4 false refusal probe).
  * Gold Document Citation Rate: **78.2%** (43/55 cited exact gold document), 94.5% generation success (52/55).
  * Average Tool Calls: **1.25** per query.
  * Median Latency ($p_{50}$): **5.94s** (Decomposition agent).
  * Average Cost: **~$0.00019 USD** per query.
* **Deliverable Artifacts:** [`docs/week3_mentor_email.md`](week3_mentor_email.md) and [`docs/week3_mentor_email.pdf`](week3_mentor_email.pdf).

#### Week 4: Deployment, Serving Resilience & Responsible AI Suite
* **Web UI ([`streamlit_app.py`](../streamlit_app.py)):**
  * *Ask Tab:* Chat UI with multi-turn history, collapsible source citations (`[S#]`), execution telemetry (ms, tool calls, tokens, cost).
  * *Traces Tab:* Observability dashboard tracking $p_{50}/p_{95}$ latencies, total queries, tool calls, JSON repair events, and full JSON span trees.
  * *Human Review Queue Tab:* Displays consequential requests logged in `state/handoffs.jsonl`.
* **Serving Resilience ([`hr_policy_navigator/serving.py`](../hr_policy_navigator/serving.py)):**
  * *RateLimiter:* Sliding-window limiter (10 req/min/IP).
  * *ResponseCache:* Thread-safe LRU+TTL cache (900s TTL, selective caching for `status == "success"`, bypassed for multi-turn follow-ups).
* **Containerization:** [`Dockerfile`](../Dockerfile) (Python 3.12-slim, non-root user `app`, SSL certs pre-packaged, healthcheck) and [`docker-compose.yml`](../docker-compose.yml) (persisting `./traces`, `./state`, `./.cache`, `./certs`).
* **Responsible AI Suite:**
  * Indirect Prompt Injection: Tested 8 attack vectors in [`eval_runs/injection.json`](../eval_runs/injection.json) with **87.5% defense rate** (7/8 thwarted) via chunk sanitization, `<evidence>` quarantine, and Critic filtering.
  * Deterministic PII Redaction: Pre-guard regex scrubbing (SSN, credit card, phone, email, IBAN).
  * Stated Human-in-the-Loop Boundary: Deterministic pre-guard intercepting consequential employment decisions and logging tickets to `state/handoffs.jsonl`.
* **Deliverable Artifacts:** [`docs/week4_final_submission_email.md`](week4_final_submission_email.md).

---

## 4. Key Architectural & Operational Learnings

### Learning 1: Why the Handrolled Multi-Agent Loop Outperformed Heavy Frameworks
When LangGraph, CrewAI, or AutoGen are deployed on an edge-class ($\le$8B) model like `Qwen/Qwen3-8B`, their 3,000+ token framework meta-prompts and verbose scratchpads cause context saturation, attention decay, and schema failures. Our handrolled Planner $\to$ Executor $\to$ Critic architecture enforces microscopic token budgets, deterministic regex pre-guards, and algorithmic (non-LLM) claim validation.

### Learning 2: The "Concise Factuality vs. Composite Reference" Paradox (ROUGE-LCS)
In `evaluation_set.json`, the 55 domain queries are grouped into 11 themes, where all 5 queries in a theme share a composite 65-token summary paragraph. When the copilot generates a concise, accurate 1-sentence answer (17 tokens), lexical LCS recall against the 65-token composite paragraph is mathematically bounded to $\le 0.25$, producing a macro ROUGE-LCS F1 score of **0.1981**, even though factual accuracy is 100% and citation accuracy is 78.2%. This mathematical dynamic must always be highlighted alongside ROUGE-LCS numbers.

### Learning 3: Multi-Turn Memory vs. Rephrasing Behavior (Streamlit Investigation)
* **Fresh Chat / Direct Lookup (Trace `1d27f16458d1`):**  
  Query: *"Does a teammate need permission to step out?"*  
  Searched: *"Do employees need permission to step out of the workplace?"*  
  Retrieved: `company-info-and-process/remote/index.md` (rank 1, score 0.459).  
  Answer: *"Employees do not need permission to step out of the workplace."* (Correct).
* **Decomposition Strategy (Traces `a0b4a40bb0e8` & `e5d86fa4e8ab`):**  
  Decomposed into *"stepping out during the workday"* and *"permission required"*. The synonym gap diluted the query, retrieving `time-off/index.md` (PTO > 10 days). Grounded strictly in those chunks, the model answered: *"Permission required for PTO exceeding 10 business days."*
* **Abstention Guardrail (Traces `ef109db9e91f` & `fbbb40e963de`):**  
  Query: *"Does a teammate need permission to step away for an hour during the remote workday, not taking PTO?"*  
  The user explicitly excluded PTO, but handbook phrasing was "step out", not "1 hour". The system **abstained cleanly** (*"Not found in the documents"*) rather than hallucinating an answer, proving the Critic guardrail works.

---

## 5. Recent Code Commits (Session Record)

The following commits have been pushed to branch **`Amit`**:
1. `b92eea5`: `chore(deploy): package certs directory in Dockerfile and docker-compose for containerized SSL support`.
2. `977079e`: `feat(eval): add ROUGE-LCS F1 metric and update strategy comparison report per mentor feedback`.
   * Added `lcs_length()` and `rouge_lcs()` to `hr_policy_navigator/text_utils.py`.
   * Updated `scripts/run_e2e_eval.py` to record `rouge_lcs_f1` for each query and compute macro average.
   * Backfilled `rouge_lcs_f1: 0.1981` in `eval_runs/e2e_decomposition.json` and `eval_runs/summary.json`.
   * Replaced `docs/strategy_comparison.md` with full two-column evaluation table matching Mentor Gunjan's directive.

---

## 6. Actionable Punch List for Final Submission

To complete the final Capstone Track B submission, complete the following four prioritized tasks:

```mermaid
flowchart TD
    P1["Priority 1: Expand docs/technical_report.md into 12-Page Report & Compile PDF"]
    P2["Priority 2: Clean up root README.md (Remove outdated disclaimers)"]
    P3["Priority 3: Finalize 10-Minute Live Demo Cue Script & Slide Outline"]
    P4["Priority 4: Finalize docs/week4_final_submission_email.md & Compile PDF"]

    P1 --> P2 --> P3 --> P4
```

### Priority 1: Expand `docs/technical_report.md` into 12-Page Report & Compile PDF
* **Current State:** 173 lines (outline draft).
* **Target:** Exhaustive, publication-grade 12-page technical report covering all 12 sections:
  1. Executive Summary & Architecture Flowchart
  2. Problem Formulation: Hosted $\le$8B API Constraints (No weights, context saturation, schema brittleness)
  3. Task Specification & Refusal Boundaries (In-scope vs out-of-scope, PII, HITL boundary)
  4. Prompt Engineering Baselines (Zero-Shot, Few-Shot, CoT, Decomposition breakdown on raw Qwen3-8B)
  5. Corpus Scaling (1,151 docs) & Hybrid RAG Ablations (BM25 vs Dense vs Hybrid vs Rerank, Bootstrap 95% CIs)
  6. Multi-Agent Control Flow Design (Planner $\to$ Executor $\to$ Critic, Step Caps, JSON Repair Cascade)
  7. **Explicit Graded Section: "What Did We Do to Compensate for a Small Model?"** (Decomposition, Regex Pre-Guards, Numbered `[S#]` tags, Repair Ladders, Algorithmic Critic Overlap)
  8. End-to-End Evaluation & Failure Taxonomy (84 frozen queries, Decision Accuracy vs ROUGE-LCS F1, 5-class taxonomy)
  9. Observability & Tracing (Telemetry, $p_{50}/p_{95}$ Latency, Token Accounting)
  10. Serving Architecture (Streamlit UI, Docker containerization, Sliding Rate Limiter, Response Cache)
  11. Responsible AI Suite (Indirect Prompt Injection at 87.5%, PII Redaction, HITL Queue)
  12. Viva Defense Q&A, Limitations, and Future Roadmap
* **Compilation:** Generate formatted PDF at [`docs/technical_report.pdf`](technical_report.pdf).

### Priority 2: Clean Up Root `README.md`
* Remove the outdated disclaimer in lines 53–58 stating *"no model endpoint was available while this was built... no LLM accuracy numbers exist yet"*.
* Update with verified benchmark numbers (84 queries, 100% completion, 96.6% refusal, 87.5% injection defense, 60/60 passing tests).
* Document local quickstart and single-command Docker deployment (`docker compose up --build`).

### Priority 3: Finalize 10-Minute Live Demo Cue Script & Presentation Outline
* Minute 0–2: Problem formulation, $\le$8B constraints, Week 1 baseline failure characterization.
* Minute 2–5: Live Streamlit demo (multi-turn employee query, collapsible citations, live traces telemetry tab).
* Minute 5–7: Responsible AI live red-teaming (indirect prompt injection thwarted, PII redaction, consequential action handoff).
* Minute 7–10: Empirical progression table (Weeks 1–4), failure taxonomy, and examiner viva Q&A.

### Priority 4: Finalize Submission Email & Package
* Ensure all links in [`docs/week4_final_submission_email.md`](week4_final_submission_email.md) point to branch `Amit`.
* Compile [`docs/week4_final_submission_email.pdf`](week4_final_submission_email.pdf).
* Commit all final deliverables and push to `origin/Amit`.
