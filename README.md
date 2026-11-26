# HR Policy Navigator — Multi-Agent Operations Copilot (Track B)

[![Test Suite](https://img.shields.io/badge/Unit%20Tests-64%2F64%20Passing-brightgreen)](tests)
[![Evaluation Benchmark](https://img.shields.io/badge/Frozen%20Benchmark-84%20Queries%20(100%25%20Completion)-blue)](eval_runs/summary.json)
[![Adversarial Safety](https://img.shields.io/badge/Adversarial%20Refusal-96.6%25-success)](eval_runs/summary.json)
[![Prompt Injection Defense](https://img.shields.io/badge/Injection%20Defense-87.5%25-success)](eval_runs/injection.json)
[![Branch](https://img.shields.io/badge/Active%20Branch-Amit--Enhanced-orange)](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/tree/Amit-Enhanced)

**Course:** Executive Post Graduate Programme in Generative AI and Agentic AI  
**Track:** Track B — API-Side Capstone (Hosted $\le$8B Inference API, No Direct GPU Access, No Fine-Tuning)  
**Corpus:** Sourcegraph Employee Handbook (1,151 documents · 3,686 chunks)  
**Host Stack:** `Qwen/Qwen3-8B` (Chat Completions) + `bge-small-en-v1.5` (Dense Embeddings, 384d)

---

## 🏗️ Architecture

```
                                  ┌───────────────────────────────┐
                                  │      Incoming User Query      │
                                  └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                      ┌───────────────────────┐
                                      │ Deterministic Regex   │  (Scrubs SSN, Card, Email, Phone, IBAN)
                                      │     PII Redactor      │
                                      └───────────┬───────────┘
                                                  │
                                                  ▼
                                      ┌───────────────────────┐
                                      │ Safety Pre-Guardrail  │──▶ [Direct Refusal + Human Review Ticket]
                                      │ (Consequential/Inject)│    (Fire, discipline, salary breach)
                                      └───────────┬───────────┘
                                                  │ (Allowed)
                                                  ▼
                                      ┌───────────────────────┐
                                      │  Multi-Agent Planner  │  (Intent classification, pronoun resolution,
                                      │   (LLM + Heuristics)  │   multi-hop decomposition)
                                      └───────────┬───────────┘
                                                  │ [Sub-queries]
                                                  ▼
                                      ┌───────────────────────┐
                                      │ Tool Executor Loop    │──▶ search_policy (BM25 + Dense RRF + Rerank)
                                      │ (Max 8 Steps, Loop Gd)│──▶ read_section (Context expansion)
                                      │                       │──▶ escalate_to_human (Consequential actions)
                                      └───────────┬───────────┘
                                                  │ [Numbered Evidence: S1, S2, ...]
                                                  ▼
                                      ┌───────────────────────┐
                                      │ Persona-Aware Gen     │  (Injected role & tenure context:
                                      │ (Structured/CoT/Decomp│   HR Ops / Manager / New Hire / Senior)
                                      └───────────┬───────────┘
                                                  │ [Draft Claims + Short IDs]
                                                  ▼
                                      ┌───────────────────────┐
                                      │ Algorithmic Critic    │  (≥60% token overlap check, exact number match,
                                      │   Verification        │   1 context revision pass, ungrounded claim purge)
                                      └───────────┬───────────┘
                                                  │
                     ┌────────────────────────────┴───────────────────────────┐
                     ▼                                                        ▼
         ┌───────────────────────┐                                ┌───────────────────────┐
         │ Grounded Policy Ans   │                                │ Clean Abstention      │
         │ + Verified Citations  │                                │ ("Documents do not    │
         │ + Voice & RLHF Thumbs │                                │   state this.")       │
         └───────────────────────┘                                └───────────────────────┘
```

---

## 🏆 Verified Empirical Achievements

Evaluated on the frozen 84-query benchmark (55 domain queries across 11 policy themes + 29 adversarial/safety probes) and 64 unit tests:

| Metric | Result | Description / Significance |
|---|---|---|
| **Unit Test Suite** | **64 / 64 Passed** (100%) | Complete regression suite running in ~27s |
| **Task Completion Rate** | **100.0%** (84 / 84) | Zero unhandled crashes, infinite loops, or JSON decode breaks |
| **Adversarial Refusal** | **96.6%** (28 / 29) | Refuses consequential actions (6/6), private data (4/4), injections (4/4) |
| **Gold Citation Rate** | **78.2%** (43 / 55) | Cites the exact gold policy file in context |
| **Generation Success** | **94.5%** (52 / 55) | Generates grounded policy answers on valid queries |
| **Indirect Injection Defense** | **87.5%** (7 / 8) | Defends against poisoned document context attacks |
| **Retrieval Hit@1 / Hit@5** | **0.709 / 0.891** | Hybrid BM25 + Dense (`bge-small`) + Lexical Reranker |
| **Retrieval MRR** | **0.793** | 95% bootstrap confidence interval: [0.782, 0.982] |
| **Median Latency ($p_{50}$)** | **5.94s** | Fast end-to-end multi-agent resolution |
| **Average Cost / Query** | **~$0.00019 USD** | Microscopic token footprint on edge-class inference |

---

## ✨ Amit-Enhanced Innovation Features

Beyond baseline Week 4 requirements, `Amit-Enhanced` introduces enterprise-grade platform features:

1. **🔐 Multi-Persona Authentication:**
   * Login screen supporting 4 enterprise personas: `hr_admin` (HR Ops), `mgr_alice` (People Manager), `emp_bob` (New Hire Associate), `emp_carol` (Tenured Associate).
2. **👥 Role & Tenure Personalization (Prompt-Engineered):**
   * *New Hire (<1 yr):* Step-by-step instructions, acronym expansions, onboarding quick tips.
   * *Tenured (3+ yr):* Direct 1–2 sentence answers without navigation steps.
   * *Manager:* Responsibility split (manager vs employee duties) and approval thresholds.
   * *HR Ops:* Full verbatim clauses, exceptions, and escalation paths.
3. **🎙️ Offline Voice I/O:**
   * *Speech-to-Text:* Browser microphone recording transcribed locally via Whisper `tiny` (39MB, CPU).
   * *Read Aloud (TTS):* Offline text-to-speech engine via `pyttsx3` with natural speech cadence.
4. **👍 Continuous RLHF Learning Loop:**
   * In-app 👍 / 👎 feedback buttons + user correction form logging to `state/feedback_log.jsonl`.
   * Real-time Feedback Analytics dashboard identifying queries requiring prompt tuning.
5. **🧑‍💼 Human-in-the-Loop Review Queue:**
   * Styled state-machine tickets: `🔴 Pending` $\to$ `🟡 Acknowledged` $\to$ `🟢 Resolved / 🔺 Escalated`.
   * Role-gated visibility (HR Ops sees full queue; Managers see team-scoped escalations).
6. **🚀 "Our Journey" Interactive Presentation Tab:**
   * Built-in presentation deck showing Week 1 $\to$ Week 4 progression.
   * Live side-by-side comparison arena (Week 1 raw LLM hallucination vs Week 4 copilot).
7. **💬 Persistent Chat History:**
   * Thread-safe JSONL session store with 7-day privacy retention and sidebar session loader.

---

## ⚡ Quick Start

### Option A — Run Locally with Virtualenv

```powershell
# 1. Clone & switch to active branch
git clone https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator.git
cd HAX---HR-Policy-Navigator
git checkout Amit-Enhanced

# 2. Setup environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Configure credentials
copy .env.example .env
# Edit .env with your OPENAI_BASE_URL, OPENAI_API_KEY, and EMBEDDING_BASE_URL

# 4. Run test suite
python -m pytest tests -q

# 5. Launch the Streamlit application
streamlit run streamlit_app.py --server.port=8501
```

Open `http://localhost:8501`.

### Option B — Single-Command Docker Deployment

```powershell
docker compose up --build
```

The container automatically packages SSL certificates (`certs/ca-bundle.crt`), mounts persistent state (`./traces`, `./state`, `./.cache`), and runs under an unprivileged `app` user on `http://localhost:8501`.

---

## 🔑 Demo Login Accounts

| Username | Password | Role | Persona Description |
|---|---|---|---|
| `sys_admin` | `sys123` | System Administrator | Unrestricted superuser access (all tabs, global traces/feedback, Evaluation Runner, Bird's Eye View) |
| `hr_admin` | `hr123` | HR Operations | Full policy verbatim, exception clauses, full HITL queue manager, Our Journey |
| `mgr_alice` | `mgr123` | People Manager | Manager vs employee obligations, team-scoped review queue |
| `emp_bob` | `emp123` | Associate (New Hire) | Step-by-step onboarding, acronym definitions, quick tips |
| `emp_carol` | `emp456` | Associate (Tenured) | Direct, concise answers for experienced employees |

---

## 📁 Repository Structure

```
HAX---HR-Policy-Navigator/
├── streamlit_app.py                # Main UI (Login, Ask, Traces, Feedback, Journey, HITL)
├── Dockerfile                      # Production container spec
├── docker-compose.yml              # Volume mounts for traces, state, cache, certs
├── requirements.txt                # Core dependencies + pyttsx3
├── hr_policy_navigator/
│   ├── agent.py                    # Planner -> Executor -> Critic multi-agent loop
│   ├── auth.py                     # Demo credentials & UserProfile personas (Sprint 1)
│   ├── chat_history.py             # Persistent per-user JSONL chat store (Sprint 1)
│   ├── chunking.py                 # Heading-capped semantic chunker (≤500 tokens)
│   ├── config.py                   # Environment configuration & cost accounting
│   ├── feedback.py                 # RLHF FeedbackStore & analytics engine (Sprint 3)
│   ├── guardrails.py               # Deterministic regex PII scrubber & injection scanner
│   ├── llm.py                      # JSON schema repair cascade & budget manager
│   ├── prompts.py                  # Persona-aware system prompts & few-shot schemas (Sprint 2)
│   ├── retrieval.py                # Hybrid BM25 + Dense RRF (k=60) + Lexical Reranker
│   ├── serving.py                  # Sliding-window RateLimiter & LRU ResponseCache
│   ├── state.py                    # SessionStore & HandoffStore state machine (Sprint 2)
│   ├── text_utils.py               # LCS & ROUGE-LCS evaluation metric calculation
│   ├── tools.py                    # Schema-validated ToolRegistry
│   ├── tracing.py                  # Hierarchical JSONL span tracing
│   └── voice.py                    # Offline Whisper STT & pyttsx3 Read Aloud (Sprint 3)
├── docs/
│   ├── STARTUP_GUIDE.md            # 1-page startup instructions for examiners
│   ├── DEMO_SCRIPT.md              # 10-minute presentation cue script & slide outline
│   ├── technical_report.md         # 12-page comprehensive publication-grade report
│   ├── specification.md            # Formal scope & refusal boundaries
│   ├── retrieval_results.md        # Retrieval chunking & mode ablations
│   └── strategy_comparison.md      # Gunjan two-column Decision Accuracy vs ROUGE-LCS table
├── eval_runs/                      # Frozen benchmark evaluation outputs
├── scripts/                        # Automated evaluation, synthetic scaling & ablation scripts
└── tests/                          # 64 unit tests covering agents, tools, serving & safety
```

---

## 🧪 Evaluation Commands

```powershell
# Run 64 unit tests
.venv\Scripts\pytest tests -q

# Run retrieval ablation (BM25 vs Dense vs Hybrid vs Reranker)
.venv\Scripts\python scripts/run_retrieval_eval.py

# Run full end-to-end benchmark (84 queries)
.venv\Scripts\python scripts/run_e2e_eval.py --strategies decomposition

# Run indirect prompt injection attack suite (8 vectors)
.venv\Scripts\python scripts/run_injection_suite.py
```

---
*Created for the Executive Post Graduate Programme in Generative AI and Agentic AI Capstone Submission.*
