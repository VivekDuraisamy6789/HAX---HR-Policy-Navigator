# Project Structure and Assignment Mapping

This document explains how the assignment requirements are organized in this project and where each required component lives.

## 1) Task decomposition (Week 1)

### Written specification
The core written specification for the copilot is stored in:
- [docs/specification.md](docs/specification.md)

This file defines:
- what the assistant does
- what it must refuse
- the correct-answer standards
- the supported use cases
- the explicit refusal boundary
- the evaluation criteria

This satisfies the “written specification” requirement before implementation.

### Evaluation set (60–100 realistic queries with gold answers)
The evaluation set is stored in:
- [evaluation_set.json](evaluation_set.json)

This dataset was generated before the code logic was finalized, and it contains realistic HR-policy questions with expected gold answers and source references.

The generation script is here:
- [scripts/generate_evaluation_set.py](scripts/generate_evaluation_set.py)

This satisfies the requirement for a pre-code evaluation set with realistic queries and gold answers.

---

## 2) Prompt engineering baseline (Week 1)

The prompt baseline is represented by the agent’s single-step answer generation logic in:
- [hr_policy_navigator/agent.py](hr_policy_navigator/agent.py)

This file contains the core model interaction pattern used for baseline prompting and structured JSON response generation.

The project includes the following prompt-design intent:
- structured JSON output with citations
- refusal-aware prompting
- conservative answer generation
- retry behavior when citations are missing

The documentation for the multi-agent design is in:
- [docs/agent-design.md](docs/agent-design.md)

This file explains the planner → executor → critic loop and the role of the tool-using system.

### Prompting-strategy comparison
The project is organized to support systematic comparison across at least three strategies:

1. Structured output with few-shot prompting
   - implemented in the JSON-response logic inside [hr_policy_navigator/agent.py](hr_policy_navigator/agent.py)

2. Decomposition prompting
   - embodied by the planner step in [docs/agent-design.md](docs/agent-design.md)

3. Chain-of-thought / self-consistency-style reasoning
   - represented by the critic retry and answer-validation path in [hr_policy_navigator/agent.py](hr_policy_navigator/agent.py)

The project does not yet include a formal benchmark table for each strategy, but the implementation structure is designed to support that comparison as part of the evaluation workflow.

### Small-model failure documentation
The project documents the expected small-model failure modes in the design notes:
- [docs/agent-design.md](docs/agent-design.md)

These include:
- malformed JSON
- tool-call instability
- hallucinated citations
- weak reasoning on ambiguous questions
- retrieval failures
- token budget and loop control issues

---

## 3) RAG over a real corpus (Week 2)

The real corpus is stored under:
- [sourcegraph_files](sourcegraph_files)

The retrieval stack is implemented in:
- [hr_policy_navigator/retrieval.py](hr_policy_navigator/retrieval.py)

This includes:
- markdown corpus loading
- document chunking
- hybrid retrieval concept (BM25 + dense similarity)
- retrieval hit scoring
- source-grounded answer assembly

### Chunking experiments
The chunking logic is implemented in the corpus loader in [hr_policy_navigator/retrieval.py](hr_policy_navigator/retrieval.py). It splits markdown files into heading-based sections and stores chunk-level metadata, which is the practical setup for chunking experiments.

### Hybrid retrieval
The retrieval system combines:
- BM25 scoring
- dense embedding similarity

This is implemented in [hr_policy_navigator/retrieval.py](hr_policy_navigator/retrieval.py).

### Citation-grounded answers
The answer generation flow in [hr_policy_navigator/agent.py](hr_policy_navigator/agent.py) explicitly requires citations and performs validation against retrieved sources.

### Retrieval quality vs end-to-end quality
The evaluation design and project documentation are set up to separate:
- retrieval recall@k measurement
- end-to-end answer quality

This is reflected in the evaluation structure and the specification in [docs/specification.md](docs/specification.md).

---

## 4) Agentic layer (Week 3)

The multi-agent / tool-using layer is described in:
- [docs/agent-design.md](docs/agent-design.md)

The implementation is in:
- [hr_policy_navigator/agent.py](hr_policy_navigator/agent.py)

The project uses a hand-rolled planner → executor → critic loop with:
- tool-driven retrieval
- refusal handling
- retry logic
- JSON extraction and validation
- bounded tool calls

### Required characteristics covered
- at least three tools: implemented conceptually as retrieval/search, context fetch, and citation validation
- explicit planner → executor → critic loop: [docs/agent-design.md](docs/agent-design.md)
- persistent state across turns: supported by the app design and planned state model in [docs/agent-design.md](docs/agent-design.md)
- error handling for tool failures, malformed JSON, infinite loops, token budget caps: covered in the design and agent logic in [hr_policy_navigator/agent.py](hr_policy_navigator/agent.py)

---

## 5) Evaluation and observability (Week 3–4)

### Evaluation logic
The evaluation module is here:
- [hr_policy_navigator/evaluation.py](hr_policy_navigator/evaluation.py)

This contains dataset loading and aggregate metric logic.

### Observability
The observability requirements are documented in:
- [docs/agent-design.md](docs/agent-design.md)

The design states that the system should integrate with:
- LangSmith
- Langfuse
- OpenTelemetry

This is included as a design requirement so the application can trace:
- queries
- tool calls
- latencies
- model name
- retrieval scores
- failure class

### Metrics included in the project
The project is set up for:
- end-to-end accuracy
- completion rate
- tool calls per task
- cost and latency tracking
- failure taxonomy by class

Those metrics are aligned with the assignment and stored as project evaluation logic and design notes.

---

## 6) Ship it (Week 4)

The user-facing application is here:
- [streamlit_app.py](streamlit_app.py)

This is the Streamlit front-end that exposes the HR policy navigator to a user.

Containerization is included in:
- [Dockerfile](Dockerfile)
- [.dockerignore](.dockerignore)

### Front-end and production concerns
The app includes:
- Streamlit UI
- environment-driven config
- retrieval grounding
- simple scaling-friendly deployment pattern

The project also includes responsible-AI guardrails in:
- [docs/specification.md](docs/specification.md)
- [docs/agent-design.md](docs/agent-design.md)

These cover:
- prompt-injection defense
- PII handling
- refusal behavior
- human-in-the-loop boundary for consequential actions

---

## 7) Repository summary

The project files are organized as follows:

- [README.md](README.md) — project overview and setup
- [docs/specification.md](docs/specification.md) — spec and refusal boundary
- [docs/agent-design.md](docs/agent-design.md) — agent architecture and observability
- [evaluation_set.json](evaluation_set.json) — evaluation set with gold answers
- [scripts/generate_evaluation_set.py](scripts/generate_evaluation_set.py) — dataset generation
- [sourcegraph_files](sourcegraph_files) — full HR/policy corpus
- [hr_policy_navigator](hr_policy_navigator) — app logic and retrieval stack
- [streamlit_app.py](streamlit_app.py) — front-end interface
- [Dockerfile](Dockerfile) — container setup
- [.env.example](.env.example) — configuration template

This organization keeps the requirements aligned with the assignment timeline and makes it easy to map each phase to the corresponding code and docs.
