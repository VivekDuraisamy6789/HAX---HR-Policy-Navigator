# Multi-Agent HR Policy Navigator

This repository contains a grounded HR policy assistant for the Sourcegraph policy corpus. It is designed for the assignment: “Multi-Agent Operations Copilot: An Agentic System Built Entirely on a Hosted ≤8B Inference API Scenario”.

## What is included

- A written specification in [docs/specification.md](docs/specification.md)
- A multi-agent system design in [docs/agent-design.md](docs/agent-design.md)
- A 60-question evaluation set in [evaluation_set.json](evaluation_set.json)
- A retrieval + citation system built over the markdown policy corpus in [sourcegraph_files](sourcegraph_files)
- A Streamlit front-end in [streamlit_app.py](streamlit_app.py)
- A Python package in [hr_policy_navigator](hr_policy_navigator)

## Topic and model

- Topic: HR policy navigator
- Model: qwen 8b instruct via OpenAI-compatible API
- Corpus: Sourcegraph policy and HR docs under [sourcegraph_files](sourcegraph_files)

## Features

- Retrieval-first policy lookup using BM25 and dense embeddings when configured
- Planner → executor → critic loop
- Citation-grounded answers with refusal handling for non-policy requests
- Structured evaluation dataset and metrics hooks
- Streamlit UI and simple rate-limited design for a production frontend

## Setup

### 1) Create and activate a virtual environment

From the project root:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 2) Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3) Configure environment variables

Copy [.env.example](.env.example) to `.env` and fill in your model endpoint settings.

```bash
copy .env.example .env
```

### 4) Run the app

```bash
streamlit run streamlit_app.py
```

If you want to run it explicitly with Python:

```bash
python -m streamlit run streamlit_app.py
```

### 5) Generate or refresh the evaluation set

```bash
python scripts/generate_evaluation_set.py
```

## Important guardrails

- The assistant should only answer questions grounded in the policy corpus.
- Do not approve HR or legal consequences without human review.
- Cite source files and refuse unsupported or personal-action requests.
- Prompt injection attempts are treated as unsafe and should be refused.

## Assignment coverage

This project covers the requested phases for the assignment:

- Specification and refusal boundary
- Prompt-engineering baseline design
- Retrieval over a real corpus with citation grounding
- Multi-agent tool-using architecture
- Evaluation and observability hooks
- Deployment-ready frontend and container-friendly layout
