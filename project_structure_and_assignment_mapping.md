# Assignment → code map

| Requirement | Where | State |
|---|---|---|
| I. Spec + 60–100 eval queries | docs/specification.md, evaluation_set.json (60), evaluation_set_adversarial.json (29) | done; golds need per-question refinement |
| II. Prompt baseline + ≥3 strategies | prompts.py, agent.py (`baseline`,`structured`,`selfconsistency`,`decomposition`), scripts/run_e2e_eval.py | code done; **comparison needs live endpoint** |
| III. RAG: chunking, BM25+dense, reranker, citations, recall@k separate | chunking.py, retrieval.py, scripts/run_retrieval_eval.py, docs/retrieval_results.md | done; full corpus measured with LSA stand-in; re-run with real embedder |
| IV. Agentic layer: ≥3 tools, planner→executor→critic, state, error handling | agent.py, tools.py (4 tools), llm.py, state.py, tests/ | done |
| V. Eval + observability + failure taxonomy | tracing.py, scripts/run_e2e_eval.py, trace_summary.py, Streamlit *Traces* | harness done; **numbers pending** |
| VI. Ship: UI, container, rate limit, cache, responsible AI | streamlit_app.py, serving.py, Dockerfile, docker-compose.yml, guardrails.py, scripts/run_injection_suite.py | done; Docker build not run in this environment |
