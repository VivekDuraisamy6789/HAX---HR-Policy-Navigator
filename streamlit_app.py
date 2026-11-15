import json
from pathlib import Path

import streamlit as st

from hr_policy_navigator.agent import PlannerExecutorCriticAgent
from hr_policy_navigator.config import APP_TITLE, DATA_ROOT
from hr_policy_navigator.evaluation import aggregate_metrics, load_evaluation_set
from hr_policy_navigator.retrieval import PolicyCorpus


@st.cache_resource
def load_corpus() -> PolicyCorpus:
    return PolicyCorpus(DATA_ROOT)


@st.cache_resource
def load_agent() -> PlannerExecutorCriticAgent:
    return PlannerExecutorCriticAgent(load_corpus())


def render_result(result: dict) -> None:
    st.subheader("Answer")
    st.write(result.get("answer", "No answer available."))

    citations = result.get("citations") or []
    if citations:
        st.caption("Citations")
        for item in citations:
            st.write(f"- {item}")

    st.caption(f"Status: {result.get('status', 'unknown')} | Confidence: {result.get('confidence', 'unknown')}")
    if result.get("follow_up"):
        st.info(result["follow_up"])


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Grounded HR policy navigator powered by a qwen 8b-compatible chat endpoint and citation-based retrieval.")

    corpus = load_corpus()
    agent = load_agent()

    with st.sidebar:
        st.markdown("### System status")
        st.write(f"Corpus documents: {len(corpus.documents)}")
        st.write(f"Chunk count: {len(corpus.chunks)}")
        st.write(f"Model config available: {bool(agent.model.client is not None)}")

    tab_main, tab_eval, tab_docs = st.tabs(["Ask policy", "Evaluation", "Corpus explorer"])

    with tab_main:
        query = st.text_area("Ask a question about the HR policy corpus:", height=150)
        if st.button("Ask") and query.strip():
            result = agent.run(query)
            render_result(result)
        elif query.strip():
            st.info("Enter a policy question to start.")

    with tab_eval:
        eval_path = Path("evaluation_set.json")
        if eval_path.exists():
            questions = load_evaluation_set(eval_path)
            st.write(f"Loaded {len(questions)} evaluation entries.")
            if st.button("Run quick evaluation"):
                rows = []
                for entry in questions:
                    result = agent.run(entry["question"])
                    rows.append({
                        "id": entry["id"],
                        "question": entry["question"],
                        "status": result.get("status", "unknown"),
                        "citations": result.get("citations", []),
                        "answer": result.get("answer", ""),
                    })
                metrics = aggregate_metrics(rows)
                st.write(metrics)
                df = st.dataframe(rows, use_container_width=True)
            else:
                st.dataframe([
                    {"id": item["id"], "question": item["question"]} for item in questions[:10]
                ], use_container_width=True)
        else:
            st.warning("No evaluation_set.json file found. Generate it first with the script in scripts/generate_evaluation_set.py.")

    with tab_docs:
        for hit in corpus.search("health benefits employee eligibility", top_k=5):
            st.markdown(f"### {hit.title}")
            st.write(f"Source: {hit.file_path}")
            st.write(hit.content[:500])


if __name__ == "__main__":
    main()
