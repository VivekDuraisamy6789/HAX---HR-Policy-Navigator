import json
import uuid
from pathlib import Path

import streamlit as st

from hr_policy_navigator.agent import Copilot
from hr_policy_navigator.config import Settings
from hr_policy_navigator.retrieval import PolicyCorpus
from hr_policy_navigator.serving import RateLimiter, ResponseCache
from hr_policy_navigator.tracing import load_traces, summarize

S = Settings()


@st.cache_resource(show_spinner="Indexing policy corpus...")
def load_agent() -> Copilot:
    return Copilot(PolicyCorpus(S.data_root, reranker=S.reranker), S)


@st.cache_resource
def shared():
    return RateLimiter(S.rate_limit_per_min, 60), ResponseCache(S.cache_ttl_s)


def render(res: dict) -> None:
    badge = {"success": "Answer", "refused": "Can't help with that here", "abstained": "Not found in the documents"}.get(res["status"], res["status"])
    st.markdown(f"**{badge}**" + ("  \n:grey[(cached)]" if res.get("cached") else ""))
    st.write(res["answer"])
    if res.get("citations"):
        st.markdown("**Sources**")
        for c in res["citations"]:
            with st.expander(f"[{c['id']}] {c['path']}  ›  {c['section'][:70]}"):
                st.write(c["snippet"])
    if res["status"] == "refused" and res.get("refusal_category") in ("consequential_action", "legal_advice"):
        st.info("A request for human follow-up was recorded. Nothing has been approved or changed.")
    st.caption(f"{res.get('latency_ms', 0):.0f} ms · {res.get('tool_calls', 0)} tool calls · {res.get('tokens', 0)} tokens · ~${res.get('cost_usd', 0):.5f} · trace {res.get('trace_id', '-')}")


def main() -> None:
    st.set_page_config(page_title=S.app_title, layout="wide")
    st.title(S.app_title)
    st.caption("Answers come only from the policy documents and are cited. This tool explains policy; it cannot approve, decide or act on HR matters.")
    agent = load_agent()
    limiter, cache = shared()
    st.session_state.setdefault("sid", uuid.uuid4().hex[:10])
    st.session_state.setdefault("log", [])

    with st.sidebar:
        st.markdown("### System")
        st.write(f"Documents: {len(agent.corpus.docs)} · Chunks: {len(agent.corpus.chunks)}")
        st.write(f"Chunking: `{agent.corpus.chunking}` · dense: `{agent.corpus.embedder.name if agent.corpus.embedder else 'off'}`")
        st.write(f"Reranker: `{agent.corpus.reranker_name}`")
        for note in agent.corpus.notes:
            st.warning(note)
        st.write(f"Model: `{S.model}`" if S.has_model else ":red[No model endpoint configured]")
        strategy = st.selectbox("Prompt strategy", ["structured", "selfconsistency", "decomposition", "baseline"], index=0)
        st.write(f"Cache hits/misses: {cache.hits}/{cache.misses}")
        if st.button("New conversation"):
            st.session_state.sid, st.session_state.log = uuid.uuid4().hex[:10], []

    tab_chat, tab_traces, tab_review = st.tabs(["Ask", "Traces", "Human review queue"])
    with tab_chat:
        for q, r in st.session_state.log:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                render(r)
        q = st.chat_input("Ask about benefits, leave, remote work, conduct...")
        if q:
            ok, wait = limiter.check(st.session_state.sid)
            if not ok:
                st.warning(f"Rate limit reached. Please wait {wait:.0f}s.")
            else:
                res = cache.get(q, strategy) if len(st.session_state.log) == 0 else None  # follow-ups depend on memory: no cache
                if res is None:
                    with st.spinner("Searching policies..."):
                        res = agent.run(q, st.session_state.sid, strategy)
                    if len(st.session_state.log) == 0:
                        cache.put(q, strategy, res)
                st.session_state.log.append((q, res))
                st.rerun()

    with tab_traces:
        traces = load_traces(S.trace_dir)
        if not traces:
            st.info("No traces yet. Ask a question first.")
        else:
            m = summarize(traces)
            c = st.columns(5)
            c[0].metric("Queries", m["queries"]); c[1].metric("p50 ms", m["p50_ms"]); c[2].metric("p95 ms", m["p95_ms"])
            c[3].metric("Avg tool calls", m["avg_tool_calls"]); c[4].metric("JSON repairs", m["json_repairs"])
            st.json(m["statuses"])
            st.dataframe([{"trace": t["trace_id"], "query": t["meta"].get("query", "")[:80], "status": t["result"]["status"],
                           "ms": t["duration_ms"], **{k: v for k, v in t["counters"].items()}} for t in reversed(traces[-200:])], width="stretch")
            pick = st.selectbox("Inspect trace", [t["trace_id"] for t in reversed(traces[-50:])])
            st.json(next(t for t in traces if t["trace_id"] == pick))

    with tab_review:
        p = S.state_dir / "handoffs.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
        st.write("Requests the assistant declined to decide. A person must handle these.")
        st.dataframe(rows, width="stretch") if rows else st.caption("Queue is empty.")


if __name__ == "__main__":
    main()
