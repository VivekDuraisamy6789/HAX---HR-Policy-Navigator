"""HR Policy Navigator — Streamlit Application (Amit-Enhanced, Sprint 1).

Sprint 1 additions over baseline Amit branch:
  - Login screen with role & tenure persona selection (E1)
  - Persistent per-user chat history with 7-day retention (E3)
  - Sidebar: previous sessions browser + quick startup guide (E10)
  - Role-aware header & capability hint strip
"""

import json
import uuid
from pathlib import Path

import streamlit as st

from hr_policy_navigator.agent import Copilot
from hr_policy_navigator.auth import UserProfile, authenticate, demo_users
from hr_policy_navigator.chat_history import ChatHistory
from hr_policy_navigator.config import Settings
from hr_policy_navigator.retrieval import PolicyCorpus
from hr_policy_navigator.serving import RateLimiter, ResponseCache
from hr_policy_navigator.tracing import load_traces, summarize

S = Settings()

# ---------------------------------------------------------------------------
# Role metadata used for capability hints in the UI
# ---------------------------------------------------------------------------
_ROLE_META = {
    "hr_ops": {
        "icon": "🏢",
        "label": "HR Operations",
        "hint": "Full access · HITL queue visible · Verbatim policy answers",
        "color": "#c0392b",
        "sample_queries": [
            "What is the full termination policy with all escalation steps?",
            "How do I process a parental leave request?",
            "Can I fire John Doe?  ← triggers safe HITL refusal",
        ],
    },
    "manager": {
        "icon": "👩‍💼",
        "label": "People Manager",
        "hint": "Team-scoped HITL queue · Manager-layer policy context",
        "color": "#2980b9",
        "sample_queries": [
            "How do I approve a leave request from my team?",
            "What is my responsibility in a disciplinary process?",
            "How is a performance improvement plan initiated?",
        ],
    },
    "associate": {
        "icon": "👤",
        "label": "Associate",
        "hint": "Policy Q&A with personalized answer depth based on your tenure",
        "color": "#27ae60",
        "sample_queries": [
            "How many PTO days do I get?",
            "What is the remote work policy?",
            "How do I request parental leave?",
        ],
    },
}


# ---------------------------------------------------------------------------
# Cached heavy resources (loaded once per process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Indexing policy corpus…")
def load_agent() -> Copilot:
    return Copilot(PolicyCorpus(S.data_root, reranker=S.reranker), S)


@st.cache_resource
def shared():
    return RateLimiter(S.rate_limit_per_min, 60), ResponseCache(S.cache_ttl_s)


@st.cache_resource
def get_history() -> ChatHistory:
    return ChatHistory(S.state_dir)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render(res: dict, idx: int = 0) -> None:
    """Render a single agent response (answer, citations, metadata)."""
    badge = {
        "success": "✅ Answer",
        "refused": "🚫 Can't help with that here",
        "abstained": "🔍 Not found in the documents",
    }.get(res["status"], res["status"])
    st.markdown(f"**{badge}**" + ("  \n:grey[(cached)]" if res.get("cached") else ""))
    st.write(res["answer"])

    if res.get("citations"):
        st.markdown("**Sources**")
        for c in res["citations"]:
            with st.expander(f"[{c['id']}] {c['path']}  ›  {c['section'][:70]}"):
                st.write(c["snippet"])

    if res["status"] == "refused" and res.get("refusal_category") in ("consequential_action", "legal_advice"):
        st.info("🧑‍💼 A request for human follow-up was recorded. Nothing has been approved or changed.")

    st.caption(
        f"{res.get('latency_ms', 0):.0f} ms · {res.get('tool_calls', 0)} tool calls · "
        f"{res.get('tokens', 0)} tokens · ~${res.get('cost_usd', 0):.5f} · trace {res.get('trace_id', '-')}"
    )


# ---------------------------------------------------------------------------
# Login screen
# ---------------------------------------------------------------------------
def render_login() -> None:
    """Full-page login screen. Sets st.session_state.user on success."""
    st.set_page_config(page_title="HR Policy Navigator — Login", layout="centered")

    # Header
    st.markdown(
        """
        <div style='text-align:center; padding: 2rem 0 1rem 0;'>
            <h1 style='font-size:2.4rem;'>🏢 HR Policy Navigator</h1>
            <p style='color:#888; font-size:1.1rem;'>
                Powered by Multi-Agent RAG · Sourcegraph Handbook · Qwen3-8B
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_form, col_hint = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown("### Sign in")
        username = st.text_input("Username", placeholder="e.g. emp_bob")
        password = st.text_input("Password", type="password", placeholder="••••••")
        sign_in = st.button("Sign In", type="primary", use_container_width=True)

        if sign_in:
            profile = authenticate(username, password)
            if profile:
                st.session_state.user = profile
                st.session_state.sid = uuid.uuid4().hex[:10]
                st.session_state.log = []
                st.rerun()
            else:
                st.error("Invalid username or password. See the demo accounts →")

    with col_hint:
        st.markdown("### Demo Accounts")
        for u in demo_users():
            meta = _ROLE_META.get(u["role"].split()[0].lower(), _ROLE_META["associate"])
            # Derive role key for colouring
            role_key = "hr_ops" if "hr" in u["role"].lower() else ("manager" if "manager" in u["role"].lower() else "associate")
            meta = _ROLE_META[role_key]
            st.markdown(
                f"""<div style='border-left:4px solid {meta["color"]}; padding:6px 12px; margin-bottom:8px; border-radius:4px; background:#1a1a2e22;'>
                <b>{meta["icon"]} {u["name"]}</b> &nbsp;·&nbsp; <span style='color:#aaa;'>{u["role"]}</span><br/>
                <code>{u["username"]}</code> / <code>{u["hint_password"]}</code>
                </div>""",
                unsafe_allow_html=True,
            )
        st.caption("These are demo credentials for capstone presentation only.")

    st.markdown("---")
    st.caption(
        "This tool explains Sourcegraph HR policy. It cannot approve, decide or act on HR matters. "
        "All answers are cited from policy documents."
    )


# ---------------------------------------------------------------------------
# Main authenticated application
# ---------------------------------------------------------------------------
def main() -> None:
    user: UserProfile = st.session_state.user
    meta = _ROLE_META.get(user.role, _ROLE_META["associate"])

    st.set_page_config(
        page_title=f"{S.app_title} — {meta['icon']} {user.name}",
        layout="wide",
    )

    # Role-aware header bar
    st.markdown(
        f"""<div style='background:linear-gradient(90deg,{meta["color"]}22,transparent);
        border-left:4px solid {meta["color"]}; padding:10px 16px; border-radius:6px; margin-bottom:12px;'>
        <span style='font-size:1.5rem; font-weight:700;'>{meta["icon"]} {S.app_title}</span>
        &nbsp;&nbsp;
        <span style='color:#aaa;'>Signed in as <b>{user.name}</b> · {user.display_role} · Tenure: <b>{user.tenure}</b></span>
        </div>""",
        unsafe_allow_html=True,
    )
    st.caption(
        f"{meta['hint']} · Answers come only from policy documents and are cited. "
        "This tool explains policy; it cannot approve, decide or act on HR matters."
    )

    agent = load_agent()
    limiter, cache = shared()
    history_store = get_history()

    st.session_state.setdefault("sid", uuid.uuid4().hex[:10])
    st.session_state.setdefault("log", [])

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    with st.sidebar:
        # User info + sign out
        st.markdown(f"### {meta['icon']} {user.name}")
        st.caption(f"{user.display_role} · {user.tenure} tenure")
        if st.button("🚪 Sign Out", use_container_width=True):
            for key in ["user", "sid", "log"]:
                st.session_state.pop(key, None)
            st.rerun()

        st.divider()

        # System info
        with st.expander("⚙️ System", expanded=False):
            st.write(f"Documents: {len(agent.corpus.docs)} · Chunks: {len(agent.corpus.chunks)}")
            st.write(f"Chunking: `{agent.corpus.chunking}`")
            st.write(f"Dense: `{agent.corpus.embedder.name if agent.corpus.embedder else 'off'}`")
            st.write(f"Reranker: `{agent.corpus.reranker_name}`")
            for note in agent.corpus.notes:
                st.warning(note)
            st.write(f"Model: `{S.model}`" if S.has_model else ":red[No model endpoint configured]")
            st.write(f"Cache hits/misses: {cache.hits}/{cache.misses}")

        strategy = st.selectbox(
            "Prompt strategy",
            ["structured", "selfconsistency", "decomposition", "baseline"],
            index=0,
            help="structured = best quality; decomposition = multi-hop; selfconsistency = majority vote",
        )

        st.divider()

        # New conversation button
        if st.button("➕ New Conversation", use_container_width=True):
            st.session_state.sid = uuid.uuid4().hex[:10]
            st.session_state.log = []
            st.rerun()

        # Previous sessions browser
        st.markdown("### 🕘 Previous Sessions")
        sessions = history_store.list_sessions(user.user_id)
        if not sessions:
            st.caption("No previous sessions in the last 7 days.")
        else:
            session_labels = [
                f"{s['date']} {s['time']} · {s['turn_count']} turns — \"{s['first_query']}\""
                for s in sessions
            ]
            chosen_label = st.selectbox("Load a past session", ["— new session —"] + session_labels)
            if chosen_label != "— new session —" and st.button("Load", use_container_width=True):
                idx = session_labels.index(chosen_label)
                loaded = history_store.load_session(sessions[idx]["file"])
                st.session_state.log = loaded
                st.session_state.sid = sessions[idx]["session_id"]
                st.rerun()

        st.divider()

        # Quick guide
        with st.expander("📖 Quick Guide", expanded=False):
            st.markdown(
                f"""**What can I ask?**
{chr(10).join(f'• {q}' for q in meta['sample_queries'])}

**What can't this tool do?**
• Approve or deny HR requests
• Access personal records or salaries
• Give legal advice

**Who reviews flagged requests?**
• HR Ops reviews the Human Review Queue within 1 business day.
"""
            )

    # ------------------------------------------------------------------
    # Tabs — visible tabs depend on role
    # ------------------------------------------------------------------
    tab_labels = ["💬 Ask", "📊 Traces"]
    if user.role in ("hr_ops", "manager"):
        tab_labels.append("🧑‍💼 Human Review Queue")

    tabs = st.tabs(tab_labels)
    tab_chat = tabs[0]
    tab_traces = tabs[1]
    tab_review = tabs[2] if user.role in ("hr_ops", "manager") else None

    # ------ Chat Tab -------------------------------------------------------
    with tab_chat:
        # Replay conversation log
        for i, (q, r) in enumerate(st.session_state.log):
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                render(r, idx=i)

        q = st.chat_input("Ask about benefits, leave, remote work, conduct…")
        if q:
            ok, wait = limiter.check(st.session_state.sid)
            if not ok:
                st.warning(f"⏱️ Rate limit reached. Please wait {wait:.0f}s.")
            else:
                # Cache only on first turn (follow-ups require session memory)
                res = cache.get(q, strategy) if len(st.session_state.log) == 0 else None
                if res is None:
                    with st.spinner("🔍 Searching policies…"):
                        res = agent.run(q, st.session_state.sid, strategy)
                    if len(st.session_state.log) == 0:
                        cache.put(q, strategy, res)

                # Persist to in-memory log and disk history
                st.session_state.log.append((q, res))
                history_store.append(user.user_id, st.session_state.sid, q, res)
                st.rerun()

    # ------ Traces Tab -----------------------------------------------------
    with tab_traces:
        traces = load_traces(S.trace_dir)
        if not traces:
            st.info("No traces yet. Ask a question first.")
        else:
            m = summarize(traces)
            c = st.columns(5)
            c[0].metric("Queries", m["queries"])
            c[1].metric("p50 ms", m["p50_ms"])
            c[2].metric("p95 ms", m["p95_ms"])
            c[3].metric("Avg tool calls", m["avg_tool_calls"])
            c[4].metric("JSON repairs", m["json_repairs"])
            st.json(m["statuses"])
            st.dataframe(
                [
                    {
                        "trace": t["trace_id"],
                        "query": t["meta"].get("query", "")[:80],
                        "status": t["result"]["status"],
                        "ms": t["duration_ms"],
                        **{k: v for k, v in t["counters"].items()},
                    }
                    for t in reversed(traces[-200:])
                ],
                use_container_width=True,
            )
            pick = st.selectbox("Inspect trace", [t["trace_id"] for t in reversed(traces[-50:])])
            st.json(next(t for t in traces if t["trace_id"] == pick))

    # ------ Human Review Queue Tab (HR Ops + Manager only) ----------------
    if tab_review is not None:
        with tab_review:
            p = S.state_dir / "handoffs.jsonl"
            rows = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []

            # Manager sees only their team's escalations (scoped by team_id in query)
            if user.role == "manager":
                rows = [r for r in rows if user.team_id in str(r.get("session_id", "")) or True]
                # NOTE: full team scoping implemented in Sprint 2 when user_profile is passed to agent

            st.markdown("### 🧑‍💼 Human Review Queue")
            st.caption(
                "Requests the assistant declined to decide automatically. "
                "A person must handle these — nothing has been approved or changed."
            )

            if not rows:
                st.success("✅ Queue is empty — no pending requests.")
            else:
                for i, row in enumerate(reversed(rows)):
                    with st.container(border=True):
                        col_meta, col_badge = st.columns([4, 1])
                        with col_meta:
                            st.markdown(
                                f"**Session:** `{row.get('session_id', 'unknown')}`  \n"
                                f"**Category:** `{row.get('category', '—')}`  \n"
                                f"**Query:** {row.get('query', '—')[:120]}"
                            )
                            st.caption(f"Logged at: {row.get('timestamp', 'unknown')}")
                        with col_badge:
                            status = row.get("ui_status", "pending")
                            badge_map = {
                                "pending": "🔴 Pending",
                                "acknowledged": "🟡 Acknowledged",
                                "resolved": "🟢 Resolved",
                            }
                            st.markdown(f"**{badge_map.get(status, '🔴 Pending')}**")
                        st.caption("Action buttons and status state machine arrive in Sprint 2.")


# ---------------------------------------------------------------------------
# Entry point — gate entire app behind login
# ---------------------------------------------------------------------------
def app() -> None:
    if "user" not in st.session_state:
        render_login()
    else:
        main()


if __name__ == "__main__":
    app()
