"""HR Policy Navigator — Streamlit Application (Amit-Enhanced, Sprint 3).

Sprint history:
  Sprint 1: Login screen, role-aware UI, persistent chat history (E1, E3, E10)
  Sprint 2: Role-based persona answers, tenure personalization, HITL state machine (E2, E4, E8)
  Sprint 3: Voice input (Whisper tiny), Read Aloud (pyttsx3), RLHF thumbs + Feedback Analytics tab (E6, E7, E9)
"""

import json
import uuid
from pathlib import Path

import streamlit as st

from hr_policy_navigator.agent import Copilot
from hr_policy_navigator.auth import UserProfile, authenticate, demo_users
from hr_policy_navigator.chat_history import ChatHistory
from hr_policy_navigator.config import Settings
from hr_policy_navigator.feedback import FeedbackStore
from hr_policy_navigator.retrieval import PolicyCorpus
from hr_policy_navigator.serving import RateLimiter, ResponseCache
from hr_policy_navigator.state import HandoffStore
from hr_policy_navigator.tracing import load_traces, summarize
from hr_policy_navigator.voice import text_to_speech, tts_available, whisper_available

S = Settings()

# ---------------------------------------------------------------------------
# Role metadata
# ---------------------------------------------------------------------------
_ROLE_META = {
    "hr_ops": {
        "icon": "🏢", "label": "HR Operations",
        "hint": "Full access · HITL queue visible · Verbatim policy answers",
        "color": "#c0392b",
        "sample_queries": [
            "What is the full termination policy with all escalation steps?",
            "How do I process a parental leave request?",
            "Can I fire John Doe?  ← triggers safe HITL refusal",
        ],
    },
    "manager": {
        "icon": "👩‍💼", "label": "People Manager",
        "hint": "Team-scoped HITL queue · Manager-layer policy context",
        "color": "#2980b9",
        "sample_queries": [
            "How do I approve a leave request from my team?",
            "What is my responsibility in a disciplinary process?",
            "How is a performance improvement plan initiated?",
        ],
    },
    "associate": {
        "icon": "👤", "label": "Associate",
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
# Cached heavy resources
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

@st.cache_resource
def get_feedback() -> FeedbackStore:
    return FeedbackStore(S.state_dir)

@st.cache_resource
def get_handoff_store() -> HandoffStore:
    return HandoffStore(S.state_dir)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render(res: dict, idx: int = 0, feedback_store: FeedbackStore = None,
           user_role: str = "", user_id: str = "", strategy: str = "") -> None:
    """Render a single agent response with citations, metadata, TTS button, and RLHF thumbs."""
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

    # ── Telemetry + TTS + RLHF row ──
    tele_col, tts_col, rlhf_col = st.columns([3, 1, 2])
    with tele_col:
        st.caption(
            f"{res.get('latency_ms', 0):.0f} ms · {res.get('tool_calls', 0)} tool calls · "
            f"{res.get('tokens', 0)} tokens · ~${res.get('cost_usd', 0):.5f} · trace {res.get('trace_id', '-')}"
        )

    # ── E7: Read Aloud button ──
    with tts_col:
        if tts_available() and res.get("answer") and res["status"] in ("success", "abstained"):
            if st.button("🔊 Read", key=f"tts_{idx}_{res.get('trace_id','x')}", help="Read answer aloud"):
                with st.spinner("Generating audio…"):
                    wav = text_to_speech(res["answer"])
                if wav:
                    st.audio(wav, format="audio/wav")
                else:
                    st.warning("TTS unavailable on this machine.")

    # ── E9: RLHF thumbs ──
    with rlhf_col:
        if feedback_store is not None and res.get("trace_id"):
            rating_key = f"rated_{res['trace_id']}"
            if st.session_state.get(rating_key):
                st.caption(f"{'👍' if st.session_state[rating_key] > 0 else '👎'} Rated — thanks!")
            else:
                r_col1, r_col2 = st.columns(2)
                if r_col1.button("👍", key=f"up_{idx}_{res.get('trace_id')}",
                                 help="This answer was helpful"):
                    feedback_store.save(
                        trace_id=res["trace_id"],
                        query=st.session_state.get("last_query", ""),
                        answer=res.get("answer", ""),
                        rating=1, user_role=user_role, user_id=user_id, strategy=strategy,
                    )
                    st.session_state[rating_key] = 1
                    st.rerun()
                if r_col2.button("👎", key=f"dn_{idx}_{res.get('trace_id')}",
                                 help="This answer needs improvement"):
                    # Show correction field
                    st.session_state[f"show_correction_{res['trace_id']}"] = True
                    st.rerun()

        # Correction input after thumbs-down
        trace_id = res.get("trace_id", "")
        if st.session_state.get(f"show_correction_{trace_id}"):
            with st.form(key=f"correction_form_{trace_id}", clear_on_submit=True):
                corr = st.text_area("What should the answer have been? (optional)", height=60)
                submitted = st.form_submit_button("Submit 👎")
                if submitted and feedback_store is not None:
                    feedback_store.save(
                        trace_id=trace_id,
                        query=st.session_state.get("last_query", ""),
                        answer=res.get("answer", ""),
                        rating=-1, correction=corr,
                        user_role=user_role, user_id=user_id, strategy=strategy,
                    )
                    st.session_state[f"rated_{trace_id}"] = -1
                    st.session_state.pop(f"show_correction_{trace_id}", None)
                    st.rerun()


# ---------------------------------------------------------------------------
# Login screen
# ---------------------------------------------------------------------------
def render_login() -> None:
    st.set_page_config(page_title="HR Policy Navigator — Login", layout="centered")
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
        if st.button("Sign In", type="primary", use_container_width=True):
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
            role_key = "hr_ops" if "hr" in u["role"].lower() else ("manager" if "manager" in u["role"].lower() else "associate")
            meta = _ROLE_META[role_key]
            st.markdown(
                f"""<div style='border-left:4px solid {meta["color"]}; padding:6px 12px; margin-bottom:8px; border-radius:4px; background:#1a1a2e22;'>
                <b>{meta["icon"]} {u["name"]}</b> &nbsp;·&nbsp; <span style='color:#aaa;'>{u["role"]}</span><br/>
                <code>{u["username"]}</code> / <code>{u["hint_password"]}</code>
                </div>""",
                unsafe_allow_html=True,
            )
        st.caption("Demo credentials for capstone presentation only.")

    st.markdown("---")
    st.caption("This tool explains Sourcegraph HR policy. It cannot approve, decide or act on HR matters.")


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

    agent        = load_agent()
    limiter, cache = shared()
    history_store  = get_history()
    feedback_store = get_feedback()

    st.session_state.setdefault("sid", uuid.uuid4().hex[:10])
    st.session_state.setdefault("log", [])

    user_profile_dict = {
        "user_id": user.user_id, "name": user.name,
        "role": user.role, "tenure": user.tenure, "team_id": user.team_id,
    }

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown(f"### {meta['icon']} {user.name}")
        st.caption(f"{user.display_role} · {user.tenure} tenure")
        if st.button("🚪 Sign Out", use_container_width=True):
            for key in ["user", "sid", "log"]:
                st.session_state.pop(key, None)
            st.rerun()

        st.divider()

        with st.expander("⚙️ System", expanded=False):
            st.write(f"Documents: {len(agent.corpus.docs)} · Chunks: {len(agent.corpus.chunks)}")
            st.write(f"Chunking: `{agent.corpus.chunking}`")
            st.write(f"Dense: `{agent.corpus.embedder.name if agent.corpus.embedder else 'off'}`")
            st.write(f"Reranker: `{agent.corpus.reranker_name}`")
            for note in agent.corpus.notes:
                st.warning(note)
            st.write(f"Model: `{S.model}`" if S.has_model else ":red[No model endpoint configured]")
            st.write(f"Cache hits/misses: {cache.hits}/{cache.misses}")
            # Sprint 3 capability flags
            st.write(f"🎙️ Voice input: {'✅ Whisper ready' if whisper_available() else '⚠️ Install openai-whisper'}")
            st.write(f"🔊 Read Aloud: {'✅ pyttsx3 ready' if tts_available() else '⚠️ Install pyttsx3'}")

        strategy = st.selectbox(
            "Prompt strategy",
            ["structured", "selfconsistency", "decomposition", "baseline"],
            index=0,
            help="structured = best quality; decomposition = multi-hop; selfconsistency = majority vote",
        )

        st.divider()

        if st.button("➕ New Conversation", use_container_width=True):
            st.session_state.sid = uuid.uuid4().hex[:10]
            st.session_state.log = []
            st.rerun()

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

**New features (Sprint 3):**
• 🎙️ Click the mic button to speak your question
• 🔊 Click "Read" below any answer to hear it aloud
• 👍 / 👎 Rate answers to help improve the system
"""
            )

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------
    tab_labels = ["💬 Ask", "📊 Traces", "👍 Feedback"]
    if user.role in ("hr_ops", "manager"):
        tab_labels.append("🧑‍💼 Human Review Queue")

    tabs = st.tabs(tab_labels)
    tab_chat    = tabs[0]
    tab_traces  = tabs[1]
    tab_fb      = tabs[2]
    tab_review  = tabs[3] if user.role in ("hr_ops", "manager") else None

    # ==================================================================
    # E6: Voice Input helpers (rendered inside chat tab)
    # ==================================================================
    def _voice_input_widget() -> str:
        """Renders mic widget; returns transcribed text or '' if unavailable/no audio."""
        if not whisper_available():
            with st.expander("🎙️ Voice Input (unavailable — install openai-whisper)"):
                st.caption(
                    "Run: `.venv\\Scripts\\pip install openai-whisper` then restart the app.\n"
                    "Whisper tiny model (~39MB) loads on first use."
                )
            return ""

        # Streamlit 1.31+ has st.audio_input()
        try:
            audio_val = st.audio_input("🎙️ Speak your question", key="mic_input")
        except AttributeError:
            st.caption("🎙️ Voice input requires Streamlit ≥ 1.31. Update with `pip install -U streamlit`.")
            return ""

        if audio_val is None:
            return ""

        audio_bytes = audio_val.getvalue()
        if not audio_bytes:
            return ""

        # Only transcribe when new audio arrives (avoid re-transcribing on every rerun)
        import hashlib
        audio_hash = hashlib.md5(audio_bytes).hexdigest()[:12]
        if st.session_state.get("last_audio_hash") == audio_hash:
            return st.session_state.get("last_transcript", "")

        with st.spinner("🎙️ Transcribing…"):
            from hr_policy_navigator.voice import transcribe
            transcript = transcribe(audio_bytes)

        st.session_state["last_audio_hash"] = audio_hash
        st.session_state["last_transcript"] = transcript
        return transcript

    # ==================================================================
    # Chat Tab
    # ==================================================================
    with tab_chat:
        # Persona active badge
        tenure_labels = {"new": "New Hire (<1 yr)", "mid": "1–3 yrs", "senior": "Tenured (3+ yrs)"}
        st.markdown(
            f"<span style='background:{meta['color']}33; border:1px solid {meta['color']}; "
            f"border-radius:12px; padding:3px 10px; font-size:0.82rem;'>"
            f"🎭 Persona: <b>{user.display_role}</b> · {tenure_labels.get(user.tenure, user.tenure)}"
            f"</span>",
            unsafe_allow_html=True,
        )
        st.write("")

        # Replay conversation log with RLHF buttons
        for i, (turn_q, turn_r) in enumerate(st.session_state.log):
            with st.chat_message("user"):
                st.write(turn_q)
            with st.chat_message("assistant"):
                render(
                    turn_r, idx=i,
                    feedback_store=feedback_store,
                    user_role=user.role, user_id=user.user_id,
                    strategy=strategy,
                )

        # ── E6: Voice input widget ──
        voice_query = _voice_input_widget()
        if voice_query:
            st.info(f"🎙️ Transcribed: *\"{voice_query}\"*")

        # ── Text chat input (also pre-filled from voice) ──
        q = st.chat_input("Ask about benefits, leave, remote work, conduct…")
        # Accept voice transcript if no text typed
        final_q = q or (voice_query if voice_query else None)

        if final_q:
            st.session_state["last_query"] = final_q
            ok, wait = limiter.check(st.session_state.sid)
            if not ok:
                st.warning(f"⏱️ Rate limit reached. Please wait {wait:.0f}s.")
            else:
                res = cache.get(final_q, strategy) if len(st.session_state.log) == 0 else None
                if res is None:
                    with st.spinner("🔍 Searching policies…"):
                        res = agent.run(
                            final_q, st.session_state.sid, strategy,
                            user_profile=user_profile_dict,
                        )
                    if len(st.session_state.log) == 0:
                        cache.put(final_q, strategy, res)

                st.session_state.log.append((final_q, res))
                history_store.append(user.user_id, st.session_state.sid, final_q, res)
                # Reset voice transcript so it doesn't re-fire
                st.session_state.pop("last_audio_hash", None)
                st.session_state.pop("last_transcript", None)
                st.rerun()

    # ==================================================================
    # Traces Tab
    # ==================================================================
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
                [{"trace": t["trace_id"], "query": t["meta"].get("query", "")[:80],
                  "status": t["result"]["status"], "ms": t["duration_ms"],
                  **{k: v for k, v in t["counters"].items()}}
                 for t in reversed(traces[-200:])],
                use_container_width=True,
            )
            pick = st.selectbox("Inspect trace", [t["trace_id"] for t in reversed(traces[-50:])])
            st.json(next(t for t in traces if t["trace_id"] == pick))

    # ==================================================================
    # E9: Feedback Analytics Tab
    # ==================================================================
    with tab_fb:
        st.markdown("### 👍 RLHF Feedback Analytics")
        st.caption(
            "Real user ratings collected via thumbs-up / thumbs-down on every answer. "
            "Negative-rated queries are flagged as improvement opportunities."
        )

        fb = feedback_store.analytics()

        if fb["total"] == 0:
            st.info("No feedback yet. Rate some answers in the Ask tab to populate this dashboard.")
            st.markdown(
                """
                **How the RLHF loop works:**
                1. User rates an answer 👍 or 👎 in the Ask tab
                2. Rating + optional correction saved to `state/feedback_log.jsonl`
                3. This dashboard surfaces low-rated queries as improvement opportunities
                4. `scripts/analyze_feedback.py` *(coming soon)* clusters negatives and proposes prompt patches
                5. Prompt is updated → re-evaluated on the 84-query frozen benchmark → metrics improve

                > This closes the **human-in-the-loop learning cycle** without any model fine-tuning —
                > pure prompt engineering driven by real user signal.
                """
            )
        else:
            # Top-line metrics
            mc = st.columns(4)
            mc[0].metric("Total Ratings", fb["total"])
            mc[1].metric("👍 Positive", fb["positive"])
            mc[2].metric("👎 Negative", fb["negative"])
            mc[3].metric("Satisfaction", f"{fb['pct_positive']}%",
                         delta=f"{fb['pct_positive'] - 70:.1f}%" if fb["total"] > 5 else None,
                         delta_color="normal")

            st.divider()
            left, right = st.columns(2)

            with left:
                st.markdown("#### By Role")
                if fb["by_role"]:
                    role_data = [
                        {"Role": role, "👍 Positive": v["pos"], "👎 Negative": v["neg"], "Total": v["total"]}
                        for role, v in fb["by_role"].items()
                    ]
                    st.dataframe(role_data, use_container_width=True, hide_index=True)

                st.markdown("#### By Strategy")
                if fb["by_strategy"]:
                    strat_data = [
                        {"Strategy": s, "👍": v["pos"], "👎": v["neg"], "Total": v["total"]}
                        for s, v in fb["by_strategy"].items()
                    ]
                    st.dataframe(strat_data, use_container_width=True, hide_index=True)

            with right:
                st.markdown("#### 🔴 Top Improvement Opportunities")
                st.caption("Queries with the most thumbs-down ratings — investigate these prompts first.")
                if fb["low_rated"]:
                    for item in fb["low_rated"]:
                        with st.container(border=True):
                            st.markdown(f"**{item['count']}× 👎** — *\"{item['query']}\"*")
                            if item.get("answer_snippet"):
                                st.caption(f"Answer: {item['answer_snippet'][:100]}…")
                else:
                    st.success("No negatively-rated queries yet!")

                st.markdown("#### 🟢 Top Positive Queries")
                if fb["high_rated"]:
                    for item in fb["high_rated"]:
                        st.markdown(f"**{item['count']}× 👍** — *\"{item['query']}\"*")

            st.divider()
            if fb["corrections"]:
                st.markdown("#### ✏️ User Corrections (most recent)")
                st.caption("These are suggested better answers from users who clicked 👎.")
                for c in fb["corrections"]:
                    with st.container(border=True):
                        st.caption(c["ts"])
                        st.markdown(f"**Q:** *{c['query']}*")
                        st.markdown(f"**Suggested correction:** {c['correction']}")

            st.divider()
            st.markdown("#### 📄 Raw Feedback Log")
            with st.expander("View all feedback records"):
                st.dataframe(feedback_store.all(), use_container_width=True)

    # ==================================================================
    # Human Review Queue Tab (HR Ops + Manager)
    # ==================================================================
    if tab_review is not None:
        with tab_review:
            handoff_store = HandoffStore(S.state_dir)
            rows = handoff_store.all(role=user.role, team_id=user.team_id)

            st.markdown("### 🧑‍💼 Human Review Queue")
            st.caption(
                "Requests the assistant declined to decide automatically. "
                "A person must handle these — nothing has been approved or changed."
            )

            if rows:
                pending_n = sum(1 for r in rows if r.get("ui_status", "pending") == "pending")
                ack_n     = sum(1 for r in rows if r.get("ui_status") == "acknowledged")
                res_n     = sum(1 for r in rows if r.get("ui_status") in ("resolved", "escalated"))
                mc = st.columns(3)
                mc[0].metric("🔴 Pending", pending_n)
                mc[1].metric("🟡 Acknowledged", ack_n)
                mc[2].metric("🟢 Resolved / Escalated", res_n)
                st.divider()

            if not rows:
                st.success("✅ Queue is empty — no pending requests.")
            else:
                from hr_policy_navigator.state import STATUS_LABELS, VALID_TRANSITIONS
                for row in rows:
                    ticket_id      = row.get("ticket_id", "unknown")
                    current_status = row.get("ui_status", "pending")
                    allowed_next   = VALID_TRANSITIONS.get(current_status, [])

                    with st.container(border=True):
                        hdr_col, badge_col = st.columns([3, 1])
                        with hdr_col:
                            st.markdown(
                                f"**🎫 Ticket:** `{ticket_id}`  \n"
                                f"**Category:** `{row.get('reason', row.get('category', '—'))}`  \n"
                                f"**Query:** {row.get('topic', row.get('query', '—'))[:160]}"
                            )
                            role_lbl = row.get("user_role", "")
                            st.caption(
                                f"🕐 {row.get('timestamp', 'unknown')}"
                                + (f"  ·  Role: {role_lbl}" if role_lbl else "")
                                + (f"  ·  Session: {row.get('session_id','')[:12]}" if row.get("session_id") else "")
                            )
                        with badge_col:
                            st.markdown(f"**{STATUS_LABELS.get(current_status, '🔴 Pending')}**")

                        if allowed_next:
                            btn_cols = st.columns(len(allowed_next) + 1)
                            for i, next_st in enumerate(allowed_next):
                                btn_label = {
                                    "acknowledged": "✅ Acknowledge",
                                    "resolved": "🟢 Mark Resolved",
                                    "escalated": "🔺 Escalate",
                                }.get(next_st, next_st.capitalize())
                                if btn_cols[i].button(btn_label, key=f"{ticket_id}_{next_st}"):
                                    handoff_store.update_status(ticket_id, next_st, actor=user.name)
                                    st.toast(f"Ticket {ticket_id[:10]}… → {next_st}", icon="✅")
                                    st.rerun()
                            with st.expander("📝 Add resolution note"):
                                note_text = st.text_area("Note", key=f"note_{ticket_id}", height=80,
                                                         placeholder="Describe action taken…")
                                if st.button("Save Note", key=f"savenote_{ticket_id}"):
                                    if note_text.strip():
                                        handoff_store.add_note(ticket_id, note_text.strip(), actor=user.name)
                                        st.toast("Note saved.", icon="📝")
                                        st.rerun()
                        else:
                            st.caption("✅ This ticket is closed — no further actions available.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def app() -> None:
    if "user" not in st.session_state:
        render_login()
    else:
        main()


if __name__ == "__main__":
    app()
