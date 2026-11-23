"""Tests for Enhanced Enterprise features (Sprint 1-4 & System Options)."""
import tempfile
from pathlib import Path
from hr_policy_navigator.auth import authenticate, demo_users
from hr_policy_navigator.chat_history import ChatHistory
from hr_policy_navigator.feedback import FeedbackStore


def test_auth_demo_users():
    users = demo_users()
    assert len(users) >= 4
    for u in users:
        auth_res = authenticate(u["username"], u["hint_password"])
        assert auth_res is not None
        assert auth_res.name == u["name"]
        assert auth_res.user_id.startswith("u")
    assert authenticate("bad_user", "bad_pass") is None


def test_chat_history_rename_and_limit():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChatHistory(state_dir=tmpdir)
        uid = "test_user_1"
        
        # Create 5 distinct sessions
        for i in range(5):
            sid = f"session_{i}"
            store.append(uid, sid, f"Query {i}", {"status": "success", "answer": f"Answer {i}"})

        # List with limit=3 (default)
        sessions = store.list_sessions(uid, limit=3)
        assert len(sessions) == 3

        # Rename a session
        store.rename_session(uid, "session_4", "My Custom Title")
        assert store.get_title(uid, "session_4") == "My Custom Title"

        sessions_after = store.list_sessions(uid, limit=3)
        assert any(s["title"] == "My Custom Title" for s in sessions_after)


def test_feedback_privacy_filtering():
    with tempfile.TemporaryDirectory() as tmpdir:
        fb = FeedbackStore(state_dir=tmpdir)
        
        fb.save("trace_1", "Q1", "A1", 1, user_role="associate", user_id="user_a")
        fb.save("trace_2", "Q2", "A2", -1, user_role="manager", user_id="user_b", correction="Better answer")
        fb.save("trace_3", "Q3", "A3", 1, user_role="associate", user_id="user_a")

        # Global analytics (HR Ops view)
        global_an = fb.analytics()
        assert global_an["total"] == 3
        assert len(fb.all()) == 3

        # User A scoped analytics
        user_a_an = fb.analytics(user_id="user_a")
        assert user_a_an["total"] == 2
        assert user_a_an["positive"] == 2
        assert user_a_an["negative"] == 0
        assert len(fb.all(user_id="user_a")) == 2

        # User B scoped analytics
        user_b_an = fb.analytics(user_id="user_b")
        assert user_b_an["total"] == 1
        assert user_b_an["negative"] == 1
        assert len(user_b_an["corrections"]) == 1
        assert len(fb.all(user_id="user_b")) == 1
