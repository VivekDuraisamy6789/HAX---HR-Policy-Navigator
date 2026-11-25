"""Tests for Enhanced Enterprise features (Sprint 1-4 & System Options)."""
import tempfile
from pathlib import Path
from hr_policy_navigator.auth import authenticate, demo_users
from hr_policy_navigator.chat_history import ChatHistory
from hr_policy_navigator.feedback import FeedbackStore


def test_auth_demo_users():
    users = demo_users()
    assert len(users) >= 5
    usernames = {u["username"] for u in users}
    assert "sys_admin" in usernames
    for u in users:
        auth_res = authenticate(u["username"], u["hint_password"])
        assert auth_res is not None
        assert auth_res.name == u["name"]
        assert auth_res.user_id.startswith("u")
    sys_admin = authenticate("sys_admin", "sys123")
    assert sys_admin is not None
    assert sys_admin.role == "sys_admin"
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


def test_eval_runner_all_weeks_mock():
    from hr_policy_navigator.config import Settings
    from hr_policy_navigator.eval_runner import (
        generate_week_wise_report,
        load_eval_datasets,
        run_week1_eval,
        run_week2_eval,
        run_week3_eval,
        run_week4_eval,
    )

    domain, adv = load_eval_datasets(n_domain=2, n_adv=1)
    assert len(domain) == 2
    assert len(adv) == 1

    s = Settings()
    res = {
        "week1": run_week1_eval(domain, adv, s, use_mock=True, strategies=["zero_shot"]),
        "week2": run_week2_eval(domain, s, chunkings=["heading_cap"]),
        "week3": run_week3_eval(domain, adv, s, use_mock=True, strategies=["decomposition"]),
        "week4": run_week4_eval(s, use_mock=True),
    }
    report = generate_week_wise_report(res, n_domain=2, n_adv=1, model_name="mock")
    assert "## 1. Week 1:" in report
    assert "## 2. Week 2:" in report
    assert "## 3. Week 3:" in report
    assert "## 4. Week 4:" in report
    assert "ROUGE-LCS F1 (Generation)" in report

