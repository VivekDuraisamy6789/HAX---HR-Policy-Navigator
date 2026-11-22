"""Authentication module for HR Policy Navigator (Sprint 1 — Amit-Enhanced).

Provides a simple credential table with four demo personas covering all roles
used in the capstone presentation. No external dependency required.

Roles:
  hr_ops       — HR Operations: full access including HITL queue management
  manager      — People Manager: team-scoped view of HITL queue, manager-layer answers
  associate    — Employee (tenured 3+ yrs): concise direct answers
  associate    — Employee (new <1 yr): step-by-step onboarding-style answers

Tenure levels: "new" (0-1 yr), "mid" (1-3 yr), "senior" (3+ yr)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    name: str
    role: str          # "hr_ops" | "manager" | "associate"
    tenure: str        # "new" | "mid" | "senior"
    team_id: str       # used to scope manager's HITL view
    display_role: str  # human-readable label for UI


# ---------------------------------------------------------------------------
# Demo user table — four personas for capstone presentation
# ---------------------------------------------------------------------------
_USER_DB: dict[str, dict] = {
    "hr_admin": {
        "password": "hr123",
        "profile": UserProfile(
            user_id="u001",
            name="Priya Sharma",
            role="hr_ops",
            tenure="senior",
            team_id="hr_ops",
            display_role="HR Operations",
        ),
    },
    "mgr_alice": {
        "password": "mgr123",
        "profile": UserProfile(
            user_id="u002",
            name="Alice Chen",
            role="manager",
            tenure="senior",
            team_id="eng_team_a",
            display_role="People Manager",
        ),
    },
    "emp_bob": {
        "password": "emp123",
        "profile": UserProfile(
            user_id="u003",
            name="Bob Patel",
            role="associate",
            tenure="new",
            team_id="eng_team_a",
            display_role="Associate (New Hire)",
        ),
    },
    "emp_carol": {
        "password": "emp456",
        "profile": UserProfile(
            user_id="u004",
            name="Carol Nair",
            role="associate",
            tenure="senior",
            team_id="eng_team_b",
            display_role="Associate (Tenured)",
        ),
    },
}


def authenticate(username: str, password: str) -> Optional[UserProfile]:
    """Return UserProfile if credentials match, else None."""
    entry = _USER_DB.get(username.strip().lower())
    if entry and entry["password"] == password:
        return entry["profile"]
    return None


def demo_users() -> list[dict]:
    """Return sanitized list of demo users (no passwords) for the login hint panel."""
    return [
        {
            "username": u,
            "name": v["profile"].name,
            "role": v["profile"].display_role,
            "tenure": v["profile"].tenure,
            "hint_password": v["password"],
        }
        for u, v in _USER_DB.items()
    ]
