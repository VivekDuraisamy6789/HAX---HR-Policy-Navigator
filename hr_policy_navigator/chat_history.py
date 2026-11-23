"""Persistent per-user chat history (Sprint 1 — Amit-Enhanced).

Stores full chat sessions as JSONL files under state/chat_history/<user_id>/.
Each file represents one session (named by session_id + date).
Provides load / save / list_sessions helpers consumed by streamlit_app.py.

Retention: 7-day rolling window to keep storage bounded and maintain privacy.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class ChatHistory:
    """Thread-safe persistent chat history store."""

    RETENTION_DAYS = 7

    def __init__(self, state_dir: str | Path = "state"):
        self._base = Path(state_dir) / "chat_history"
        self._base.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _user_dir(self, user_id: str) -> Path:
        d = self._base / user_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _titles_path(self, user_id: str) -> Path:
        return self._user_dir(user_id) / "titles.json"

    def get_titles(self, user_id: str) -> dict[str, str]:
        """Return mapping of session_id -> custom title for the user."""
        p = self._titles_path(user_id)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def rename_session(self, user_id: str, session_id: str, title: str) -> None:
        """Assign or update a custom title for a chat session."""
        title = title.strip()
        if not title:
            return
        with self._lock:
            titles = self.get_titles(user_id)
            titles[session_id] = title
            p = self._titles_path(user_id)
            p.write_text(json.dumps(titles, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_title(self, user_id: str, session_id: str, default: str = "") -> str:
        """Get custom title for session or return default."""
        titles = self.get_titles(user_id)
        return titles.get(session_id, default)

    def _session_path(self, user_id: str, session_id: str) -> Path:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filename = f"{session_id}_{date_str}.jsonl"
        return self._user_dir(user_id) / filename

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def append(self, user_id: str, session_id: str, query: str, response: dict[str, Any]) -> None:
        """Append a (query, response) turn to the session file."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "query": query,
            "response": {
                "status": response.get("status"),
                "answer": response.get("answer", ""),
                "citations": response.get("citations", []),
                "latency_ms": response.get("latency_ms", 0),
                "tool_calls": response.get("tool_calls", 0),
                "tokens": response.get("tokens", 0),
                "cost_usd": response.get("cost_usd", 0.0),
                "trace_id": response.get("trace_id", ""),
                "refusal_category": response.get("refusal_category"),
                "cached": response.get("cached", False),
            },
        }
        with self._lock:
            path = self._session_path(user_id, session_id)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def list_sessions(self, user_id: str, limit: int = 3) -> list[dict[str, Any]]:
        """Return metadata for up to `limit` newest sessions within retention window."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.RETENTION_DAYS)
        sessions = []
        user_dir = self._user_dir(user_id)
        titles = self.get_titles(user_id)
        for p in sorted(user_dir.glob("*.jsonl"), reverse=True):
            try:
                lines = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            except (json.JSONDecodeError, OSError):
                continue
            if not lines:
                continue
            first_ts = datetime.fromisoformat(lines[0]["ts"])
            if first_ts < cutoff:
                continue
            sid = lines[0]["session_id"]
            first_q = lines[0]["query"][:60] + ("…" if len(lines[0]["query"]) > 60 else "")
            title = titles.get(sid) or first_q
            sessions.append({
                "session_id": sid,
                "title": title,
                "file": str(p),
                "date": first_ts.strftime("%b %d, %Y"),
                "time": first_ts.strftime("%H:%M"),
                "turn_count": len(lines),
                "first_query": first_q,
                "lines": lines,
            })
            if len(sessions) >= limit:
                break
        return sessions

    def load_session(self, file_path: str) -> list[tuple[str, dict[str, Any]]]:
        """Load a session file; return list of (query, response) tuples."""
        path = Path(file_path)
        if not path.exists():
            return []
        try:
            lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        except (json.JSONDecodeError, OSError):
            return []
        return [(r["query"], r["response"]) for r in lines]

    def purge_old(self, user_id: str) -> int:
        """Delete files older than RETENTION_DAYS. Returns count deleted."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.RETENTION_DAYS)
        deleted = 0
        for p in self._user_dir(user_id).glob("*.jsonl"):
            try:
                lines = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
                if lines:
                    ts = datetime.fromisoformat(lines[0]["ts"])
                    if ts < cutoff:
                        p.unlink()
                        deleted += 1
            except (json.JSONDecodeError, OSError):
                pass
        return deleted
