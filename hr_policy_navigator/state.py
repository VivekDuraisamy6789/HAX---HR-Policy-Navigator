"""Persistent per-session state + handoff status management (Sprint 2 — Amit-Enhanced).

Changes over baseline:
  - SessionStore unchanged (backward compat)
  - HandoffStore: thread-safe read/write of handoffs.jsonl with status update support
    (pending -> acknowledged -> resolved / escalated)
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class SessionStore:
    """Persistent per-session conversation memory (rolling 6-turn window)."""

    def __init__(self, state_dir: str | Path = "state", max_turns: int = 6):
        self.dir, self.max_turns, self._lock = Path(state_dir) / "sessions", max_turns, threading.Lock()
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, sid: str) -> Path:
        return self.dir / (re.sub(r"[^A-Za-z0-9_-]", "_", sid)[:64] + ".json")

    def load(self, sid: str) -> Dict[str, Any]:
        p = self._path(sid)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {"session_id": sid, "turns": [], "last_sources": []}

    def append(self, sid: str, turn: Dict[str, Any]) -> None:
        with self._lock:
            s = self.load(sid)
            s["turns"] = (s["turns"] + [turn])[-self.max_turns:]
            s["last_sources"] = turn.get("sources", s.get("last_sources", []))
            self._path(sid).write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def history_text(s: Dict[str, Any], n: int = 3) -> str:
        return "\n".join(f"User: {t['query']}\nAssistant: {t['answer'][:300]}" for t in s["turns"][-n:])


# ---------------------------------------------------------------------------
# Sprint 2 — HandoffStore: styled HITL queue with status state machine
# ---------------------------------------------------------------------------

VALID_TRANSITIONS: Dict[str, List[str]] = {
    "pending":      ["acknowledged", "resolved", "escalated"],
    "acknowledged": ["resolved", "escalated"],
    "resolved":     [],   # terminal
    "escalated":    ["resolved"],
}

STATUS_LABELS = {
    "pending":      "🔴 Pending",
    "acknowledged": "🟡 Acknowledged",
    "resolved":     "🟢 Resolved",
    "escalated":    "🔺 Escalated",
}


class HandoffStore:
    """Thread-safe reader/writer for state/handoffs.jsonl with status transitions."""

    def __init__(self, state_dir: str | Path = "state"):
        self._path = Path(state_dir) / "handoffs.jsonl"
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def all(self, role: Optional[str] = None, team_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load all handoffs, newest first. Managers see only their team's tickets."""
        rows = self._read_all()
        if role == "manager" and team_id:
            rows = [r for r in rows if r.get("team_id") == team_id]
        return list(reversed(rows))

    def update_status(self, ticket_id: str, new_status: str, actor: str = "") -> bool:
        """
        Transition ticket_id to new_status if the transition is valid.
        Returns True on success, False if the ticket was not found or transition invalid.
        """
        with self._lock:
            rows = self._read_all()
            updated = False
            for row in rows:
                if row.get("ticket_id") == ticket_id:
                    current = row.get("ui_status", "pending")
                    if new_status in VALID_TRANSITIONS.get(current, []):
                        row["ui_status"] = new_status
                        row["last_updated"] = datetime.now(timezone.utc).isoformat()
                        row["updated_by"] = actor
                        updated = True
                    break
            if updated:
                self._write_all(rows)
        return updated

    def add_note(self, ticket_id: str, note: str, actor: str = "") -> bool:
        """Append a free-text resolution note to a handoff ticket."""
        with self._lock:
            rows = self._read_all()
            updated = False
            for row in rows:
                if row.get("ticket_id") == ticket_id:
                    row.setdefault("notes", []).append({
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "actor": actor,
                        "text": note[:500],
                    })
                    updated = True
                    break
            if updated:
                self._write_all(rows)
        return updated

    # ------------------------------------------------------------------
    def _read_all(self) -> List[Dict[str, Any]]:
        if not self._path.exists():
            return []
        rows = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                # Back-fill fields added in Sprint 2
                row.setdefault("ui_status", "pending")
                row.setdefault("ticket_id", row.get("trace_id", f"t_{len(rows):04d}"))
                row.setdefault("timestamp", row.get("ts", ""))
                rows.append(row)
            except json.JSONDecodeError:
                continue
        return rows

    def _write_all(self, rows: List[Dict[str, Any]]) -> None:
        self._path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
