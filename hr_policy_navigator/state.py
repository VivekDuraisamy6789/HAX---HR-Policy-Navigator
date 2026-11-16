"""Persistent per-session state (conversation memory, last evidence). JSON file per session."""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, List


class SessionStore:
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
