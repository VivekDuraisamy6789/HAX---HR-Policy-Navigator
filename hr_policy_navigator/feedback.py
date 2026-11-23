"""RLHF feedback store for HR Policy Navigator (Sprint 3 — Amit-Enhanced).

Provides:
  FeedbackStore — thread-safe JSONL writer + analytics reader
    .save(trace_id, query, answer, rating, correction, user_role, user_id)
    .all()           -> List[dict]      all feedback records
    .analytics()     -> dict            aggregated stats for the Feedback tab

Rating scale: +1 (thumbs up), -1 (thumbs down)
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FeedbackStore:
    """Thread-safe append-only JSONL feedback log with analytics helpers."""

    def __init__(self, state_dir: str | Path = "state"):
        self._path = Path(state_dir) / "feedback_log.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def save(
        self,
        trace_id: str,
        query: str,
        answer: str,
        rating: int,              # +1 or -1
        correction: str = "",     # optional free-text from user
        user_role: str = "",
        user_id: str = "",
        strategy: str = "",
    ) -> None:
        """Append a feedback record to the JSONL log."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "user_id": user_id,
            "user_role": user_role,
            "query": query[:300],
            "answer_snippet": answer[:200],
            "rating": rating,          # +1 or -1
            "correction": correction[:500] if correction else "",
            "strategy": strategy,
        }
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def all(self, user_id: str = "") -> list[dict[str, Any]]:
        """Return feedback records (optionally filtered by user_id), newest first."""
        if not self._path.exists():
            return []
        rows = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if user_id and rec.get("user_id") != user_id:
                    continue
                rows.append(rec)
            except json.JSONDecodeError:
                continue
        return list(reversed(rows))

    def analytics(self, user_id: str = "") -> dict[str, Any]:
        """
        Compute analytics summary for the Feedback Analytics tab.
        If user_id is provided, only that user's feedback is analyzed.
        """
        rows = self.all(user_id=user_id)
        if not rows:
            return {
                "total": 0, "positive": 0, "negative": 0, "pct_positive": 0.0,
                "by_role": {}, "by_strategy": {}, "low_rated": [], "high_rated": [], "corrections": [],
            }

        total = len(rows)
        positive = sum(1 for r in rows if r.get("rating", 0) > 0)
        negative = total - positive

        # By-role breakdown
        by_role: dict[str, dict] = {}
        for r in rows:
            role = r.get("user_role") or "unknown"
            bucket = by_role.setdefault(role, {"pos": 0, "neg": 0, "total": 0})
            bucket["total"] += 1
            if r.get("rating", 0) > 0:
                bucket["pos"] += 1
            else:
                bucket["neg"] += 1

        # By-strategy breakdown
        by_strategy: dict[str, dict] = {}
        for r in rows:
            strat = r.get("strategy") or "unknown"
            bucket = by_strategy.setdefault(strat, {"pos": 0, "neg": 0, "total": 0})
            bucket["total"] += 1
            if r.get("rating", 0) > 0:
                bucket["pos"] += 1
            else:
                bucket["neg"] += 1

        # Cluster negative feedback by query text (simple prefix match)
        neg_queries: dict[str, dict] = {}
        pos_queries: dict[str, dict] = {}
        for r in rows:
            q_key = r.get("query", "")[:80]
            if r.get("rating", 0) < 0:
                entry = neg_queries.setdefault(q_key, {"query": q_key, "answer_snippet": r.get("answer_snippet", ""), "count": 0})
                entry["count"] += 1
            else:
                entry = pos_queries.setdefault(q_key, {"query": q_key, "answer_snippet": r.get("answer_snippet", ""), "count": 0})
                entry["count"] += 1

        low_rated = sorted(neg_queries.values(), key=lambda x: -x["count"])[:5]
        high_rated = sorted(pos_queries.values(), key=lambda x: -x["count"])[:5]

        # Recent corrections (non-empty)
        corrections = [
            {"ts": r.get("ts", ""), "query": r.get("query", "")[:80], "correction": r.get("correction", "")}
            for r in rows if r.get("correction", "").strip()
        ][:10]

        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "pct_positive": round(100 * positive / total, 1) if total else 0.0,
            "by_role": by_role,
            "by_strategy": by_strategy,
            "low_rated": low_rated,
            "high_rated": high_rated,
            "corrections": corrections,
        }
