"""Tracing: JSONL spans always on; Langfuse mirrored when keys are set.

One JSON line per *trace* (= one user query) with nested spans: planner, each tool call,
each LLM call (latency, tokens), critic. `scripts/trace_summary.py` and the Streamlit
'Traces' tab read these files, so the dashboard shows real executions.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional


class Trace:
    def __init__(self, tracer: "Tracer", name: str, meta: Dict[str, Any]):
        self.tracer, self.id = tracer, uuid.uuid4().hex[:12]
        self.data: Dict[str, Any] = {"trace_id": self.id, "name": name, "ts": time.time(), "meta": meta,
                                     "spans": [], "counters": {"llm_calls": 0, "tool_calls": 0, "prompt_tokens": 0,
                                                              "completion_tokens": 0, "tool_errors": 0, "json_repairs": 0}}
        self._t0 = time.perf_counter()
        self._lf = tracer._lf_start(self)

    @contextmanager
    def span(self, kind: str, name: str, **attrs):
        sp = {"kind": kind, "name": name, "attrs": dict(attrs), "start_ms": round((time.perf_counter() - self._t0) * 1000, 1)}
        t = time.perf_counter()
        try:
            yield sp
            sp["status"] = sp.get("status", "ok")
        except Exception as e:  # keep the trace even when the span fails
            sp["status"], sp["error"] = "error", f"{type(e).__name__}: {e}"[:300]
            raise
        finally:
            sp["duration_ms"] = round((time.perf_counter() - t) * 1000, 1)
            self.data["spans"].append(sp)

    def count(self, key: str, n: int = 1):
        self.data["counters"][key] = self.data["counters"].get(key, 0) + n

    def end(self, **result):
        self.data["duration_ms"] = round((time.perf_counter() - self._t0) * 1000, 1)
        self.data["result"] = result
        self.tracer._write(self)
        return self.data


class Tracer:
    def __init__(self, trace_dir: str | Path = "traces", langfuse: Optional[Dict[str, str]] = None, enabled: bool = True):
        self.dir, self.enabled, self._lock = Path(trace_dir), enabled, threading.Lock()
        self._lf_client = None
        if langfuse and langfuse.get("public_key") and langfuse.get("secret_key"):
            try:
                from langfuse import Langfuse  # optional dependency
                self._lf_client = Langfuse(public_key=langfuse["public_key"], secret_key=langfuse["secret_key"],
                                           host=langfuse.get("host") or None)
            except Exception:
                self._lf_client = None
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def start(self, name: str = "query", **meta) -> Trace:
        return Trace(self, name, meta)

    def _lf_start(self, trace: Trace):
        return None  # spans are mirrored at end (keeps hot path free of network calls)

    def _write(self, trace: Trace):
        if self.enabled:
            line = json.dumps(trace.data, ensure_ascii=False, default=str)
            with self._lock, open(self.dir / "traces.jsonl", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        if self._lf_client is not None:
            try:
                d = trace.data
                t = self._lf_client.trace(id=d["trace_id"], name=d["name"], input=d["meta"].get("query"),
                                          output=d.get("result", {}).get("answer"), metadata=d["counters"])
                for sp in d["spans"]:
                    t.span(name=f"{sp['kind']}:{sp['name']}", metadata=sp["attrs"], level="ERROR" if sp.get("status") == "error" else "DEFAULT")
                self._lf_client.flush()
            except Exception:
                pass


def load_traces(trace_dir: str | Path = "traces"):
    p = Path(trace_dir) / "traces.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def summarize(traces):
    import statistics as st
    if not traces:
        return {}
    lat = sorted(t["duration_ms"] for t in traces)
    pick = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))]
    c = lambda k: [t["counters"].get(k, 0) for t in traces]
    return {"queries": len(traces), "p50_ms": pick(0.5), "p95_ms": pick(0.95),
            "avg_tool_calls": round(st.mean(c("tool_calls")), 2), "avg_llm_calls": round(st.mean(c("llm_calls")), 2),
            "avg_prompt_tokens": round(st.mean(c("prompt_tokens"))), "avg_completion_tokens": round(st.mean(c("completion_tokens"))),
            "tool_errors": sum(c("tool_errors")), "json_repairs": sum(c("json_repairs")),
            "statuses": {s: sum(1 for t in traces if t.get("result", {}).get("status") == s)
                         for s in {t.get("result", {}).get("status") for t in traces}}}
