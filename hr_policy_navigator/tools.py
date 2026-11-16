"""Tool layer: JSON-schema-validated arguments, uniform ToolResult, no exception escapes.

Tools
  search_policy      hybrid retrieval over the corpus
  read_section       fetch more text of a document section (expand an answer)
  validate_citations check that cited evidence ids exist and support the claims
  escalate_to_human  human-in-the-loop handoff (writes a ticket; takes NO HR action itself)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import jsonschema


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str = ""
    ms: float = 0.0


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    fn: Callable[..., Any]


class ToolRegistry:
    def __init__(self, tools: List[Tool]):
        self.tools = {t.name: t for t in tools}

    def spec(self) -> str:
        return "\n".join(f"- {t.name}: {t.description} args={json.dumps(t.schema['properties'])}" for t in self.tools.values())

    def call(self, name: str, args: Dict[str, Any]) -> ToolResult:
        t0 = time.perf_counter()
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(False, error=f"unknown tool '{name}'", ms=0)
        try:
            jsonschema.validate(args, tool.schema)
            return ToolResult(True, tool.fn(**args), ms=round((time.perf_counter() - t0) * 1000, 1))
        except jsonschema.ValidationError as e:
            return ToolResult(False, error=f"bad arguments: {e.message}", ms=round((time.perf_counter() - t0) * 1000, 1))
        except Exception as e:  # noqa: BLE001  - tool failures must not crash the agent
            return ToolResult(False, error=f"{type(e).__name__}: {e}"[:300], ms=round((time.perf_counter() - t0) * 1000, 1))


def build_tools(corpus, handoff_path: Path, claim_checker: Callable) -> ToolRegistry:
    def search_policy(query: str, top_k: int = 6):
        hits = corpus.search(query, top_k=top_k)
        return [{"path": h.file_path, "section": h.chunk.section, "chunk_id": h.chunk.chunk_id,
                 "score": round(h.score, 4), "text": h.chunk.text, "stages": h.sources} for h in hits]

    def read_section(path: str, section: str = ""):
        txt = corpus.read_section(path, section)
        if not txt:
            raise FileNotFoundError(f"no content for {path} / '{section}'")
        return txt

    def validate_citations(claims: list, evidence: dict):
        return claim_checker(claims, evidence)

    def escalate_to_human(topic: str, reason: str):
        handoff_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.time(), "topic": topic[:300], "reason": reason[:300], "status": "pending_human_review"}
        with open(handoff_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return {"ticket": "pending_human_review"}

    S = lambda props, req: {"type": "object", "properties": props, "required": req, "additionalProperties": False}
    return ToolRegistry([
        Tool("search_policy", "Search the HR policy corpus; returns ranked passages.",
             S({"query": {"type": "string", "minLength": 2, "maxLength": 400}, "top_k": {"type": "integer", "minimum": 1, "maximum": 12}}, ["query"]),
             search_policy),
        Tool("read_section", "Read more text from a policy document section.",
             S({"path": {"type": "string"}, "section": {"type": "string"}}, ["path"]), read_section),
        Tool("validate_citations", "Check each claim is supported by its cited evidence.",
             S({"claims": {"type": "array"}, "evidence": {"type": "object"}}, ["claims", "evidence"]), validate_citations),
        Tool("escalate_to_human", "Record a request for human review. Never performs the action.",
             S({"topic": {"type": "string"}, "reason": {"type": "string"}}, ["topic", "reason"]), escalate_to_human),
    ])
