"""LLM access layer for a hosted <=8B OpenAI-compatible endpoint.

Everything that makes a small model usable lives here:
  * constrained decoding when the host supports it (json_schema / vLLM guided_json /
    json_object), auto-detected and auto-downgraded on 4xx errors
  * tolerant JSON extraction (code fences, prose around the object, trailing commas)
  * jsonschema validation + *repair loop* that feeds the validation error back
  * per-query token & call budgets (hard stop -> BudgetExceeded)
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import jsonschema


class BudgetExceeded(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


class JSONFailure(LLMError):
    def __init__(self, msg, raw=""):
        super().__init__(msg)
        self.raw = raw


@dataclass
class Budget:
    max_tokens: int
    max_calls: int
    tokens: int = 0
    calls: int = 0

    def check(self):
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"LLM call cap {self.max_calls} reached")
        if self.tokens >= self.max_tokens:
            raise BudgetExceeded(f"token cap {self.max_tokens} reached")


_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_think(text: str) -> str:
    """Remove Qwen3-style reasoning blocks; a never-closed <think> means truncated reasoning, so drop it."""
    text = _THINK.sub("", text)
    i = text.lower().find("<think>")
    if i >= 0:
        text = text[:i]
    return text.strip()


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def extract_json(raw: str) -> Any:
    """Parse the first JSON object in `raw`, tolerating fences, chatter and trailing commas."""
    text = strip_think(raw)
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    for candidate in (text, _first_balanced(text)):
        if not candidate:
            continue
        for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                pass
    raise JSONFailure("no parseable JSON object found", raw)


def _first_balanced(s: str) -> Optional[str]:
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, max_tokens: int = 700,
                 constrained_mode: str = "auto", json_retries: int = 2, tracer_trace=None, no_think: bool = False,
                 transport: Optional[Callable[..., Dict[str, Any]]] = None):
        self.model, self.max_tokens, self.json_retries = model, max_tokens, json_retries
        self.no_think = no_think
        self.mode = "json_schema" if constrained_mode == "auto" else constrained_mode
        self.auto_mode = constrained_mode == "auto"
        self.trace = tracer_trace
        self.budget: Optional[Budget] = None
        self._transport = transport
        self._client = None
        if transport is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=60, max_retries=0)

    # ---- per-query wiring
    def bind(self, trace, budget: Budget):
        c = LLMClient.__new__(LLMClient)
        c.__dict__.update(self.__dict__)
        c.trace, c.budget = trace, budget
        return c

    # ---- raw completion
    def _create(self, messages, temperature, max_tokens, schema):
        if self.no_think and messages and messages[0]["role"] == "system" and "/no_think" not in messages[0]["content"]:
            messages = [{"role": "system", "content": messages[0]["content"] + "\n/no_think"}, *messages[1:]]  # Qwen3 soft switch
        kw: Dict[str, Any] = dict(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if schema is not None:
            if self.mode == "json_schema":
                kw["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema, "strict": True}}
            elif self.mode == "guided_json":
                kw["extra_body"] = {"guided_json": schema}
            elif self.mode == "json_object":
                kw["response_format"] = {"type": "json_object"}
        if self._transport is not None:
            return self._transport(**kw)
        r = self._client.chat.completions.create(**kw)
        return {"text": r.choices[0].message.content or "",
                "prompt_tokens": getattr(r.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(r.usage, "completion_tokens", 0) or 0}

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.1, max_tokens: Optional[int] = None,
             schema: Optional[dict] = None, stage: str = "llm") -> str:
        if self.budget:
            self.budget.check()
        t0 = time.perf_counter()
        last_err = None
        for attempt in range(5):  # transient errors: backoff; capability errors: downgrade mode
            try:
                r = self._create(messages, temperature, max_tokens or self.max_tokens, schema)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e).lower()
                if schema is not None and any(k in msg for k in ("response_format", "guided", "json_schema", "unsupported", "400", "422")) and self.mode != "none":
                    self.mode = {"json_schema": "guided_json", "guided_json": "json_object"}.get(self.mode, "none")
                    continue
                time.sleep(0.4 * (2 ** attempt))
        else:
            raise LLMError(f"model endpoint failed: {last_err}")
        pt = r.get("prompt_tokens") or sum(len(m["content"]) for m in messages) // 4
        ct = r.get("completion_tokens") or len(r["text"]) // 4
        if self.budget:
            self.budget.calls += 1
            self.budget.tokens += pt + ct
        if self.trace:
            self.trace.count("llm_calls"); self.trace.count("prompt_tokens", pt); self.trace.count("completion_tokens", ct)
            self.trace.data["spans"].append({"kind": "llm", "name": stage, "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
                                             "attrs": {"prompt_tokens": pt, "completion_tokens": ct, "mode": self.mode if schema else "free"},
                                             "status": "ok"})
        return strip_think(r["text"])

    # ---- validated JSON call with repair loop
    def json_call(self, system: str, user: str, schema: dict, stage: str = "json", temperature: float = 0.1,
                  max_tokens: Optional[int] = None) -> Dict[str, Any]:
        messages = [{"role": "system", "content": system + "\nRespond with a single JSON object only. No prose."},
                    {"role": "user", "content": user}]
        err: Optional[str] = None
        raw = ""
        for attempt in range(self.json_retries + 1):
            raw = self.chat(messages, temperature=temperature if attempt == 0 else 0.0, max_tokens=max_tokens,
                            schema=schema, stage=stage if attempt == 0 else f"{stage}:repair{attempt}")
            try:
                obj = extract_json(raw)
                jsonschema.validate(obj, schema)
                return obj
            except (JSONFailure, jsonschema.ValidationError) as e:
                err = e.message if isinstance(e, jsonschema.ValidationError) else str(e)
                if self.trace:
                    self.trace.count("json_repairs")
                messages = messages + [{"role": "assistant", "content": raw[:1500]},
                                       {"role": "user", "content": f"That output was invalid: {err[:300]}. "
                                        "Return ONLY the corrected JSON object matching the schema."}]
        raise JSONFailure(f"JSON invalid after {self.json_retries} repairs: {err}", raw)
