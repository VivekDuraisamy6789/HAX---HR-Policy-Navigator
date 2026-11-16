import json, re, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hr_policy_navigator.agent import Copilot
from hr_policy_navigator.config import Settings
from hr_policy_navigator.llm import LLMClient
from hr_policy_navigator.retrieval import PolicyCorpus
from hr_policy_navigator.state import SessionStore
from hr_policy_navigator.tracing import Tracer

ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    """Scripted stand-in for the hosted 8B model. `answer_fn(sources_dict, question) -> dict`."""

    def __init__(self, plan=None, answer_fn=None, raw_overrides=None):
        self.calls, self.plan, self.raw = [], plan, list(raw_overrides or [])
        self.answer_fn = answer_fn or self.default_answer

    @staticmethod
    def parse_sources(user):
        return {m.group(1): m.group(2) for m in re.finditer(r"\[(S\d+)\] \(.*?\)\n(.*?)(?=\n\n\[S\d+\]|\n\nQUESTION|\Z)", user, re.S)}

    @staticmethod
    def default_answer(sources, question):
        if not sources:
            return {"answerable": False, "answer": "The documents do not say.", "claims": [], "confidence": "low"}
        sid, text = next(iter(sources.items()))
        sent = re.split(r"(?<=[.!?])\s", re.sub(r"\s+", " ", text).strip())[0][:300]
        return {"answerable": True, "answer": sent, "claims": [{"text": sent, "sources": [sid]}], "confidence": "medium"}

    def __call__(self, **kw):
        msgs = kw["messages"]
        self.calls.append(kw)
        if self.raw:
            txt = self.raw.pop(0)
            return {"text": txt, "prompt_tokens": 100, "completion_tokens": 50}
        system = msgs[0]["content"]
        user = next((m["content"] for m in reversed(msgs) if m["role"] == "user" and ("SOURCES:" in m["content"] or "Latest user message" in m["content"])), msgs[-1]["content"])
        if "plan how to answer" in system:
            q = re.search(r"Latest user message: (.*)", user).group(1)
            out = self.plan or {"intent": "policy_lookup", "rewritten_query": q, "sub_queries": []}
        else:
            out = self.answer_fn(self.parse_sources(user), user)
        return {"text": json.dumps(out), "prompt_tokens": 400, "completion_tokens": 120}


@pytest.fixture(scope="session")
def corpus():
    return PolicyCorpus(ROOT / "sourcegraph_files", use_dense=True)


def make_agent(corpus, tmp_path, model=None, **settings):
    s = Settings(trace_dir=tmp_path / "traces", state_dir=tmp_path / "state", **settings)
    fm = model or FakeModel()
    llm = LLMClient("", "", "fake", transport=fm, json_retries=s.json_retries)
    agent = Copilot(corpus, s, llm, Tracer(s.trace_dir), SessionStore(s.state_dir))
    return agent, fm


@pytest.fixture
def agent_factory(corpus, tmp_path):
    return lambda model=None, **kw: make_agent(corpus, tmp_path, model, **kw)
