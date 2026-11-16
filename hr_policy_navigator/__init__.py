"""HR Policy Navigator: hybrid RAG + planner/executor/critic agent for a hosted <=8B model."""
from .agent import Copilot
from .retrieval import PolicyCorpus

# Backwards-compatible alias for the old class name
PlannerExecutorCriticAgent = Copilot

__all__ = ["Copilot", "PolicyCorpus", "PlannerExecutorCriticAgent"]
