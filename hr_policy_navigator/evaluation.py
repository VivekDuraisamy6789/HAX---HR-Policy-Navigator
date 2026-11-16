"""Kept for backwards compatibility. The real harnesses live in scripts/:
run_retrieval_eval.py (recall@k), run_e2e_eval.py (accuracy, taxonomy, cost), run_injection_suite.py."""
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]


def load_evaluation_set(path: str | Path = ROOT / "evaluation_set.json") -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
