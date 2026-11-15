import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "evaluation_set.json"


def load_evaluation_set(path: str | Path = EVAL_PATH) -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    completed = sum(1 for item in results if item.get("status") == "success")
    refusal_count = sum(1 for item in results if item.get("status") == "refused")
    no_results = sum(1 for item in results if item.get("status") == "no_results")
    metrics = {
        "total_queries": total,
        "completed": completed,
        "task_completion_rate": round((completed / total) * 100, 2) if total else 0.0,
        "refused": refusal_count,
        "no_results": no_results,
    }
    return metrics
