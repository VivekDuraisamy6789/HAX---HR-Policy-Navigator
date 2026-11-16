"""Print aggregate stats from traces/traces.jsonl (the same numbers the Streamlit Traces tab shows)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hr_policy_navigator.tracing import load_traces, summarize
print(json.dumps(summarize(load_traces(sys.argv[1] if len(sys.argv) > 1 else "traces")), indent=1))
