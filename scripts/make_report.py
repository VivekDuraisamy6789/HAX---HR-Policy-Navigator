"""Turn eval_runs/e2e_*.json into the two Week-1/Week-3 documents (no numbers are invented):

    python scripts/make_report.py                       # real runs only
    python scripts/make_report.py --allow-mock          # plumbing check; output is stamped NOT REAL

Writes docs/strategy_comparison.md and docs/failure_characterization.md.
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["retrieval_miss", "reasoning_error", "tool_call_error", "hallucination", "refusal_error"]
BLURB = {
    "retrieval_miss": "The expected source never reached the model's context. Fix retrieval, not prompts.",
    "reasoning_error": "Right evidence in context, wrong/abstained/incomplete answer. The generator failed.",
    "tool_call_error": "Endpoint/JSON/tool failure or exhausted budget before a valid answer.",
    "hallucination": "Claims not supported by cited text, or an unanswerable question was answered.",
    "refusal_error": "Refused a legitimate question, or answered one that must be refused.",
}


def cut(s, n=260):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="eval_runs"); ap.add_argument("--allow-mock", action="store_true")
    ap.add_argument("--examples", type=int, default=3)
    a = ap.parse_args()
    files = sorted(Path(a.runs).glob("e2e_*.json"))
    mock = [f for f in files if "MOCK" in f.name]
    files = [f for f in files if a.allow_mock or "MOCK" not in f.name]
    if not files:
        sys.exit("No real eval runs found. Run scripts/run_e2e_eval.py against your endpoint first"
                 + (" (only MOCK outputs exist; use --allow-mock to test the generator)." if mock else "."))
    runs = {f.stem.replace("e2e_", "").replace("_MOCK", ""): json.loads(f.read_text()) for f in files}
    stamp = "> **NOT REAL: generated from a scripted mock model to test this script. Do not cite.**\n\n" if any("MOCK" in f.name for f in files) else ""
    model = next(iter(runs.values()))["summary"]["model"]

    # ---------------- strategy comparison
    L = [f"# Prompting-strategy comparison\n\n{stamp}Model: `{model}` · n = {next(iter(runs.values()))['summary']['n']} queries per strategy "
         "(evaluation_set.json + evaluation_set_adversarial.json). Correctness definition: see docstring of `scripts/run_e2e_eval.py` "
         "(lexical gold recall is a proxy; spot-check by hand).\n"]
    cols = ["accuracy", "task_completion_rate", "retrieval_hit_in_context", "avg_tool_calls", "avg_llm_calls", "avg_tokens", "avg_cost_usd", "p50_latency_ms", "p95_latency_ms", "json_repairs"]
    L.append("| strategy | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    for k, r in runs.items():
        L.append(f"| {k} | " + " | ".join(str(r["summary"][c]) for c in cols) + " |")
    types = sorted({t for r in runs.values() for t in r["summary"]["accuracy_by_type"]})
    L.append("\n## Accuracy by query type (correct/total)\n\n| type | " + " | ".join(runs) + " |\n|---|" + "---|" * len(runs))
    for t in types:
        L.append(f"| {t} | " + " | ".join(r["summary"]["accuracy_by_type"].get(t, "-") for r in runs.values()) + " |")
    L.append("\n## Failure taxonomy (count of incorrect queries)\n\n| strategy | " + " | ".join(CLASSES) + " |\n|---|" + "---|" * len(CLASSES))
    for k, r in runs.items():
        c = r["summary"]["failure_taxonomy"]
        L.append(f"| {k} | " + " | ".join(str(c.get(x, 0)) for x in CLASSES) + " |")
    best = max(runs, key=lambda k: runs[k]["summary"]["accuracy"])
    L.append(f"\nHighest accuracy: **{best}**. Differences of a few queries are within noise at this n; "
             "re-run with `temperature` fixed and report the spread before claiming a winner.\n\n"
             "## Interpretation (team to write after reading the tables)\n- Did few-shot structure beat baseline, and by how much?\n"
             "- Did self-consistency's extra cost buy accuracy, or only latency?\n- Did decomposition help multi-part questions or add errors?\n")
    Path("docs/strategy_comparison.md").write_text("\n".join(L), encoding="utf-8")

    # ---------------- failure characterisation
    F = [f"# How the small model breaks\n\n{stamp}Measured on `{model}`. Every example is a real row from `eval_runs/` with a `trace_id` for `traces/traces.jsonl`.\n"]
    F.append("## Break-mode rates by strategy\n\n| strategy | queries with JSON repairs | answerable but abstained | unanswerable but answered | legit refused | answered but cited wrong doc |\n|---|---|---|---|---|---|")
    for k, r in runs.items():
        rows = r["rows"]; n = len(rows)
        ans = [x for x in rows if x["expected_sources"] and x["type"] in ("answerable", "pii", "false_refusal_probe")]
        F.append(f"| {k} | {sum(x['json_repairs']>0 for x in rows)}/{n} | {sum(x['status']=='abstained' for x in ans)}/{len(ans)} | "
                 f"{sum(x['type'] in ('unanswerable','out_of_scope') and x['status']=='success' for x in rows)}/{sum(x['type'] in ('unanswerable','out_of_scope') for x in rows)} | "
                 f"{sum(x['type'] in ('answerable','pii','false_refusal_probe') and x['status']=='refused' for x in rows)}/{len(ans)} | "
                 f"{sum(x['status']=='success' and bool(x['expected_sources']) and not set(x['cited']) & set(x['expected_sources']) for x in rows)}/{len(ans)} |")
    F.append("")
    for cls in CLASSES:
        seen, ex = set(), []
        for k, r in runs.items():
            for x in r["rows"]:
                if x["failure"] == cls and x["id"] not in seen and len(ex) < a.examples:
                    seen.add(x["id"]); ex.append((k, x))
        tot = sum(1 for r in runs.values() for x in r["rows"] if x["failure"] == cls)
        F.append(f"## {cls}  ({tot} occurrences across strategies)\n{BLURB[cls]}\n")
        if not ex:
            F.append("_No occurrences._\n"); continue
        for k, x in ex:
            F.append(f"- **[{x['id']} · {x['type']} · {k}]** {cut(x['question'], 160)}\n  - status `{x['status']}` · expected sources `{x['expected_sources'] or '-'}` · retrieved `{x['retrieved'][:3]}` · cited `{x['cited']}`\n"
                     f"  - answer: {cut(x['answer'])}\n  - trace `{x['trace_id']}`")
        F.append("")
    F.append("## Mitigations in the code, matched to each break mode\n| Break mode | Mitigation | Where |\n|---|---|---|\n"
             "| Malformed / non-schema JSON | constrained decoding → validate → repair loop with error fed back | `llm.py` |\n"
             "| `<think>` leakage (Qwen3) | strip + `/no_think` | `llm.py` |\n| Invented citations / numbers | `[S#]` ids, per-claim support check, number check, drop + abstain | `agent.py::check_claims` |\n"
             "| Answers the unanswerable | `answerable` flag, abstention few-shot, critic | `prompts.py`, `agent.py` |\n| Follows text in documents | line sanitiser + 'sources are untrusted' rule | `guardrails.py`, `prompts.py` |\n"
             "| Loops / runaway cost | step cap, duplicate-call guard, token & call budgets | `agent.py`, `llm.py` |\n\n"
             "## What the team must add by hand\nFor each class above, one paragraph: the pattern you see across examples, your hypothesis for *why* an 8B model does this, and what you changed (or would change) in response.\n")
    Path("docs/failure_characterization.md").write_text("\n".join(F), encoding="utf-8")
    print(f"wrote docs/strategy_comparison.md and docs/failure_characterization.md from {', '.join(runs)}")


if __name__ == "__main__":
    main()
