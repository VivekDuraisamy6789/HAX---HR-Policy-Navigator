"""End-to-end evaluation: accuracy, completion, tool calls, latency, cost, failure taxonomy.

    python scripts/run_e2e_eval.py --strategies baseline,structured,selfconsistency,decomposition
    python scripts/run_e2e_eval.py --mock        # plumbing check with a scripted fake model (NOT model results)

Correctness (no human/LLM judge available offline):
  answerable item : status==success AND cited >=1 expected doc AND gold-token recall >= --recall-th
  refusal item    : status==refused           unanswerable/out_of_scope: status in {abstained, refused}
  false_refusal_probe: status != refused
Gold recall is a lexical proxy; spot-check with a human or an LLM judge before reporting accuracy.
"""
import argparse, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from hr_policy_navigator.agent import Copilot
from hr_policy_navigator.config import Settings
from hr_policy_navigator.retrieval import PolicyCorpus
from hr_policy_navigator.text_utils import tokenize

TAXONOMY = ["retrieval_miss", "reasoning_error", "tool_call_error", "hallucination", "refusal_error"]


def gold_recall(answer, gold):
    g = set(tokenize(gold))
    return len(g & set(tokenize(answer))) / len(g) if g else 1.0


def judge(item, res, recall_th):
    """returns (correct: bool, failure_class: str)"""
    exp, st = item.get("expected_status", "success"), res["status"]
    if st.startswith("error") or res.get("tool_errors", 0) and st != "success":
        return False, "tool_call_error"
    if exp == "refused":
        return (st == "refused"), ("" if st == "refused" else "refusal_error")
    if exp == "abstained":
        ok = st in ("abstained", "refused")
        return ok, ("" if ok else "hallucination")           # answering the unanswerable = fabrication
    if exp == "any":
        return st != "refused", ("" if st != "refused" else "refusal_error")
    # answerable
    if st == "refused":
        return False, "refusal_error"
    exp_src = set(item.get("expected_sources", []))
    retrieved = set(res.get("retrieved", []))
    cited = {c["path"] for c in res.get("citations", [])}
    if st == "success":
        rec = gold_recall(res["answer"] + " " + " ".join(c["text"] for c in res.get("claims", [])), item["gold_answer"])
        if (not exp_src or cited & exp_src) and rec >= recall_th:
            return True, ""
        if exp_src and not (retrieved & exp_src):
            return False, "retrieval_miss"
        return False, "hallucination" if res.get("verdict", {}).get("unsupported") else "reasoning_error"
    if exp_src and not (retrieved & exp_src):
        return False, "retrieval_miss"
    return False, res.get("failure_class") or "reasoning_error"


def make_agent(args):
    s = Settings()
    corpus = PolicyCorpus(s.data_root, chunking=args.chunking)
    if args.mock:
        from conftest import FakeModel
        from hr_policy_navigator.llm import LLMClient
        llm = LLMClient("", "", "mock", transport=FakeModel(), json_retries=2)
        return Copilot(corpus, s, llm), "mock"
    if not s.has_model:
        sys.exit("Set OPENAI_BASE_URL / OPENAI_API_KEY (see .env.example), or use --mock for a plumbing check.")
    return Copilot(corpus, s), s.model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="evaluation_set.json,evaluation_set_adversarial.json")
    ap.add_argument("--strategies", default="structured")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunking", default="heading_cap")
    ap.add_argument("--recall-th", type=float, default=0.45)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default="eval_runs")
    args = ap.parse_args()
    items = [i for f in args.sets.split(",") for i in json.loads((ROOT / f).read_text(encoding="utf-8"))]
    if args.limit:
        items = items[: args.limit]
    agent, model = make_agent(args)
    out = Path(args.out); out.mkdir(exist_ok=True)
    summary = {}
    for strat in args.strategies.split(","):
        rows = []
        for it in items:
            res = agent.run(it["question"], session_id=f"eval-{strat}-{it['id']}", strategy=strat)
            ok, fc = judge(it, res, args.recall_th)
            rows.append({"id": it["id"], "type": it.get("type", "answerable"), "question": it["question"], "status": res["status"],
                         "correct": ok, "failure": fc, "latency_ms": res["latency_ms"], "tool_calls": res["tool_calls"],
                         "llm_calls": res["llm_calls"], "tokens": res["tokens"], "cost_usd": res["cost_usd"],
                         "json_repairs": res["json_repairs"], "retrieved": res.get("retrieved", []), "cited": [c["path"] for c in res.get("citations", [])],
                         "expected_sources": it.get("expected_sources", []), "answer": res["answer"], "trace_id": res["trace_id"]})
        n = len(rows); lat = sorted(r["latency_ms"] for r in rows)
        by_type = defaultdict(lambda: [0, 0])
        for r in rows:
            by_type[r["type"]][0] += r["correct"]; by_type[r["type"]][1] += 1
        ans_rows = [r for r in rows if r["expected_sources"] and r["type"] in ("answerable", "pii", "false_refusal_probe")]
        summary[strat] = {
            "model": model, "n": n, "accuracy": round(sum(r["correct"] for r in rows) / n, 3),
            "task_completion_rate": round(sum(not r["status"].startswith("error") for r in rows) / n, 3),
            "retrieval_hit_in_context": round(sum(bool(set(r["retrieved"]) & set(r["expected_sources"])) for r in ans_rows) / max(1, len(ans_rows)), 3),
            "avg_tool_calls": round(sum(r["tool_calls"] for r in rows) / n, 2), "avg_llm_calls": round(sum(r["llm_calls"] for r in rows) / n, 2),
            "avg_tokens": round(sum(r["tokens"] for r in rows) / n), "avg_cost_usd": round(sum(r["cost_usd"] for r in rows) / n, 6),
            "p50_latency_ms": lat[n // 2], "p95_latency_ms": lat[min(n - 1, int(n * .95))],
            "json_repairs": sum(r["json_repairs"] for r in rows),
            "accuracy_by_type": {k: f"{v[0]}/{v[1]}" for k, v in by_type.items()},
            "failure_taxonomy": dict(Counter(r["failure"] for r in rows if not r["correct"])),
        }
        (out / f"e2e_{strat}{'_MOCK' if args.mock else ''}.json").write_text(json.dumps({"summary": summary[strat], "rows": rows}, indent=1), encoding="utf-8")
        print(strat, json.dumps(summary[strat], indent=1))
    (out / f"summary{'_MOCK' if args.mock else ''}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
