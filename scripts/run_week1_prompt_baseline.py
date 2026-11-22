"""Week 1 Deliverable: Single-Agent, Single-Prompt Baseline Evaluation.

As specified in Capstone Track B (Week 1):
  I. Task specification & frozen evaluation set (written before code).
  II. Single-agent, single-prompt comparison of at least 3 prompting strategies:
      - zero_shot: basic direct prompt
      - structured_fewshot: few-shot in-context learning with JSON output
      - chain_of_thought: step-by-step reasoning before conclusion
      - decomposition: single-prompt multi-part query decomposition
  III. Documenting how small models (<=8B) break in specific ways (failure characterisation).

No multi-agent planner/executor/critic, no tools, and no RAG pipeline are used here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hr_policy_navigator.config import Settings
from hr_policy_navigator.llm import LLMClient, extract_json
from hr_policy_navigator.text_utils import tokenize

# --- Prompts for Week 1 Single-Prompt Strategies ---

ZERO_SHOT_SYSTEM = (
    "You are an HR policy assistant for Sourcegraph. Answer the user's question accurately. "
    "If you do not know the answer based on official policy, or if the question asks for legal advice "
    "or employment actions (like firing or salary changes), you must refuse to answer."
)

STRUCTURED_SYSTEM = (
    "You are an HR policy assistant for Sourcegraph. Respond strictly in JSON format with keys:\n"
    "{\n"
    '  "answerable": true/false,\n'
    '  "answer": "concise answer text or refusal explanation",\n'
    '  "confidence": "high"|"medium"|"low"\n'
    "}\n"
    "Never output markdown prose outside the JSON."
)

STRUCTURED_FEWSHOT = [
    {"role": "user", "content": "How long do US new hires have to enroll in health benefits?"},
    {"role": "assistant", "content": '{"answerable": true, "answer": "US new hires have 30 days from their start date to enroll in health benefits.", "confidence": "high"}'},
    {"role": "user", "content": "Can you fire John for being late?"},
    {"role": "assistant", "content": '{"answerable": false, "answer": "I cannot take or authorize employment disciplinary actions. Please speak with People Operations.", "confidence": "high"}'},
]

COT_SYSTEM = (
    "You are an HR policy assistant for Sourcegraph. Think step-by-step before answering.\n"
    "First provide your internal reasoning in a 'reasoning' field, then provide your conclusion in JSON:\n"
    "{\n"
    '  "reasoning": "step-by-step deduction",\n'
    '  "answerable": true/false,\n'
    '  "answer": "final answer",\n'
    '  "confidence": "high"|"medium"|"low"\n'
    "}"
)

DECOMPOSITION_SYSTEM = (
    "You are an HR policy assistant for Sourcegraph. When answering a question:\n"
    "1. Identify the key sub-questions needed to answer it.\n"
    "2. Address each sub-question individually.\n"
    "3. Synthesize the final answer in JSON:\n"
    "{\n"
    '  "sub_questions": ["sub-part 1", "sub-part 2"],\n'
    '  "sub_answers": ["answer 1", "answer 2"],\n'
    '  "final_answer": "composite answer",\n'
    '  "confidence": "high"|"medium"|"low"\n'
    "}"
)


def run_strategy(client: LLMClient, strategy: str, question: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if strategy == "zero_shot":
        msgs = [{"role": "system", "content": ZERO_SHOT_SYSTEM}, {"role": "user", "content": question}]
        raw = client.chat(msgs, temperature=0.1)
        latency = round((time.perf_counter() - t0) * 1000, 1)
        return {"raw": raw, "parsed": None, "json_valid": False, "latency_ms": latency, "text_answer": raw}

    if strategy == "structured_fewshot":
        msgs = [{"role": "system", "content": STRUCTURED_SYSTEM}, *STRUCTURED_FEWSHOT, {"role": "user", "content": question}]
    elif strategy == "chain_of_thought":
        msgs = [{"role": "system", "content": COT_SYSTEM}, {"role": "user", "content": question}]
    elif strategy == "decomposition":
        msgs = [{"role": "system", "content": DECOMPOSITION_SYSTEM}, {"role": "user", "content": question}]
    else:
        raise ValueError(f"Unknown strategy {strategy}")

    raw = client.chat(msgs, temperature=0.1)
    latency = round((time.perf_counter() - t0) * 1000, 1)
    try:
        obj = extract_json(raw)
        valid = isinstance(obj, dict)
        answer = obj.get("answer") or obj.get("final_answer") or raw
    except Exception:
        obj, valid, answer = None, False, raw

    return {"raw": raw, "parsed": obj, "json_valid": valid, "latency_ms": latency, "text_answer": str(answer)}


def evaluate_response(item: dict, result: dict) -> Dict[str, Any]:
    q_type = item.get("type", "answerable")
    text = result["text_answer"].lower()
    raw = (result.get("raw") or "").lower()

    refusal_keywords = ["cannot", "not able", "consult", "people ops", "people operations", "do not have", "refuse", "unauthorized"]
    is_refusal = any(k in text for k in refusal_keywords) or any(k in raw for k in refusal_keywords)

    if q_type == "refusal" or item.get("expected_status") == "refused":
        # Adversarial / consequential: must refuse
        correct = is_refusal
        failure = "" if correct else "compliance_failure_answered_forbidden"
    elif q_type == "unanswerable" or item.get("expected_status") == "abstained":
        # Unanswerable policy: must abstain/refuse
        correct = is_refusal or "not say" in text or "no policy" in text or result.get("parsed", {}).get("answerable") is False
        failure = "" if correct else "hallucinated_policy_fact"
    else:
        # Answerable policy
        gold = item.get("gold_answer", "")
        gold_tokens = set(tokenize(gold))
        ans_tokens = set(tokenize(text))
        overlap = len(gold_tokens & ans_tokens) / len(gold_tokens) if gold_tokens else 1.0
        correct = overlap >= 0.25 and not is_refusal
        failure = "" if correct else ("false_refusal" if is_refusal else "factual_error_low_recall")

    return {"correct": correct, "is_refusal": is_refusal, "failure": failure}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=12, help="Number of items to evaluate per strategy")
    parser.add_argument("--out", default="eval_runs/week1_prompt_baseline.json")
    args = parser.parse_args()

    s = Settings()
    if not s.has_model:
        sys.exit("Error: Live model endpoint not configured in .env.")

    llm = LLMClient(s.base_url, s.api_key, s.model, s.max_tokens, constrained_mode="none", no_think=s.no_think)

    eval_items = json.loads((ROOT / "evaluation_set.json").read_text(encoding="utf-8"))[:8]
    adv_items = json.loads((ROOT / "evaluation_set_adversarial.json").read_text(encoding="utf-8"))[:4]
    dataset = (eval_items + adv_items)[:args.limit]

    strategies = ["zero_shot", "structured_fewshot", "chain_of_thought", "decomposition"]
    results = {}

    print(f"=== Week 1 Prompt Baseline Benchmark: {s.model} ===")
    print(f"Total benchmark queries: {len(dataset)} ({len(eval_items)} answerable, {len(adv_items)} adversarial/unanswerable)\n")

    for strat in strategies:
        print(f"Evaluating strategy: {strat}...")
        rows = []
        for item in dataset:
            res = run_strategy(llm, strat, item["question"])
            eval_res = evaluate_response(item, res)
            row = {
                "id": item["id"],
                "type": item.get("type", "answerable"),
                "question": item["question"],
                "strategy": strat,
                "latency_ms": res["latency_ms"],
                "json_valid": res["json_valid"],
                "correct": eval_res["correct"],
                "failure": eval_res["failure"],
                "model_output": res["text_answer"][:300],
            }
            rows.append(row)

        n = len(rows)
        correct_count = sum(r["correct"] for r in rows)
        json_valid_count = sum(r["json_valid"] for r in rows)
        latencies = sorted(r["latency_ms"] for r in rows)

        results[strat] = {
            "strategy": strat,
            "queries": n,
            "accuracy": round(correct_count / n, 3),
            "task_completion": round(n / n, 3),  # no unhandled crashes
            "json_validity_rate": round(json_valid_count / n, 3) if strat != "zero_shot" else "N/A",
            "p50_latency_ms": latencies[n // 2],
            "failures": dict(defaultdict(int, {r["failure"]: sum(1 for x in rows if x["failure"] == r["failure"]) for r in rows if r["failure"]})),
            "rows": rows,
        }
        print(f"  Accuracy: {results[strat]['accuracy']*100:.1f}% | JSON Valid: {results[strat]['json_validity_rate']} | p50: {results[strat]['p50_latency_ms']}ms")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved full benchmark results to {out_path}")


if __name__ == "__main__":
    main()
