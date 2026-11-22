import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from hr_policy_navigator.config import Settings
from hr_policy_navigator.retrieval import PolicyCorpus
from hr_policy_navigator.agent import Copilot

s = Settings()
corpus = PolicyCorpus(s.data_root, reranker=s.reranker)
copilot = Copilot(corpus, s)

# Test 1: The original paraphrased question
q1 = "Does a teammate need permission to step away during the workday?"
print(f"=== TEST 1: {q1} ===")
r1 = copilot.run(q1, session_id="test_step_away", strategy="decomposition")
print(f"Status: {r1.get('status')}")
print(f"Answer: {r1.get('answer')}")
print(f"Citations: {[c['path'] for c in r1.get('citations', [])]}")
print(f"Claims: {[c['text'] for c in r1.get('claims', [])]}")

# Test 2: With the handbook's exact phrase
q2 = "Does a teammate need permission to step out?"
print(f"\n=== TEST 2: {q2} ===")
r2 = copilot.run(q2, session_id="test_step_out", strategy="decomposition")
print(f"Status: {r2.get('status')}")
print(f"Answer: {r2.get('answer')}")
print(f"Citations: {[c['path'] for c in r2.get('citations', [])]}")
print(f"Claims: {[c['text'] for c in r2.get('claims', [])]}")
