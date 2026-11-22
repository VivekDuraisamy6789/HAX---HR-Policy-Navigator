"""Smoke-test your hosted Qwen-8B endpoint BEFORE running evals. Prints what works and what to set.

    python scripts/check_endpoint.py
Checks: chat call, <think> leakage, each constrained-decoding mode (json_schema / guided_json /
json_object / none) against the real answer schema, and the embeddings endpoint.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hr_policy_navigator.config import Settings
from hr_policy_navigator.llm import LLMClient, extract_json
from hr_policy_navigator.prompts import ANSWER_SCHEMA, ANSWER_RULES, FEWSHOT
import jsonschema

s = Settings()
if not s.has_model:
    sys.exit("Set OPENAI_BASE_URL and OPENAI_API_KEY in .env first.")
print(f"model={s.model}  no_think={s.no_think}  base_url={s.base_url}")
q = ("SOURCES:\n[S1] (benefits.md > Enrollment) New hires have 30 days from their start date to enroll in health benefits.\n\n"
     "QUESTION: How long do new hires have to enroll?")
msgs = [{"role": "system", "content": ANSWER_RULES + "\nRespond with a single JSON object only."}, *FEWSHOT, {"role": "user", "content": q}]
ok_modes = []
for mode in ["json_schema", "guided_json", "json_object", "none"]:
    llm = LLMClient(s.base_url, s.api_key, s.model, s.max_tokens, mode, no_think=s.no_think)
    llm.auto_mode = False
    t = time.perf_counter()
    try:
        raw = llm._create(msgs, 0.1, s.max_tokens, ANSWER_SCHEMA)["text"]
        leaked = "<think>" in raw.lower()
        obj = extract_json(raw); jsonschema.validate(obj, ANSWER_SCHEMA)
        print(f"  [{mode:12s}] OK   {time.perf_counter()-t:4.1f}s  think_leak={leaked}  answer={obj['answer'][:60]!r}")
        ok_modes.append(mode)
    except Exception as e:
        print(f"  [{mode:12s}] FAIL {type(e).__name__}: {str(e)[:110]}")
print("\nRecommended CONSTRAINED_MODE:", ok_modes[0] if ok_modes else "none (rely on the repair loop)")
if s.embedding_model:
    try:
        from hr_policy_navigator.retrieval import OpenAIEmbedder
        emb_base = s.embedding_base_url or s.base_url
        emb_key = s.embedding_api_key or s.api_key
        v = OpenAIEmbedder(emb_base, emb_key, s.embedding_model, cache_dir=".cache/embcheck").encode_query("health insurance")
        print(f"embeddings OK: model={s.embedding_model} dim={len(v)}")
    except Exception as e:
        print(f"embeddings FAIL: {str(e)[:150]}")
else:
    print("EMBEDDING_MODEL not set -> using offline LSA fallback. A chat model cannot embed; set a real embedding model (e.g. Qwen3-Embedding, bge-m3).")
