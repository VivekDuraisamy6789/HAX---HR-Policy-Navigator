import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import MAX_TOOL_CALLS, MAX_TOKENS, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, has_model_config
from .retrieval import PolicyCorpus, RetrievalHit


class QwenClient:
    def __init__(self) -> None:
        self.client = None
        if has_model_config():
            self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    def chat_completion(self, messages: List[Dict[str, str]], response_format: Optional[Dict[str, str]] = None) -> str:
        if self.client is None:
            raise RuntimeError("OpenAI-compatible endpoint is not configured")
        payload = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": MAX_TOKENS,
        }
        if response_format:
            payload["response_format"] = response_format
        response = self.client.chat.completions.create(**payload)
        return response.choices[0].message.content or ""


class PlannerExecutorCriticAgent:
    def __init__(self, corpus: PolicyCorpus):
        self.corpus = corpus
        self.model = QwenClient()

    @staticmethod
    def _is_refusal_required(question: str) -> bool:
        lower = question.lower()
        blocked = [
            "termination",
            "fire",
            "discipline",
            "legal advice",
            "private employee record",
            "personally identifiable",
            "ssn",
            "salary",
            "disability accommodation decisions",
            "without a manager",
        ]
        return any(term in lower for term in blocked)

    def plan(self, query: str) -> Dict[str, Any]:
        state = {"query": query, "tool_calls": 0, "stage": "planning"}
        if self._is_refusal_required(query):
            state["needs_refusal"] = True
            state["reason"] = "Consequence-heavy or outside-policy question"
        else:
            state["needs_refusal"] = False
        return state

    def _retrieve(self, query: str, top_k: int = 8) -> List[RetrievalHit]:
        return self.corpus.search(query, top_k=top_k)

    def _build_answer_prompt(self, query: str, hits: List[RetrievalHit]) -> List[Dict[str, str]]:
        context_blocks = []
        for hit in hits:
            if not hit.file_path:
                continue
            context_blocks.append(
                f"Source: {hit.file_path}\nTitle: {hit.title}\n\n{hit.content[:1500]}\n"
            )
        context_text = "\n---\n".join(context_blocks)
        return [
            {
                "role": "system",
                "content": (
                    "You are a conservative HR policy navigator. Use only the provided source material. "
                    "Cite every factual answer with file paths. If the answer cannot be supported, say so. "
                    "Do not invent policy details."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\nOnly answer from the following policy excerpts.\n\n{context_text}\n\n"
                    "Return JSON with keys: answer, citations, confidence, follow_up. "
                    "citations should be a list of relative file paths."
                ),
            },
        ]

    def _critic_check(self, answer_json: Dict[str, Any], hits: List[RetrievalHit]) -> Dict[str, Any]:
        citations = answer_json.get("citations", [])
        answer = answer_json.get("answer", "")
        if not citations:
            answer_json["critique"] = "Missing citations; the answer was rejected and rewritten."
            answer_json["needs_retry"] = True
            return answer_json
        for citation in citations:
            if citation not in {hit.file_path for hit in hits if hit.file_path}:
                answer_json["critique"] = f"Citation {citation} is not traceable to the retrieval set."
                answer_json["needs_retry"] = True
                return answer_json
        answer_json["critique"] = "Answer is grounded and cites the retriever evidence."
        answer_json["needs_retry"] = False
        return answer_json

    def run(self, query: str) -> Dict[str, Any]:
        state = self.plan(query)
        if state.get("needs_refusal"):
            return {
                "answer": "I can help with policy questions based on the repository, but I cannot approve or decide an employment action without a human policy owner or manager review.",
                "citations": [],
                "confidence": "low",
                "status": "refused",
                "tool_calls": 1,
            }

        hits = self._retrieve(query, top_k=8)
        state["tool_calls"] += 1

        if not hits or not hits[0].file_path:
            return {
                "answer": "I could not find policy evidence in the repository for this question. Please rephrase it or ask a question grounded in the HR documents.",
                "citations": [],
                "confidence": "low",
                "status": "no_results",
                "tool_calls": state["tool_calls"],
            }

        prompt = self._build_answer_prompt(query, hits)
        try:
            raw = self.model.chat_completion(prompt, response_format={"type": "json_object"})
        except Exception as exc:
            return {
                "answer": f"The model endpoint failed while generating the answer: {exc}. I recommend retrying with a more specific question.",
                "citations": [hit.file_path for hit in hits[:2] if hit.file_path],
                "confidence": "low",
                "status": "model_error",
                "tool_calls": state["tool_calls"],
            }

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.S)
            if match:
                parsed = json.loads(match.group(0))
            else:
                parsed = {
                    "answer": raw,
                    "citations": [hit.file_path for hit in hits[:2] if hit.file_path],
                    "confidence": "low",
                    "follow_up": "Please ask for a simpler policy lookup.",
                }

        checked = self._critic_check(parsed, hits)
        if checked.get("needs_retry"):
            retry_prompt = [
                {
                    "role": "system",
                    "content": "Rewrite the answer so every factual statement is tied to a source file and do not speculate.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\nOriginal answer: {json.dumps(parsed, ensure_ascii=False)}\n\n"
                        "Use only the following trusted excerpts and cite them.\n\n"
                        + "\n---\n".join(f"{hit.file_path}\n{hit.content[:1500]}" for hit in hits[:4])
                    ),
                },
            ]
            raw_retry = self.model.chat_completion(retry_prompt, response_format={"type": "json_object"})
            try:
                checked = json.loads(raw_retry)
                checked = self._critic_check(checked, hits)
            except Exception:
                pass

        final = {
            "answer": checked.get("answer", parsed.get("answer", "I could not determine a grounded answer.")),
            "citations": checked.get("citations", [hit.file_path for hit in hits[:3] if hit.file_path]),
            "confidence": checked.get("confidence", "medium"),
            "follow_up": checked.get("follow_up", ""),
            "status": "success",
            "tool_calls": state["tool_calls"],
            "retrieval_hits": [hit.file_path for hit in hits[:4] if hit.file_path],
        }
        return final
