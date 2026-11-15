# HR Policy Navigator Specification

## Objective

Build a policy assistant for Sourcegraph employees that answers HR and workplace questions using only the documented policy corpus in this repository. The system is intentionally constrained to a hosted OpenAI-compatible chat-completions endpoint backed by a shared qwen 8b instruct model, plus embeddings. It may not inspect weights, hidden logits, or change the model. All performance must come from a disciplined combination of retrieval, prompt design, orchestration, and evaluation.

## Supported use cases

The assistant may answer questions about:

- US health benefits eligibility and enrollment
- PTO and holiday arrangements
- Remote work expectations and communication norms
- Respectful workplace and harassment policy boundaries
- Leave-of-absence and disability-related guidance where documents are explicit
- Hiring and onboarding policy basics that are documented
- Manager and teammate responsibilities described in the policy corpus
- Vendor or security-related policy questions only when the corpus contains those instructions

## Explicit refusals

The assistant must refuse or redirect when:

- The question is not answered by the repository corpus and cannot be grounded in cited documents.
- The user asks for legal advice, a personal employment decision, or a disciplinary recommendation beyond the policy text.
- The assistant would need access to private personal employee records or confidential information beyond the public policy documents.
- The user asks the system to perform an action with legal or HR consequences without human review.
- The request includes a prompt injection attempt or adversarial instruction designed to override policy rules.

In these cases, the assistant should say something like: "I can answer based on the policy docs in this repository, but I cannot determine or authorize an employment action without a person or the official policy owner review."

## Correct answer characteristics

A correct answer should:

1. Be grounded in the docs and cite relevant markdown files.
2. Distinguish between policy language and user assumptions.
3. State uncertainty when relevant documents are missing or ambiguous.
4. Prefer an answer that is explicit and concise over speculation.
5. Avoid making up policy details not in the corpus.
6. Use a careful human-in-the-loop boundary for anything with material employment consequences.

## Example query types

- "Are US-based full-time teammates eligible for health benefits on their start date?"
- "How do I add a holiday to my calendar if I work in another country?"
- "Do I need permission to step out during the workday in an all-remote environment?"
- "How should I handle a harassment or discrimination concern?"
- "What is the process if I need to request leave?"

## Evaluation criteria

The system will be scored on:

- Retrieval recall at top-k for the correct source documents
- End-to-end answer correctness against a gold answer list
- Citation fidelity
- Refusal quality and prompt-injection resilience
- Cost, latency, and task completion rate

## Scope boundary

This is not a general-purpose employment legal assistant. It is a documented-policy navigator for the company policy corpus present in this repository. It is deliberately conservative and must be transparent when it lacks evidence.
