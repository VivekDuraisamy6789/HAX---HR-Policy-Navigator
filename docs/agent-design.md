# Multi-Agent HR Policy Navigator Design

## Architecture

The system uses a planner → executor → critic loop with a retrieval-first toolchain.

1. Planner
   - Parses the user's request.
   - Classifies it as policy lookup, ambiguous guidance, or refusal case.
   - Selects the smallest relevant retrieval strategy.

2. Retrieval tools
   - `search_documents(query, top_k=8)`
   - `get_document_context(path, snippet_window=300)`
   - `collect_citations(results)`

3. Executor
   - Builds a grounded answer from the top retrieved chunks.
   - Uses the qwen 8b model through an OpenAI-compatible endpoint.
   - Returns structured JSON with answer, citations, confidence, and whether a refusal was required.

4. Critic
   - Checks for unsupported claims, missing citations, and policy safety issues.
   - Retries with stricter prompts if needed.
   - Rejects hallucinated or non-cited answers.

## Tools required

The system includes at least these tools:

- Policy search tool: uses BM25 + dense retrieval and returns ranked chunks.
- Context fetch tool: loads precise document ranges and section headings.
- Citation validation tool: confirms every claim maps back to a source path.
- Refusal guard: blocks unsupported or sensitive requests.

## Persistent state

The app tracks:

- conversation history
- retrieval results per turn
- tool call trace
- answer metadata and confidence
- failure taxonomy

## Error handling

The system handles:

- malformed JSON from the model
- empty retrieval results
- model timeouts or API errors
- infinite loops by enforcing max tool budget and max iterations
- token budget caps on the prompt and response

## Observability

The app is designed for Langfuse or OpenTelemetry tracing with fields for:

- query
- tool names and durations
- retrieval scores
- model name
- answer creation stage
- failure class

## Responsible AI guardrails

- Prompt injection tests against the document corpus
- PII handling: no storage of personal employee details
- Refusal behavior for consequential actions
- Human-in-the-loop boundary for employment decisions
- Conservative answer generation based on retrieved docs only
