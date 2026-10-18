# Current state (audited 2026-10-18, before the Draco layer was added)

Evidence: source read of `app/`, baseline test run, `git log`. Nothing below is taken from the README (which is two lines).

## What the repository was
A FastAPI RAG/agent **service**, not a package: import root `app/`, ~14k lines of Python, no `pyproject.toml`, no public exports (every `__init__.py` empty), no examples, no lint or type config. Baseline: **328 passed, 15 skipped** (pytest) / 343 tests OK, 15 skipped (the CI command, `unittest discover`).

| Area | What exists (verified in source) | Gaps found |
|---|---|---|
| Ingestion `app/ingestion` | Docling convert + OCR detector, hybrid chunker, BGE-M3 embed, Qdrant + OpenSearch writes with deterministic chunk ids, SQLite registry, RQ worker | Import-time env config, lazy global model/clients, PDF only, `print()` in worker |
| Retrieval `app/retrieval` | `RetrievalService` with injectable dense/lexical/reranker/generator, RRF, ColBERT rerank, calibrated selection, per-request `Trace`, degraded mode, cancellation re-raised | `resolve_filters` is a no-op (no ACLs); imports torch via `ingestion.embed`; `get_service()` global singleton; no production generator class |
| Inference `app/inference` | vLLM via `ChatOpenAI`, `ChatOrchestrator` rag/agent modes, `map_error`, `ConversationStore`/`MemoryProvider` Protocols, SQLite store | Qwen-specific; duplicated `build_llm` |
| Agent `app/agent` | LangGraph `AgentRunner` (per-request graph), `CapabilityRegistry.execute` (permissions, validation, timeout, size cap, error mapping), 21 capabilities, `ContextBudget`, `EvidenceStore`, SSRF-safe web fetcher | `requires_approval`/`parallel_safe`/`cost` flags are declared but **not enforced** by the executor; `InMemorySaver` per request (no durability, no resume); two agent loops (`ChatOrchestrator` and `AgentRunner`) |
| Platform `app/core`, `app/api` | Pure-ASGI middleware, Redis rate limit, auth, metrics | Module-global state (`_runner`, `_limiter`, `_sem`) |
| Evals | Opt-in scripts against the live stack (`scripts/eval_*`) | Not run in CI; no deterministic control-plane evals |

Cross-cutting: no common exception base (`RetrievalError`, `InferenceError`, `ToolResult` envelopes, `HTTPException`); four settings dataclasses plus raw `os.getenv`; `Services` fields typed `Any`; no execution identity shared across layers (retrieval `request_id`, agent `trace_id`, capability `execution_id` were all unrelated); no governance beyond per-request step limits.

## Seams that made reuse possible
`RetrievalService` constructor injection, `Services` provider container, `RuntimeAdapter(client_cls=...)`, `ChatOrchestrator(llm_factory=...)`, the capability registry's `ToolResult` envelope, and the conversation-store Protocols. Draco builds on these; none were modified.

## Unverified / not exercisable here
Live vLLM, Qdrant, OpenSearch, Redis, the Docker image and GitHub Actions were not available; everything touching them is tested only through the repo's own fakes.
