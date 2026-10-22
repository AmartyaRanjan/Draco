# Draco: what exists today and where it is going

Written 2026-10-14. This is the entry point to the project. It summarises what has been built, then lays out the plan to turn it into a reusable Python package for building RAG and agentic systems. Detailed evidence for each part lives in the linked documents; this file does not repeat their numbers beyond the headline ones.

Contents: [1 Purpose](#1-purpose) · [2 What is built today](#2-what-is-built-today) · [3 Known gaps](#3-known-gaps-today) · [4 The package plan](#4-the-package-plan) · [5 Design principles](#5-design-principles) · [6 Proposed layout](#6-proposed-package-layout) · [7 Roadmap](#7-roadmap) · [8 Risks](#8-risks-and-how-to-avoid-them) · [9 Document map](#9-document-map)

---

## 1. Purpose

Draco began as a hands-on project on agentic RAG: agents in the loop, tools, and agent evaluation. It is now a working service with ingestion, hybrid retrieval, grounded chat, a tool-using agent, an evaluation suite and a production hardening layer.

The next step is to extract the reusable parts into a **Python package** so that a new RAG or agent system takes hours rather than weeks. The package is for the author, their company and friends. It is **not** meant to compete with Haystack, LangGraph or LlamaIndex. It is an opinionated toolkit with ready-made recipes, shaped by the systems actually being built.

---

## 2. What is built today

About 8,500 lines of Python in `app/`, 27 commits, 343 tests (per [production.md](../production.md)). The service is a FastAPI app over four backing services: Redis, Qdrant, OpenSearch and a vLLM server.

### 2.1 Ingestion (`app/ingestion/`, `app/db/`)

- Upload a PDF, queue it on RQ (Redis), and a worker converts, chunks, embeds and indexes it.
- Docling for conversion, with a PDF detector that decides per page whether OCR is needed (page routing, gap bridging, an OCR-fraction fallback).
- Token-aware chunking, with tiny chunks merged into their neighbours.
- Pages that yielded almost no text are flagged as `low_yield` instead of silently missing from the index.
- BGE-M3 embeddings into Qdrant (dense) and OpenSearch (BM25).
- SQLite document registry with content-hash duplicate detection.
- Details: `CHANGELOG.md` (iterations 1 to 5), `docs/ingestion-map.html`.

### 2.2 Retrieval (`app/retrieval/`)

Dense and BM25 retrieval in parallel, exact `chunk_id` deduplication, Reciprocal Rank Fusion (k=60), a ColBERT rerank that shares the BGE-M3 backbone, a relevance threshold with a minimum-evidence fallback, and token-budgeted context building with `[S#]` evidence labels. Retrieval p50 is about 464 ms. Details: [RETRIEVAL_LOG.md](../RETRIEVAL_LOG.md).

### 2.3 Inference (`app/inference/`, `deploy/vllm/`)

- A grounded chat orchestrator with streaming, conversation memory (SQLite) and citation extraction.
- Qwen3-4B-FP8 served by vLLM with a pinned image and revision, thinking mode toggled per request, hermes tool-call parsing.
- A recorded search for a QAT checkpoint found none that could be verified, so a PTQ FP8 model is used.
- Details: [INFERENCE_LOG.md](../INFERENCE_LOG.md).

### 2.4 Agent (`app/agent/`)

A three-level hierarchy built on LangGraph, documented in [tools.md](../tools.md):

1. **Supervisor graph.** Classify the request, configure the reasoning mode, plan capabilities, execute, check the context budget, summarise if needed, evaluate the evidence, generate the answer, validate citations.
2. **Capability groups.** 21 capabilities across knowledge, web research, knowledge management, artifact creation, visualization, file analysis, computation, validation and context management.
3. **Atomic services.** Retrieval, the registry, the ingestion queue, web fetcher, search and image providers, an artifact store, an agent store and the runtime adapter.

Notable mechanisms:
- **Deterministic first.** Classification, planning, routing and retries are plain functions. A grounded question costs one LLM call and a calculation costs none.
- **Structural provenance.** Citations are built from a run-scoped evidence store and never from model-written ids.
- **Approval-gated ingestion.** The agent can propose adding a web source to the knowledge base, but a human approves it.
- **Safety controls on web content.** SSRF defences, size and redirect limits, and an injection filter.
- **Artifacts.** DOCX, PDF, PPTX, XLSX, CSV and chart output.
- **Context-budget management** with hierarchical summarisation.

### 2.5 Platform hardening (`app/core/`, `app/api/`)

API-key authentication on every data route with brute-force lockout, a Redis-backed token-bucket rate limiter, an LLM admission gate, request-size caps, upload validation, JSON logs with request ids, Prometheus `/metrics`, liveness and readiness probes, boot-time config validation, a Dockerfile and a CI workflow. Scope: a single-tenant trust domain. Details: [production.md](../production.md).

### 2.6 Evaluation (`scripts/`, `tests/eval/`, `eval_results/`)

An 84-question RAG gold set and a 45-case agent suite, covering retrieval ablations, judged generation, tool calling, multi-step tasks, reliability (pass^k), prompt-injection safety and fault injection. Headline numbers: hit@5 0.79 as users write questions and 0.97 with the document descriptor removed, judged faithfulness 0.95, 8 of 8 unanswerable questions correctly declined, agent tool selection 0.71. Evaluation found two real agent defects that code review did not, and both were fixed. These are not leaderboard-comparable numbers. Details: [evals.md](../evals.md).

---

## 3. Known gaps today

These are the most important items from the existing documents. The full lists are in [production.md](../production.md) section 6 and [tools.md](../tools.md) section 10.

1. **No per-user document access control.** Any authenticated user can retrieve any indexed document.
2. **Prompt injection is mitigated, not solved.** A paraphrased injection was not caught by the filter.
3. **Retrieval is sensitive to phrasing.** Contextual chunk headers and query rewriting would help, but both need re-indexing.
4. **The 4B model skips tools.** It does arithmetic in its head instead of calling the calculator, and it misses parallel tool use.
5. **Agent graph state is in memory.** A restart loses an in-flight run, and approval is not a LangGraph `interrupt`.
6. **Single-host data stores,** with no replication or backup.
7. **The Docker image and CI workflow have never been built or run.**
8. **Runtime uploads and local databases were committed in repository history.** The files are excluded from the current tree and future commits, but remain available in published Git history until repository administrators coordinate history cleanup and cache remediation. Treat previously committed contents as exposed and rotate any credentials they may have contained.

---

## 4. The package plan

### 4.1 Goal

A package that lets the author, colleagues and friends build the following with little boilerplate:

| System | What it is |
|---|---|
| RAG pipeline | Ingest, chunk, embed, index, retrieve, rerank, answer with citations |
| Single agentic RAG | One agent whose `search` tool is a retrieval pipeline |
| Multi-agent system | A supervisor delegating to worker agents |
| Agent orchestration | Graphs of agents and tools with routing, budgets and approvals |

Agentic RAG is not a separate mode: it is an agent that has a retrieval pipeline as one of its tools. That keeps the four use cases on one small core.

### 4.2 What makes it different from the frameworks

It is shaped by one person's repeated needs, with evaluation and observability built in. Evaluations and the deterministic-first agent design in this repository are the parts most worth packaging.

---

## 5. Design principles

1. **Extract, do not invent.** Build two or three concrete systems first, then pull out what they share. Code that gets copied between projects is the real API.
2. **Small core, many recipes.** The core stays tiny. Prebuilt recipes ("RAG pipeline", "agentic RAG", "supervisor and workers") carry the convenience.
3. **Components behind interfaces.** LLMs, embedders, vector stores, rerankers, converters and search providers are adapters. The current code already has the seams (`RuntimeAdapter`, `SearchProvider`, `ImageProvider`, `Services`).
4. **Optional dependencies.** Heavy integrations are extras (for example `pkg[qdrant]`, `pkg[docling]`, `pkg[agents]`), so a user who wants basic RAG does not install torch and Docling.
5. **Configuration over code.** One config selects the model, store, embedder and chunking, so a new system starts from configuration.
6. **Evaluation and tracing are first-class.** Every recipe ships with an eval harness and request tracing, because this is where existing frameworks are weakest.
7. **Deterministic first.** Do in plain code what plain code can do, and use the LLM for the rest. This keeps cost and latency down and makes behaviour testable.
8. **Safe defaults.** Auth, rate limits, SSRF defences and approval gates are on by default and are not an afterthought.
9. **Honest documentation.** Every claim states what was measured and what was not, as in the existing evaluation and production documents.

---

## 6. Proposed package layout

This is a **starting sketch**, to be revised as extraction proceeds. The mapping to current code shows where each part would come from.

```
src/<package>/
  core/            Component and Tool interfaces, Pipeline (DAG), settings, errors, tracing
                   <- app/core/*, app/retrieval/schema.py, app/retrieval/timing.py
  ingestion/       converters, chunkers, embedders, indexers, worker
                   <- app/ingestion/*, app/db/registry.py
  retrieval/       retrievers, fusion, rerankers, selection, context building
                   <- app/retrieval/*
  stores/          Qdrant, OpenSearch (and later others) as adapters
                   <- app/db/qdrant.py, app/db/opensearch.py
  llm/             model profiles, runtime adapter, generation, memory
                   <- app/inference/*, app/agent/runtime.py
  agents/          agent loop, capability registry, budget, summarisation, evidence store
                   <- app/agent/* (registry, budget, evidence, summarize, graph)
  capabilities/    ready-made tools (search, web, artifacts, charts, calculation, validation)
                   <- app/agent/capabilities/*
  orchestration/   supervisor/worker patterns, routing, approvals, durable state
                   <- app/agent/graph.py (generalised)
  evals/           retrieval metrics, RAG judging, agent suites, safety cases, harness
                   <- scripts/eval_*.py, tests/eval/*
  serve/           optional FastAPI layer: auth, rate limits, health, metrics
                   <- app/api/*, app/core/middleware.py, ratelimit.py
  recipes/         rag, agentic_rag, supervisor_workers
  cli/             `<package> init <name> --template agentic-rag`
```

The existing FastAPI service becomes the first consumer of the package, which keeps the package honest.

---

## 7. Roadmap

Phases are ordered by dependency. Each ends with something that works, so the package never sits half-built.

### Phase 0: housekeeping (before any refactor)
- Keep runtime uploads and local databases out of Git; they are now ignored and excluded from package data. Existing public Git history still requires coordinated cleanup.
- Replace the placeholder `README.md` with a pointer to this file.
- Build the Docker image and run the CI workflow once, and fix what breaks.
- Consolidate the duplicate `enable_thinking` mapping in `app/inference/client.py` into `app/agent/runtime.py`.

### Phase 1: package skeleton
- `pyproject.toml`, `src/` layout, a package name, and a version policy (semantic versioning with a changelog).
- Move modules without changing behaviour. All 343 tests must still pass, and that is the acceptance test.
- Define optional-dependency extras.

### Phase 2: the core
- A `Component` interface with typed inputs and outputs, a `Pipeline` for composing components, and a `Tool` interface.
- Adapter interfaces for LLM, embedder, vector store, lexical store, reranker and converter.
- Rebuild the current ingestion and retrieval on these interfaces. This is the **RAG recipe**, and it is the proof that the core is right.

### Phase 3: agent layer
- Generalise the capability registry and supervisor graph so they are not tied to Draco's 21 capabilities.
- The **agentic RAG recipe**: an agent whose search tool is the Phase 2 retrieval pipeline.
- Close the known agent gaps: durable checkpoints and real `interrupt()` approval, and forced tool use as a configurable policy.

### Phase 4: multi-agent and orchestration
- The **supervisor and workers recipe**, with routing, per-agent tool scoping, shared evidence and budgets.
- Add parallel fan-out, human-in-the-loop and durable state only when a real use case needs them.

### Phase 5: evaluation and observability as package features
- Package the retrieval metrics, judge, agent suite and safety cases behind a small API, so every recipe can be evaluated with one command.
- Tracing hooks and a metrics exporter.

### Phase 6: developer experience
- Config-driven setup and the `init` scaffolding CLI with one template per recipe.
- One worked example per recipe, plus API docs.
- Optionally publish to a private index for the company.

### Parallel track: closing platform gaps
Per-user document ownership and access control (the owner column in the registry and `resolve_filters`), better prompt-injection filtering, contextual chunk headers, a larger agent eval set (9 regression cases to 50 or more), the Qwen3-8B-AWQ model upgrade when GPU headroom allows, and streaming for `/agent/chat`.

---

## 8. Risks and how to avoid them

| Risk | Mitigation |
|---|---|
| Designing the API before enough real use, then rewriting it | Extract from working systems. Build the second and third recipe before freezing any interface. |
| Scope creep into a general framework | The four target systems are the scope. Anything else needs a real project asking for it. |
| Breaking friends' and colleagues' projects | Semantic versioning, a changelog, and pinned versions in dependants. |
| Heavy install (torch, Docling, CUDA) | Optional extras, and a light core that imports nothing heavy. |
| The package and the service drifting apart | The service consumes the package, with no copies. |
| Refactor regressions | Keep the existing tests green at every step. Run the eval suite before and after Phase 2 and Phase 3 and compare. |
| Overstating quality | Keep the habit of the existing documents: state what was measured, on what, and what was not tested. |

---

## 9. Document map

| File | What it covers |
|---|---|
| [README.md](../README.md) | One-line project description |
| [CHANGELOG.md](../CHANGELOG.md) | PDF detection and Docling conversion iterations 1 to 5 |
| [RETRIEVAL_LOG.md](../RETRIEVAL_LOG.md) | Hybrid retrieval implementation |
| [INFERENCE_LOG.md](../INFERENCE_LOG.md) | Model selection, vLLM deployment, inference layer |
| [tools.md](../tools.md) | The hierarchical agent and tool layer, capability status, limitations, next steps |
| [evals.md](../evals.md) | Evaluation method, results, what the evaluations changed |
| [production.md](../production.md) | Production hardening decisions, configuration, runbook, gaps |
| [docs/ingestion-map.html](ingestion-map.html) | Ingestion pipeline map |
