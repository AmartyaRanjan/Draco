# 🐉 Draco

> An execution harness for RAG and agent systems, with a production-oriented starter app.

[![CI](https://github.com/AmartyaRanjan/Draco/actions/workflows/ci.yml/badge.svg)](https://github.com/AmartyaRanjan/Draco/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Status: alpha](https://img.shields.io/badge/status-alpha-orange)

[📖 Documentation](docs/README.md) · [🤝 Contributing](CONTRIBUTING.md) · [🛡️ Security](SECURITY.md) · [🗺️ Roadmap](docs/architecture/roadmap.md) · [📜 Changelog](docs/DRACO_CHANGELOG.md)

---

## 🏗️ Overview

Draco is built in two distinct layers:

1. **`src/draco/` (The Framework)**  
   A lightweight Python package (hard dependency: `pydantic`) that runs any async workload under a unified lifecycle. It provides trusted identity, role-based authorization, default-deny tool policies, budgets, rate limits, deadlines, idempotency, redacted tracing, and durable state (in-memory, SQLite, or PostgreSQL). It includes an agent registry/router, a guarded tool runtime (native tools + optional MCP), resumable steps, and adapters for LangGraph, OpenAI-compatible, and Anthropic providers.

2. **`app/` (The Starter Service)**  
   A production-ready FastAPI RAG and agent service built *on top of* the framework. It features Redis, Qdrant, OpenSearch, and a vLLM server. Capabilities include multi-format ingestion (PDF, Markdown, text, HTML, DOCX), hybrid retrieval with reranking, grounded chat, a 21-capability LangGraph agent with human-approved ingestion, long-term memory, scoped API keys, Docker images, and a comprehensive evaluation suite.

> 💡 **Unified Execution:** The service is the framework's first consumer. Both chat paths (`/chat`, `/agent/chat`) and approval resumes run as Draco workloads through a single harness, maintaining one execution ID from the HTTP request down to the checkpoint.

---

## 🏛️ Architecture

```text
                         ┌────────────────────────────────────────────┐
 HTTP clients ── TLS ──► │ FastAPI platform (app/api, app/core)       │
                         │ auth + lockout · rate limits · admission   │
                         │ request caps · health · metrics · logs     │
                         └───────┬──────────────┬─────────────┬───────┘
                                 │              │             │
              ┌──────────────────┘              │             └───────────────┐
              ▼                                 ▼                             ▼
   ┌────────────────────┐          ┌────────────────────┐         ┌────────────────────┐
   │ Ingestion          │          │ Retrieval          │         │ Inference          │
   │ (app/ingestion,    │          │ (app/retrieval)    │         │ (app/inference)    │
   │  app/db)           │          │ dense + BM25 → RRF │         │ vLLM · Qwen3-4B-FP8│
   │ Docling · 5 formats│          │ → ColBERT rerank   │         │ streaming · memory │
   │ chunk hdrs · BGE-M3│          │ → thresholded      │         │ citations          │
   │ RQ worker + outbox │          │   selection        │         │ tool-policy checks │
   └────────┬───────────┘          └─────────┬──────────┘         └─────────┬──────────┘
            │                                │                              │
            ▼                                └──────────────┬───────────────┘
   ┌────────────────────┐                                   ▼
   │ Stores             │                        ┌────────────────────┐
   │ Qdrant (dense)     │                        │ Agent              │
   │ OpenSearch (BM25)  │                        │ (app/agent)        │
   │ SQLite registry    │                        │ LangGraph supervisor│
   │ (owner-scoped)     │                        │ 21 capabilities    │
   │ Redis (RQ, limits) │                        │ evidence + budgets │
   └────────────────────┘                        └─────────┬──────────┘
                                                           │
                         ┌─────────────────────────────────┘
                         ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ draco harness (src/draco)                                   │
   │ core: context · results · typed errors · budgets · policy   │
   │ harness: runner · hooks · guardrails · idempotency · resume │
   │ state: SQLite/Postgres · retention · encryption · limits    │
   │ agents: registry · router (rules/LLM) · handoffs            │
   │ tools: ToolRuntime over CapabilityRegistry · MCP adapter    │
   │ adapters: retrieval/RAG/agent/graph workloads               │
   │ observability: redacted tracer · JSONL/OTel exporters       │
   └─────────────────────────────────────────────────────────────┘