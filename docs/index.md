# Draco

Draco is a small Python execution harness for agent and RAG workloads. Anything shaped like
`async def workload(ctx, input)` can be registered and run under one lifecycle: trusted identity and
authorization, default-deny tool policy, budgets and rate limits, deadlines and cancellation, idempotency,
redacted tracing, and durable state with resumable steps. The native framework depends only on `pydantic`.

The repository also contains a starter application built on it: a FastAPI RAG and agent service
(hybrid retrieval with reranking, a tool-using LangGraph agent, human-approved ingestion, an evaluation suite).

## Start here

- **Run something:** `pip install -e .` then `PYTHONPATH=src:. python -m examples.a_simple`
- **Use the framework:** [API guide](api-guide.md)
- **Extend it:** [Extending Draco](extension-guide.md)
- **See what is done and what is not:** [Roadmap and status](architecture/roadmap.md)

## Status

Early (version 0.1.0). The roadmap page separates *done*, *partial* and *planned*. The starter application is
hardened for a single-tenant trust domain, not multi-tenant use. Evaluation results, the production runbook and
the agent design notes live in the repository root: see
[evals.md](https://github.com/AmartyaRanjan/Draco/blob/main/evals.md),
[production.md](https://github.com/AmartyaRanjan/Draco/blob/main/production.md) and
[tools.md](https://github.com/AmartyaRanjan/Draco/blob/main/tools.md).

## Source

[github.com/AmartyaRanjan/Draco](https://github.com/AmartyaRanjan/Draco)
