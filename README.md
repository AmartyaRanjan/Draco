<<<<<<< HEAD
# Draco — Production-Oriented Agentic RAG

Draco is an agentic retrieval-augmented generation system designed around hierarchical orchestration, controlled capability execution, evidence-grounded responses, and explicit context-budget management.

## Why Draco?

Conventional RAG pipelines execute a mostly fixed sequence of retrieval and generation steps. Draco explores how a supervisor-driven agent can select capabilities, manage evidence, control reasoning, and produce verifiable responses while keeping execution bounded and observable.

## Architecture

[Insert architecture diagram]

Core subsystems:
- Agent orchestration and conditional routing
- Typed capability registry and execution policies
- Hybrid dense and lexical retrieval with reranking
- Evidence provenance and citation validation
- Context-budget management and evidence summarization
- Model serving and reasoning-mode adapters
- Evaluation, tracing, and failure handling

## Features and Implementation Status

| Capability | Status | Verification |
|---|---|---|
| Hierarchical agent orchestration | [Status] | [Tests or source link] |
| Hybrid retrieval and reranking | [Status] | [Tests or evaluation] |
| Tool permissions and execution limits | [Status] | [Security tests] |
| Context compaction | [Status] | [Long-context tests] |
| Persistent checkpoint recovery | [Status] | [Recovery tests] |
| Tenant isolation | [Status] | [Authorization tests] |
| Live external integrations | [Status] | [Integration tests] |

## Quick Start

Document prerequisites, environment configuration, database initialization, model-server setup, API startup, and a working example request.

## Example Execution Trace

Show the user's query, routing decision, capabilities invoked, retrieval results, evidence identifiers, context-management actions, model response, citation validation, and final result.

## Evaluation

Document the dataset, metrics, baselines, model versions, hardware, reproducibility instructions, and results. Include failures and limitations.

## Deployment and Security

Link to the deployment runbook, threat model, configuration reference, backup and recovery instructions, and security reporting policy.

## Known Limitations

Clearly identify untested integrations, deployment constraints, evaluation gaps, and unsupported configurations.

## Roadmap

List planned work separately from implemented features.

## Contributing and License

Provide development setup, testing requirements, contribution guidelines, and the project's license.
=======
# Draco
This repository is a project on Agentic RAG, this is my first hands on work with agents in the loops, tools and agent evaluations

## Draco: execution harness for this engine

The `app/` service (ingestion, hybrid retrieval, vLLM inference, LangGraph agent with 21 capabilities) is unchanged. `src/draco/` adds a reusable layer around it: a harness with execution identity, policy, budgets, cancellation, tracing and persistence; a plugin/service host; an agent router; a guarded tool runtime (native tools, optional MCP); resumable steps.

```
pip install -e ".[dev]"
PYTHONPATH=src:. python -m examples.a_simple        # examples a..f in examples/
python -m pytest tests/draco -q                      # new tests
python -m unittest discover -s tests -t .            # original suite, unchanged
```
Docs: [API guide](docs/api-guide.md) · [extension guide](docs/extension-guide.md) · [current state](docs/architecture/current-state.md) · [roadmap](docs/architecture/roadmap.md) · [reference study](docs/architecture/reference-lessons.md) · [implementation log and verification matrix](docs/IMPLEMENTATION_LOG.md).
>>>>>>> draco-framework
