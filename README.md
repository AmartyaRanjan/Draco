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
