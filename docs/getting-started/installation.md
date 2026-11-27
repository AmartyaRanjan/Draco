# Installation

Draco needs **Python 3.11 or newer**. The package on PyPI is named `draco-harness`; the module you import is `draco`.

```bash
pip install draco-harness
```

That installs the framework and its only hard dependency, `pydantic`. Everything else is an optional extra, imported only when you use it.

!!! warning "Use the right name"
    `pip install draco` installs a different, unrelated project. Draco's package name is **`draco-harness`**.

## Optional extras

Install an extra with square brackets, for example `pip install "draco-harness[providers,agents]"`.

| Extra | Adds | You need it for |
|---|---|---|
| `providers` | `httpx` | The OpenAI-compatible and Anthropic model providers |
| `agents` | `langgraph`, `langchain-core`, `langgraph-checkpoint-sqlite` | Running LangGraph graphs as workloads, with durable checkpoints |
| `postgres` | `psycopg[binary]` | `PostgresStore`, for shared durable state |
| `crypto` | `cryptography` | Encrypting persisted state at rest |
| `mcp` | `mcp` | Registering tools from an MCP server |
| `otel` | `opentelemetry-api`, `opentelemetry-sdk` | Exporting traces as OpenTelemetry spans |
| `dev` | `pytest`, `ruff`, `mypy`, `pip-audit` | Working on Draco itself |
| `docs` | `mkdocs`, `mkdocs-material`, `mkdocstrings` | Building this documentation |

SQLite persistence, the tracer, the tool runtime, the router and the plugin host need no extra.

## Check it works

<!-- doctest -->
```python
import draco
print(draco.__version__)
```

<!-- output -->
```text
0.1.0
```

## Install from source

To work on Draco, or to try an unreleased change:

```bash
git clone https://github.com/AmartyaRanjan/Draco.git
cd Draco
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,agents,providers]"
python -m pytest tests/draco -q
```

The [contributing guide](https://github.com/AmartyaRanjan/Draco/blob/main/CONTRIBUTING.md) describes the checks a change has to pass.

## Next

Run something: the [Quickstart](quickstart.md).
