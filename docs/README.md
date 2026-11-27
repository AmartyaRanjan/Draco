# Documentation index

**Start at the documentation site: <https://amartyaranjan.github.io/Draco/>.** It has the quickstart, tutorials, concepts, how-to guides and the Python API reference, and this folder is its source. Everything below is the same material grouped by file, for browsing on GitHub.

Where to find things. Documents are grouped by what you are trying to do. Dates in logs are when they were written; the guides and the roadmap are kept current.

## Start here

| Document | What it is |
|---|---|
| [README](../README.md) | What Draco is, architecture, install, quick start, status and known limitations |
| [PROJECT_OVERVIEW](PROJECT_OVERVIEW.md) | Purpose, what is built, design principles, and the package plan |
| [Roadmap](architecture/roadmap.md) | What is done, partial and planned, in priority order |

## Using the framework (`src/draco`)

| Document | What it is |
|---|---|
| [api-guide](api-guide.md) | The framework API: harness, contracts, agents, tools, security model, adapters, observability |
| [extension-guide](extension-guide.md) | Plugins, hooks, tools, providers and stores: extending Draco without editing its source |
| [DRACO_CHANGELOG](DRACO_CHANGELOG.md) | Framework changes by version, with what is verified and what is still open |
| [`examples/`](../examples) | Runnable examples (`PYTHONPATH=src:. python -m examples.a_simple`); they also run as tests |

## The RAG service

The FastAPI RAG and agent service that used to live here as the starter app (`app/`) is now its own repository, [draco-rag](https://github.com/AmartyaRanjan/draco-rag). Its runbook (`production.md`), agent notes (`tools.md`), evaluation (`evals.md`), retrieval and inference logs, production audit, document-ownership guide, ingestion map, changelog and deployment files moved with it.

## Project history

| Document | What it is |
|---|---|
| [IMPLEMENTATION_LOG](IMPLEMENTATION_LOG.md) | Implementation log and verification matrix for the framework |
| [Current state](architecture/current-state.md) | The audit of the repository before the framework was added, and later updates |
| [Reference study](architecture/reference-lessons.md) | Lessons taken from other agent frameworks and the decisions made |

## Security and operations

| Document | What it is |
|---|---|
| [SECURITY](../SECURITY.md) | How to report a vulnerability, scope, and known limitations |
| [HISTORY_PURGE](HISTORY_PURGE.md) | Runbook for removing runtime data from Git history |

## Contributing and community

| Document | What it is |
|---|---|
| [CONTRIBUTING](../CONTRIBUTING.md) | Setup, checks, code and test standards, commits and pull requests |
| [CODE_OF_CONDUCT](../CODE_OF_CONDUCT.md) | Expected behaviour and how conduct problems are handled |
| [SUPPORT](../SUPPORT.md) | Where to ask questions and report problems |
| [LICENSE](../LICENSE) | MIT License |

## Keeping this index honest

When you add a document, add a line here. If a document is superseded, say so in its first lines rather than leaving a statement that is no longer true; see [CONTRIBUTING](../CONTRIBUTING.md#9-documentation-and-claims).
