# Documentation index

Where to find things. Documents are grouped by what you are trying to do. Dates in logs are when they were written; the guides and the roadmap are kept current.

## Start here

| Document | What it is |
|---|---|
| [README](../README.md) | What Draco is, architecture, install, quick start, HTTP API, status and known limitations |
| [PROJECT_OVERVIEW](PROJECT_OVERVIEW.md) | Purpose, what is built, design principles, and the package plan |
| [Roadmap](architecture/roadmap.md) | What is done, partial and planned, in priority order |

## Using the framework (`src/draco`)

| Document | What it is |
|---|---|
| [api-guide](api-guide.md) | The framework API: harness, contracts, agents, tools, security model, adapters, observability, plus the starter app's HTTP additions |
| [extension-guide](extension-guide.md) | Plugins, hooks, tools, providers and stores: extending Draco without editing its source |
| [DRACO_CHANGELOG](DRACO_CHANGELOG.md) | Framework changes by version, with what is verified and what is still open |
| [`examples/`](../examples) | Runnable examples (`PYTHONPATH=src:. python -m examples.a_simple`); they also run as tests |

## Running the starter app (`app/`)

| Document | What it is |
|---|---|
| [production.md](../production.md) | Configuration reference, operations runbook, alerts, backups and the **known gaps** list |
| [tools.md](../tools.md) | The hierarchical agent and tool layer: capabilities, limits, risks and next steps |
| [DOCUMENT_OWNERSHIP](DOCUMENT_OWNERSHIP.md) | Per-user document ownership and migrating legacy rows |
| [CHANGELOG](../CHANGELOG.md) | Starter-app and ingestion changes, iteration by iteration |
| [`deploy/`](../deploy) | vLLM, ingestion-worker and maintenance deployment files |

## Evidence: how it was measured

| Document | What it is |
|---|---|
| [evals.md](../evals.md) | RAG and agent evaluation: datasets, metrics, results and their limits |
| [RETRIEVAL_LOG](../RETRIEVAL_LOG.md) | How the retrieval pipeline was built and tuned |
| [INFERENCE_LOG](../INFERENCE_LOG.md) | The inference layer: vLLM serving, conversation memory, measured behaviour |
| [IMPLEMENTATION_LOG](IMPLEMENTATION_LOG.md) | Implementation log and verification matrix for the framework |
| [Current state](architecture/current-state.md) | The audit of the repository before the framework was added, and later updates |
| [Reference study](architecture/reference-lessons.md) | Lessons taken from other agent frameworks and the decisions made |
| [`ingestion-map.html`](ingestion-map.html) | Visual map of the ingestion pipeline |

## Security and operations

| Document | What it is |
|---|---|
| [SECURITY](../SECURITY.md) | How to report a vulnerability, scope, and known limitations |
| [PRODUCTION_AUDIT](../PRODUCTION_AUDIT.md) | A production bug audit and its findings |
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
