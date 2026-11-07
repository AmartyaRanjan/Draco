# Getting support

Draco is maintained by volunteers. This page says where to take what, so your question reaches the right place.

| I want to... | Go to |
|---|---|
| Ask how to use Draco, share an idea, or discuss a design | [GitHub Discussions](https://github.com/AmartyaRanjan/Draco/discussions) |
| Report a bug | [Open an issue](https://github.com/AmartyaRanjan/Draco/issues/new/choose) with the bug-report form |
| Request a feature or an extension point | [Open an issue](https://github.com/AmartyaRanjan/Draco/issues/new/choose) with the feature-request form |
| Report a security problem | **Not** an issue. Follow [SECURITY.md](SECURITY.md) |
| Report a conduct problem | Follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) |
| Contribute code or docs | [CONTRIBUTING.md](CONTRIBUTING.md) |

## Before you ask

Most questions are answered in the docs. Start with the [documentation index](docs/README.md):

- [README](README.md): what Draco is, install, quick start, HTTP API.
- [docs/api-guide.md](docs/api-guide.md): the framework's API and security model.
- [docs/extension-guide.md](docs/extension-guide.md): plugins, hooks, tools, adapters.
- [production.md](production.md): configuration, operations runbook, known gaps.
- [examples/](examples/): runnable examples (`PYTHONPATH=src:. python -m examples.a_simple`).

## A good bug report

It saves everyone time if it includes:

- what you did, what you expected and what happened,
- the smallest code or request that reproduces it,
- your Python version, operating system and how you installed Draco (`pip install -e ".[...]"`, which extras),
- relevant log or trace output with **secrets, tokens and private documents removed**,
- the commit you are on (`git rev-parse --short HEAD`).

## Response expectations

There is no service-level agreement. Maintainers answer when they can. Clear, reproducible reports with a focused scope get answered fastest. Please be patient and kind; the [Code of Conduct](CODE_OF_CONDUCT.md) applies everywhere.
