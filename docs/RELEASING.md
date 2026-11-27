# Releasing to PyPI

Package name: **`draco-harness`** (import name `draco`). Releases use PyPI *trusted publishing*, so no token is
stored in GitHub. The `release` workflow builds and checks the package on every run, and publishes only when a
person publishes a GitHub release (or starts the workflow manually for TestPyPI).

## One-time setup (done by the project owner, on the websites)

1. Create accounts on [pypi.org](https://pypi.org) and [test.pypi.org](https://test.pypi.org), with two-factor
   authentication on both.
2. On each site, add a **pending publisher** (Account settings > Publishing > "Add a new pending publisher"):

   | Field | Value |
   |---|---|
   | PyPI project name | `draco-harness` |
   | Owner | `AmartyaRanjan` |
   | Repository | `Draco` |
   | Workflow filename | `release.yml` |
   | Environment | `pypi` on pypi.org, `testpypi` on test.pypi.org |

3. In the GitHub repository settings, create two environments named `pypi` and `testpypi`. Add yourself as a
   required reviewer on `pypi` so a release cannot publish without your approval.

## Releasing

1. Update `version` in `pyproject.toml` and add a dated entry to [DRACO_CHANGELOG.md](DRACO_CHANGELOG.md). Merge it.
2. **Dry run:** Actions > release > Run workflow > `testpypi`. Then, in a clean virtualenv:
   `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ draco-harness`
   and run `python -c "import draco; print(draco.__file__)"`.
3. Create a GitHub release whose tag is `v<version>` (for example `v0.1.0`). The workflow checks that the tag matches
   `pyproject.toml`, then waits for your approval on the `pypi` environment and publishes.

A version published to PyPI can be yanked but never deleted or re-uploaded, so always do the TestPyPI dry run first.
