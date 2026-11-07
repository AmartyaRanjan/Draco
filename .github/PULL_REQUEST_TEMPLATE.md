## What and why

<!-- What does this change, and why is it needed? Link the issue: "Closes #123". -->

## Type of change

- [ ] Bug fix (with a regression test that fails without the fix)
- [ ] New feature or extension point
- [ ] Behaviour change or breaking change (described below)
- [ ] Security-sensitive change (authorization, tenant isolation, policy, auth)
- [ ] Documentation, tests, CI or dependency update only

## How I tested it

<!-- Commands you ran and their results. For a bug fix, say that you checked the test fails without the fix. -->

## What I did not verify

<!-- Be explicit: not run against a live service, not measured, only tested on one Python version, etc. -->

## Behaviour changes, configuration and migrations

<!-- New or renamed settings, changed defaults, a re-index or data migration, anything an operator must do. Write "None" if none. -->

## Checklist

- [ ] `python -m pytest tests/draco -q`, `ruff check src tests/draco examples` and `mypy src/draco` pass
- [ ] `python -m unittest discover -s tests -t .` passes (if `app/` or its adapters changed)
- [ ] If I touched framework code, I ran the lean subset (`pytest tests/draco/unit tests/draco/reliability`) in an environment with only `.[dev,providers]`
- [ ] New or changed behaviour has tests; no test was weakened or deleted to pass
- [ ] Docs updated (API guide, production.md, roadmap as needed) and an entry added to the changelog
- [ ] No secrets, `.env` files, databases, uploads or evaluation output are committed
- [ ] No permission is widened and no new global request state is introduced (or I explain why above)
- [ ] I have read [CONTRIBUTING.md](../CONTRIBUTING.md)
