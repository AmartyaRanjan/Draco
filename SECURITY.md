# Security Policy

Draco enforces identity, authorization, tenant isolation and tool policy in code. We take reports of weaknesses seriously and appreciate responsible disclosure.

## Supported versions

Draco has not made a tagged release yet. Security fixes are made on `main`, and only the current `main` is supported.

| Version | Supported |
|---|---|
| `main` (unreleased, 0.x) | Yes |
| Older commits and forks | No |

## Reporting a vulnerability

**Please do not report security problems in a public issue, pull request or discussion.**

1. **If GitHub's "Report a vulnerability" button is available** on the repository's **Security** tab, use it. It creates a private advisory that only you and the maintainers can see.
2. **Otherwise**, open an issue titled **"Security contact request"** that says only that you have a security report and, if you wish, how you would like to be contacted. **Put no technical details in the issue.** A maintainer will arrange a private channel to receive them.

Please include, once a private channel exists:

- what the problem is and which component is affected (for example `draco.security` or `draco.state.sqlite`),
- the commit or version you tested,
- steps to reproduce or a proof of concept,
- the impact as you understand it (for example "a user of tenant A can read tenant B's documents"),
- whether it is already public.

### What to expect

This is a volunteer-maintained project, so these are targets, not guarantees:

- acknowledgement within about **7 days**,
- an initial assessment (accepted, needs more information, or declined with a reason) within about **14 days**,
- a fix or mitigation for confirmed issues as quickly as the severity warrants, and credit in the release notes and advisory if you want it.

Please give us a reasonable chance to fix an issue before disclosing it publicly. We will keep you informed and will not take legal action against good-faith research that follows this policy.

## Scope

**In scope**

- Authorization or tenant-isolation bypass in `draco.security`, `draco.harness`, `draco.tools` or `draco.adapters.retrieval`.
- Policy bypass: a tool running without an explicit allow, a child execution widening its parent's `Policy`, an approval-required tool running without approval.
- Leaks of secrets, API keys, provider response bodies or other tenants' identifiers in traces, persisted events, logs or error messages.
- Injection in the SQL stores (`draco.state`), and resource-exhaustion paths in the harness.

Problems in the RAG service (API-key auth, uploads, web fetching, ingestion, document ownership) belong to [draco-rag](https://github.com/AmartyaRanjan/draco-rag); report them there the same way.
- Vulnerable dependencies where the vulnerable code is reachable from Draco.

**Out of scope**

- Prompt injection that merely changes what a model says. It is mitigated, not solved, and the limits are documented. A prompt injection that **bypasses authorization, policy or tenant isolation** is in scope.
- Findings that need an already-compromised host, a malicious operator, or access to the process environment.
- Missing hardening that the documentation assigns to the deployment: TLS termination, WAF and DDoS protection, network egress filtering, and exposing the application port directly.
- Denial of service by sheer volume against an unprotected deployment.
- Reports produced by an automated scanner with no demonstrated impact.

## Known limitations

These are documented and are not, by themselves, vulnerabilities. Read them before deploying:

- [docs/api-guide.md](docs/api-guide.md) (Security) lists what Draco enforces, what you must configure, and known gaps.
- **Committed runtime files in earlier Git history.** They were removed from all branches on 2026-11-27, but copies made before then (clones, forks, GitHub's cached pull-request refs) may remain. Treat anything that was ever committed here as exposed and rotate the credentials involved. [docs/HISTORY_PURGE.md](docs/HISTORY_PURGE.md) has the details, and `scripts/check_history_exposure.sh` checks a clone.
- Draco supports tenant isolation, but you must supply authentication and configure authorization (see the API guide).

## Handling secrets when contributing

- Never commit `.env` files, API keys, databases, uploads or evaluation output. CI rejects tracked runtime data.
- Do not paste real keys, tokens or customer documents into issues, logs or test fixtures.
- If you accidentally commit a secret, tell a maintainer immediately and rotate the secret. Removing it in a later commit does not remove it from history.

## Dependency and supply-chain checks

CI audits the framework's dependencies with `pip-audit` on every pull request, and a failing audit fails the build. See [CONTRIBUTING.md](CONTRIBUTING.md#10-dependencies).
