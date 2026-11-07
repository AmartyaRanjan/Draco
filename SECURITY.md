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

- what the problem is and which component is affected (`src/draco/...` or `app/...`),
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

- Authorization or tenant-isolation bypass in `draco.security`, `draco.harness`, `draco.tools` or the document-ownership checks in `app/`.
- Policy bypass: a tool running without an explicit allow, a child execution widening its parent's `Policy`, an approval-required tool running without approval.
- Leaks of secrets, API keys, provider response bodies or other tenants' identifiers in traces, persisted events, logs or error messages.
- Authentication problems in the starter app: API-key scope or expiry bypass, brute-force lockout bypass, rate-limit bypass.
- Injection in the SQL stores, unsafe handling of uploaded files (PDF, DOCX, HTML, Markdown), SSRF in the web-fetch capability, and zip-bomb or resource-exhaustion paths in ingestion.
- Vulnerable dependencies where the vulnerable code is reachable from Draco.

**Out of scope**

- Prompt injection that merely changes what a model says. It is mitigated, not solved, and the limits are documented. A prompt injection that **bypasses authorization, policy or tenant isolation** is in scope.
- Findings that need an already-compromised host, a malicious operator, or access to the process environment.
- Missing hardening that the documentation assigns to the deployment: TLS termination, WAF and DDoS protection, network egress filtering, and exposing the application port directly. See [production.md](production.md).
- Denial of service by sheer volume against an unprotected deployment.
- Reports produced by an automated scanner with no demonstrated impact.

## Known limitations

These are documented and are not, by themselves, vulnerabilities. Read them before deploying:

- [production.md](production.md) section 6 lists known gaps (single-host data stores, static API-key list, state not encrypted in the starter app, unmeasured retrieval changes, and more).
- [docs/api-guide.md](docs/api-guide.md) (Security) lists what Draco enforces, what you must configure, and known gaps.
- [PRODUCTION_AUDIT.md](PRODUCTION_AUDIT.md) records an earlier audit and its findings.
- **Committed runtime files in Git history.** Earlier commits contained runtime files. Treat anything that was ever in this repository's history as exposed and rotate the credentials involved. [docs/HISTORY_PURGE.md](docs/HISTORY_PURGE.md) describes the purge, and `scripts/check_history_exposure.sh` checks exposure.
- Draco is hardened for a **single-tenant trust domain** in the starter app. The framework supports tenant isolation, but you must supply authentication and configure authorization (see the API guide).

## Handling secrets when contributing

- Never commit `.env` files, API keys, databases, uploads or evaluation output. CI rejects tracked runtime data.
- Do not paste real keys, tokens or customer documents into issues, logs or test fixtures.
- If you accidentally commit a secret, tell a maintainer immediately and rotate the secret. Removing it in a later commit does not remove it from history.

## Dependency and supply-chain checks

CI audits the framework's dependencies and the pinned `requirements.txt` with `pip-audit` on every pull request, and a failing audit fails the build. See [CONTRIBUTING.md](CONTRIBUTING.md#10-dependencies).
