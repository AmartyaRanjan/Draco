# Draco Production Bug Audit

Audit date: 2026-10-22
Audited branch: `docs/readme-architecture`
Audited commit: `de499c3d967aee50fe2d544a9a40f01a0ff8bac6`
Remotes: `origin` points to `harshitsharmaaaa/Draco`; `upstream` points to `AmartyaRanjan/Draco`.

## 1. Executive summary

This audit covered the FastAPI application, RAG retrieval and indexing path, SQLite registries, RQ ingestion worker, agent capabilities, web fetcher, deployment configuration, CI, framework package, and tests. No application source, tests, dependencies, migrations, or deployment files were changed.

The repository is **not production-ready** at this commit. Four high-priority issues can expose internal network services or private repository data, or take down the ingestion worker. Six medium-priority issues can cause unauthorized resource consumption, incorrect retrieval results, weakened abuse controls, or stale data ingestion.

| Severity | Findings | Confidence | Notes |
|---|---:|---|---|
| P0 | 0 | — | No catastrophic compromise or widespread destruction was demonstrated. |
| P1 | 4 | High | SSRF configuration bypass, OpenSearch TLS verification default, unsandboxed PDF parsing, and sensitive runtime data retained in Git history. |
| P2 | 6 | High | Failed documents visible to retrieval, unbounded owner fan-out, uncancellable blocking work, limiter failover, proxy trust, and stale approvals. |
| P3 | 0 | — | Lower-impact defects were not promoted to major findings. |

Verification was materially limited by the local environment: the Python environment is missing `langchain_core`, `sentence_transformers`, `prometheus_client`, `opensearchpy`, and the `draco` editable package; `croniter` is also missing despite being pinned in `requirements.txt`. Redis, Qdrant, OpenSearch, vLLM, and Docling-backed worker execution were not available. The Docker daemon was not running.

## 2. Production readiness verdict

**Not production-ready.**

At minimum, production deployment must block private-host web fetching, enforce TLS certificate verification for HTTPS OpenSearch, isolate and resource-limit PDF conversion, and handle the sensitive Git history. The retrieval visibility and capacity issues must also be fixed before serving multi-document or multi-user workloads. Existing unit tests cover several historical fixes, but they do not detect these paths; the full application and framework verification suites could not pass in this environment.

## 3. Findings summary

| ID | Severity | Confidence | Title | Affected module | Reproduction status | Recommended priority |
|---|---|---|---|---|---|---|
| DRACO-001 | P1 | High | Production can enable private-host web fetching and SSRF | `app/agent/config.py`, `app/agent/web.py` | PASS | Immediate |
| DRACO-002 | P1 | High | HTTPS OpenSearch defaults to disabled certificate verification | `app/db/opensearch.py` | Code-confirmed; dynamic check blocked | Immediate |
| DRACO-003 | P1 | High | Untrusted PDFs are parsed without a sandbox or per-file resource limits | `app/ingestion/worker.py`, `app/ingestion/convert.py` | Control gap confirmed; exploit run BLOCKED | Immediate |
| DRACO-004 | P2 | High | Failed and partial documents remain eligible for retrieval | `app/db/registry.py`, `app/retrieval/service.py`, `app/ingestion/pipeline.py` | PASS | Before production corpus use |
| DRACO-005 | P2 | High | Default retrieval loads an unbounded owner document ID list | `app/db/registry.py`, `app/retrieval/retrievers.py` | PASS | Before scale-out |
| DRACO-006 | P2 | High | Retrieval timeouts abandon, but do not stop, blocking worker threads | `app/retrieval/retrievers.py` | PASS for runtime behavior; app import partially blocked | Before load |
| DRACO-007 | P2 | High | Redis outage multiplies rate limits per process | `app/core/ratelimit.py` | PASS | Before multi-worker deployment |
| DRACO-008 | P2 | High | `TRUSTED_PROXY_HOPS` permits client-supplied identity when misconfigured | `app/core/ratelimit.py`, `app/core/http_settings.py` | PASS | Before direct exposure |
| DRACO-009 | P2 | High | Approval remains valid after the source proposal expires | `app/agent/store.py` | PASS | Before web ingestion |
| DRACO-010 | P1 | High | Private runtime documents and database history remain in Git objects | Git history / `app/storage` | PASS | Immediate security response |

## 4. Detailed findings

### DRACO-001 — Production can enable private-host web fetching and SSRF

Severity: **P1**
Confidence: **High / confirmed**
References: `app/agent/config.py:45,56-67,98,105`; `app/main.py:24-28`; `app/agent/web.py:99-125`

#### Preconditions and trigger

An operator sets `APP_ENV=production` and `WEB_ALLOW_PRIVATE_HOSTS=true`. An authenticated user then invokes an agent path that permits web access and supplies an allowed URL or redirect chain targeting an internal address such as `127.0.0.1`, a private RFC1918 address, or a cloud metadata endpoint.

#### Root cause and failure behavior

`WEB_ALLOW_PRIVATE_HOSTS` is documented as “tests/dev only”, but `AgentSettings.validate()` never rejects it in production. Application startup validates platform and inference settings only; it never validates agent settings. `WebFetcher._validated_ip()` skips the public-address check when this flag is true, and `fetch()` connects directly to the resolved address with `trust_env=False`.

The local reproduction set the production environment and flag, loaded settings, and passed a resolver returning `127.0.0.1`:

```powershell
@'
import os, asyncio
from types import SimpleNamespace
from urllib.parse import urlsplit
from app.agent.config import get_agent_settings
os.environ["APP_ENV"] = "production"
os.environ["WEB_ALLOW_PRIVATE_HOSTS"] = "true"
get_agent_settings.cache_clear()
print(get_agent_settings().validate().web_allow_private_hosts)
from app.agent.web import WebFetcher
async def main():
    f = WebFetcher(SimpleNamespace(web_allow_private_hosts=True),
                   resolver=lambda h, p: asyncio.sleep(0, result=["127.0.0.1"]))
    print(await f._validated_ip(urlsplit("http://internal.example:80/")))
asyncio.run(main())
'@ | python -
```

Observed output:

```text
agent_settings_validated True True
private_ip_accepted 127.0.0.1
```

#### Impact

An authenticated attacker with web capability can use the application as an internal HTTP client. Depending on network placement, this can read cloud metadata, Redis, OpenSearch, Qdrant, internal admin endpoints, or other services reachable from the application host. The response is returned to the agent path and may be exposed in the conversation or tool result.

#### Minimal fix and regression test

Fail startup when `APP_ENV=production` and `WEB_ALLOW_PRIVATE_HOSTS` is true. Prefer an explicit destination allowlist and network egress policy for any production web-fetch capability. Add a startup validation test and a fetch test asserting private, loopback, link-local, and metadata addresses are rejected in production regardless of environment defaults.

#### History overlap

No local commit was found that enforces this production invariant. The existing comments and tests only establish the intended development-only use.

### DRACO-002 — HTTPS OpenSearch defaults to disabled certificate verification

Severity: **P1**
Confidence: **High / code-confirmed**
References: `app/db/opensearch.py:18-45`

#### Preconditions and trigger

`OPENSEARCH_URL` is configured with an `https://` URL and `OPENSEARCH_VERIFY_CERTS` is omitted or set to anything other than the literal case-insensitive string `true`.

#### Root cause and failure behavior

The client enables SSL based on the URL, but independently defaults `verify_certs` to false. Hostname assertion and certificate warnings are also disabled because they are tied to the same false value. There is no production startup validation requiring certificate verification for HTTPS.

The exact constructor path is:

```python
use_ssl=url.startswith("https://")
verify_certs=os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true"
ssl_assert_hostname=verify_certs
```

A dynamic constructor harness could not import the module because `opensearchpy` is absent in this environment; the unsafe default is deterministic from the source and does not depend on a live cluster.

The attempted dependency check was:

```text
python -c "import opensearchpy"
ModuleNotFoundError: No module named 'opensearchpy'
```

#### Impact

On a network where traffic can be intercepted, an attacker can impersonate OpenSearch, observe credentials and indexed document contents, or return manipulated search results. This is especially dangerous because the application sends optional basic-auth credentials to the configured endpoint.

#### Minimal fix and regression test

Default certificate verification to true for HTTPS, require an explicit CA bundle where needed, and fail production startup for `https://` plus disabled verification. Add a constructor test for omitted, false, and true settings and a startup validation test.

#### History overlap

No local fix for this default was found.

### DRACO-003 — Untrusted PDFs are parsed without a sandbox or per-file resource limits

Severity: **P1**
Confidence: **High for the missing control; exploit impact unverified locally**
References: `app/ingestion/worker.py:14-37,60-88`; `app/ingestion/convert.py:100-228`; `app/ingestion/dispatch.py:57-64`; `production.md:205-206`

#### Preconditions and trigger

An authenticated user uploads a syntactically valid but computationally expensive, decompression-bomb, malformed, or memory-intensive PDF. The RQ worker processes it.

#### Root cause and failure behavior

`process_document()` imports and calls `convert_pdf()` directly in the worker process. Docling conversion and OCR execute without a subprocess/container boundary, memory quota, CPU quota, file descriptor quota, or per-file page/object limit. The only stated bound is the RQ `job_timeout=3600`, which allows one job to occupy a worker for up to an hour and does not cap memory or terminate all native work safely.

The repository’s own production runbook confirms this gap: “Uploaded PDFs are parsed by Docling with no sandbox or per-file resource limit beyond a 3600 s RQ timeout.” Docling is not installed locally, so a malicious-PDF execution was not attempted.

#### Impact

A crafted upload can exhaust worker memory or CPU, crash/restart the worker, monopolize the ingestion queue, and potentially affect a shared host where the worker and other services run. The 100 MB upload limit does not prevent highly expanding or structurally expensive PDFs.

#### Minimal fix and regression test

Run conversion in an isolated worker container or subprocess with cgroup memory/CPU limits, a hard wall-clock timeout, temporary-storage quota, file descriptor quota, and PDF structural/page limits. Add a resource-limited fixture test and verify that timeout/kill transitions the registry to `failed` and releases the job lease.

#### History overlap

This is documented as known gap 7 in `production.md`, but it remains present at the audited commit.

### DRACO-004 — Failed and partial documents remain eligible for retrieval

Severity: **P2**
Confidence: **High / confirmed**
References: `app/db/registry.py:222-245,406-428`; `app/retrieval/service.py:33-37,87-98`; `app/retrieval/retrievers.py:38-75`; `app/ingestion/pipeline.py:59-87`; `app/ingestion/worker.py:70-88`

#### Preconditions and trigger

A document has status `uploaded`, `processing`, or `failed`, or a new version writes one or more batches before a later embedding/index write fails. A user queries without an explicit document list.

#### Root cause and failure behavior

The owner filter selects every row owned by the principal and does not include `status='completed'`. Retrieval passes those IDs to Qdrant and OpenSearch. The ingestion pipeline writes each batch to both indexes and only deletes stale versions after all batches complete. If a later batch fails, partial new-version chunks and any old-version chunks remain indexed, while the registry status becomes `failed`.

The minimal SQLite reproduction created one owner’s documents with `failed`, `processing`, and `completed` statuses:

```powershell
@'
import tempfile
from pathlib import Path
import app.db.registry as r
with tempfile.TemporaryDirectory() as d:
    r.DB_PATH = Path(d) / "documents.db"
    r.init_db()
    for doc_id, status in (("failed", "failed"), ("processing", "processing"), ("done", "completed")):
        r.create_document(doc_id, "x.pdf", "/tmp/x.pdf", "alice")
        r.update_status(doc_id, status)
    print(r.list_document_ids_for_owner("alice"))
'@ | python -
```

Observed output:

```text
owner_ids ['done', 'failed', 'processing']
```

#### Impact

Users can receive evidence from failed or incomplete uploads. During reindex failures, responses can mix old and partial new content, producing incorrect answers and making recovery state-dependent. The issue is owner-scoped, so this finding is a correctness and data-integrity problem rather than a cross-user authorization bypass.

#### Minimal fix and regression test

Restrict retrieval eligibility to a published/completed index state. Use an explicit document version publication record or an index alias/visibility marker, and publish only after both backends have completed. Add tests asserting that non-completed documents never reach either retriever and that a failed reindex leaves exactly the previous published version visible.

#### History overlap

`tests/test_pipeline.py` covers preservation of the previous version after a failed write, but no current test checks registry status filtering during retrieval.

### DRACO-005 — Default retrieval loads an unbounded owner document ID list

Severity: **P2**
Confidence: **High / confirmed**
References: `app/db/registry.py:222-235`; `app/retrieval/service.py:33-37,91`; `app/retrieval/retrievers.py:40-45,62-73`; `app/api/query.py:18`

#### Preconditions and trigger

An authenticated principal owns many documents and submits a normal query without `document_ids`. The explicit request field is capped at 50, but the default owner lookup has no limit.

#### Root cause and failure behavior

`list_document_ids_for_owner()` returns every matching document ID. The retrieval service passes the full list to a Qdrant `MatchAny` filter and an OpenSearch `terms` filter. No page size, hard ceiling, or server-side owner field filter is applied.

A temporary SQLite reproduction inserted 1,500 documents and observed:

```powershell
@'
import tempfile
from pathlib import Path
import app.db.registry as r
with tempfile.TemporaryDirectory() as d:
    r.DB_PATH = Path(d) / "documents.db"
    r.init_db()
    for i in range(1500):
        r.create_document("d" + str(i), "x.pdf", "/tmp/x.pdf", "alice")
    print(len(r.list_document_ids_for_owner("alice")))
'@ | python -
```

Observed output:

```text
returned_ids 1500 elapsed_ms 4.54 first_last d0 d999
```

The local measurement confirms the unbounded application behavior; backend failure thresholds and latency were not measured without live services.

#### Impact

Large tenants cause request bodies, filter objects, query parsing, memory use, and latency to grow with total corpus size. OpenSearch terms limits or Qdrant request limits can turn this into errors; repeated requests can exhaust API, backend, or network capacity.

#### Minimal fix and regression test

Use a backend-side owner/tenant filter on indexed metadata, or maintain a bounded published-document selector. Enforce an operational maximum and return a clear error or require an explicit scope when exceeded. Add a capacity test with realistic tenant sizes and assert the query payload stays bounded.

#### History overlap

No local fix was found. The API-level 50-ID cap does not cover the default owner lookup.

### DRACO-006 — Retrieval timeouts abandon, but do not stop, blocking worker threads

Severity: **P2**
Confidence: **High / runtime behavior confirmed**
References: `app/retrieval/retrievers.py:29-31,50-55,77-80`; `app/retrieval/service.py:109-113`; `docs/IMPLEMENTATION_LOG.md:61-63`

#### Preconditions and trigger

Qdrant, OpenSearch, embedding, or reranking becomes slow or unresponsive while request volume continues. The request timeout expires.

#### Root cause and failure behavior

Blocking functions are submitted through `asyncio.to_thread()`. `asyncio.wait_for()` cancels the awaiting future, but Python cannot interrupt the already-running worker thread. Dense query embedding is not wrapped in a timeout at all; Qdrant and OpenSearch calls are wrapped, but their underlying threads continue. The global GPU lock can remain held by slow embedding/reranking work.

The minimal runtime reproduction demonstrated the relevant cancellation behavior:

```powershell
@'
import asyncio, threading, time
finished = threading.Event()
def blocking():
    time.sleep(0.2)
    finished.set()
async def main():
    try:
        await asyncio.wait_for(asyncio.to_thread(blocking), 0.01)
    except asyncio.TimeoutError:
        print("TimeoutError", finished.is_set())
    await asyncio.sleep(0.25)
    print(finished.is_set())
asyncio.run(main())
'@ | python -
```

Observed output:

```text
wait_for_result TimeoutError worker_finished_immediately False
worker_finished_later True
```

The repository’s implementation log explicitly records that `to_thread` work is “abandoned, not interrupted.”

#### Impact

Repeated backend stalls can fill the default executor, queue more work than the service can execute, retain GPU/CPU resources after clients receive errors, and turn a transient dependency slowdown into API-wide latency or outage.

#### Minimal fix and regression test

Prefer truly asynchronous clients with request-level cancellation. For unavoidable blocking or native work, use a bounded executor or isolated worker process with admission control and a killable timeout. Add a stress test that injects slow embedding/search calls and verifies active work remains bounded after request timeouts.

#### History overlap

The behavior is acknowledged in `docs/IMPLEMENTATION_LOG.md`; no complete cancellation or capacity fix is present.

### DRACO-007 — Redis outage multiplies rate limits per process

Severity: **P2**
Confidence: **High / confirmed**
References: `app/core/ratelimit.py:76-93`; `production.md:208`

#### Preconditions and trigger

Redis is unavailable while the application runs with multiple workers or replicas.

#### Root cause and failure behavior

Each `RedisLimiter` owns a separate `MemoryLimiter`. On Redis errors or during its five-second breaker window, requests use that local bucket. The fallback state is not shared across processes or replicas.

A reproduction created two limiters pointing at an unavailable Redis port and hit the same policy/key once through each:

```powershell
@'
import asyncio
from app.core.ratelimit import RedisLimiter
async def main():
    a = RedisLimiter("redis://127.0.0.1:1/0")
    b = RedisLimiter("redis://127.0.0.1:1/0")
    print((await a.hit("authfail", "same-client", 1, 1)).allowed,
          (await b.hit("authfail", "same-client", 1, 1)).allowed)
asyncio.run(main())
'@ | python -
```

Observed output (with two backend-unavailable warnings):

```text
limiter_a_allowed True limiter_b_allowed True
```

Each process can therefore consume its own burst budget. The production runbook already notes that the effective limit becomes N times higher with N processes.

#### Impact

During a Redis incident, brute-force protection and flood backstops weaken exactly when dependency failures may already be increasing load. Attackers can multiply upload, authentication-failure, or expensive-query traffic across workers.

#### Minimal fix and regression test

Fail closed for authentication-failure and upload policies when the shared limiter is unavailable, or use a separately shared durable fallback. Keep a bounded local fallback only for low-risk read traffic. Add a multi-process/fault-injection test asserting the chosen failover policy.

#### History overlap

Documented as known gap 10 in `production.md`; no fix found.

### DRACO-008 — `TRUSTED_PROXY_HOPS` permits client-supplied identity when misconfigured

Severity: **P2**
Confidence: **High / confirmed**
References: `app/core/ratelimit.py:112-125`; `app/core/http_settings.py:27,70`; `production.md:64-65,203`

#### Preconditions and trigger

`TRUSTED_PROXY_HOPS` is greater than zero while the app is directly reachable, or an untrusted intermediary can reach the app and set `X-Forwarded-For`.

#### Root cause and failure behavior

`client_ip_from()` trusts the selected rightmost XFF entry solely because the numeric setting is nonzero. It does not verify that the TCP peer belongs to configured proxy CIDRs or that a trusted proxy actually appended the chain.

With `TRUSTED_PROXY_HOPS=1`, the local harness returned an attacker-supplied header value:

```powershell
@'
import os, sys, types
p = types.ModuleType("prometheus_client")
class M:
    def __init__(self, *a, **k): pass
    def inc(self, *a, **k): pass
p.CollectorRegistry = p.Counter = p.Gauge = p.Histogram = M
p.generate_latest = lambda r: b""
p.CONTENT_TYPE_LATEST = "text/plain"
sys.modules["prometheus_client"] = p
os.environ["TRUSTED_PROXY_HOPS"] = "1"
from app.core.ratelimit import client_ip_from
print(client_ip_from({"x-forwarded-for": "203.0.113.77"}, "192.0.2.10"))
'@ | python -
```

Observed output:

```text
spoofed_client_ip 203.0.113.77
```

#### Impact

An attacker can rotate the perceived IP address to bypass per-IP authentication-failure and flood limits. This is deployment-dependent and is not a bypass when the app is correctly isolated behind a trusted proxy with sanitized headers.

#### Minimal fix and regression test

Configure trusted proxy network ranges and only parse forwarded headers when the immediate peer is in those ranges. Otherwise use the socket peer address. Fail startup if proxy-hop trust is enabled without a trusted-proxy allowlist. Add direct-client and trusted-proxy tests.

#### History overlap

The documentation warns that the setting is only for the project’s own proxies, but the trust boundary is not enforced in code.

### DRACO-009 — Approval remains valid after the source proposal expires

Severity: **P2**
Confidence: **High / confirmed**
References: `app/agent/store.py:105-137`; `app/agent/capabilities/knowledge_management.py:104-125`; `app/api/agent.py:79-90`

#### Preconditions and trigger

A web-source proposal is approved near the end of its proposal TTL. Its source row then expires before ingestion is submitted, while the approval row still has its own later expiry.

#### Root cause and failure behavior

`approve_source()` validates source expiry when approval is created, then sets approval expiry to `now + approval_ttl`. `valid_approval()` checks only the approval expiry, decision, revocation, URL hash, and content hash. It does not check `src['expires_at']` or require an active source status.

The reproduction expired the source after approval and observed:

```powershell
@'
import tempfile, time
from pathlib import Path
from app.agent.store import AgentStore
with tempfile.TemporaryDirectory() as d:
    s = AgentStore(Path(d) / "agent.db")
    src = s.create_source(user_id="alice", url="https://example.test",
        normalized_url="https://example.test", url_hash="u", title="test",
        content_hash="c", content_text="snapshot", reason="test",
        preview={}, trust={}, ttl_hours=0.001)
    ap = s.approve_source(src["source_id"], "alice", ttl_hours=1)
    s._write("UPDATE sources SET expires_at=?, status='expired' WHERE source_id=?",
             (time.time() - 1, src["source_id"]))
    print(s.get_source(src["source_id"])["status"],
          s.valid_approval(src["source_id"], "alice") is not None)
'@ | python -
```

Observed output:

```text
source_expired expired approval_valid True approval_expires_in_s 3600.0
```

`ingest_approved_source()` relies on `valid_approval()`, so the expired source can proceed to job claiming and ingestion.

#### Impact

The documented proposal TTL is not an effective upper bound. An expired proposal can be ingested after the source should no longer be actionable. This weakens approval/audit semantics and can retain stale external content longer than intended.

#### Minimal fix and regression test

Make `valid_approval()` require `src['expires_at'] >= now` and an allowed active status, or make approval expiry the minimum of source and approval expiry. Add tests for approval immediately before source expiry and ingestion after source expiry.

#### History overlap

No local fix was found. Existing tests cover approval expiry but not source expiry after approval.

### DRACO-010 — Private runtime documents and database history remain in Git objects

Severity: **P1**
Confidence: **High / confirmed**
References: Git history for `app/storage`; commits `bf738fd`, `1113e09`, `9ba0eed`, and `2e6efd2`

#### Preconditions and trigger

Anyone who can access the public repository, a fork, a clone, or a cached Git object can inspect historical commits before the runtime data was removed from the tip.

#### Root cause and failure behavior

Runtime SQLite data, uploaded PDFs, and generated result documents were committed in the repository’s history. Later commits deleted the files and added ignore rules, but normal deletion does not remove them from existing Git objects or forks.

The local audit confirmed the history without printing document contents:

```text
git log --all --oneline --decorate -- app/storage
2e6efd2 fix: address four major security and reliability issues (#9)
9ba0eed security: remove sensitive runtime data from repository
1113e09 removing documents from the repository
bf738fd Initial commit: RAG ingestion pipeline
```

The affected history includes a SQLite database and personal/research PDFs and result files. Whether secrets are present in the database was not disclosed or assumed.

#### Impact

Private documents, metadata, and any credentials or tokens that may have been stored in the database remain recoverable from repository history and existing forks. This is an exposure independent of the current working tree.

#### Minimal fix and regression test

Treat all historical runtime content as compromised: rotate any credentials found during an authorized secret scan, coordinate a history rewrite and force-update with all maintainers, invalidate caches/forks where possible, and verify the rewritten repository has no runtime artifacts. Add CI checks preventing `app/storage`, uploads, generated results, and secrets from entering Git.

#### History overlap

The deletion commits are directly related and are already present, but they do not remediate old Git objects without a coordinated history rewrite.

## 5. Security audit

Confirmed attack paths:

1. **SSRF:** authenticated agent user plus web capability and production `WEB_ALLOW_PRIVATE_HOSTS=true`; affected asset is every network service reachable from the app host.
2. **TLS interception:** network attacker plus an HTTPS OpenSearch deployment that leaves the verification variable unset; affected assets are OpenSearch credentials, document contents, and search integrity.
3. **Historical data exposure:** repository access; affected assets are committed runtime database and uploaded/generated documents.
4. **Rate-limit bypass:** direct or multi-worker deployment during Redis failure; affected assets are authentication and expensive ingestion/query capacity.
5. **Forwarded-header spoofing:** direct/untrusted access combined with `TRUSTED_PROXY_HOPS>0`; affected asset is IP-based abuse control.

The current owner checks prevent ordinary cross-user document selection in the tested paths, but they do not filter by document lifecycle status. Prompt injection is treated as a residual, documented limitation rather than a new confirmed bypass in this audit.

## 6. Reliability and data integrity

- Ingestion status transitions are persisted, but retrieval does not treat `completed` as the publication boundary.
- Batch indexing is not atomically published across Qdrant and OpenSearch. A failed later batch can leave partial content visible.
- Blocking embedding, search, and reranking work continues after async request cancellation, creating resource accumulation under dependency stalls.
- Redis fallback preserves availability at the cost of distributed enforcement; the security policy is weaker during an outage.
- SQLite connection lifecycle and durable outbox recovery were inspected and the focused tests passed. Those existing fixes were not reported as new findings.

## 7. Performance and capacity

Confirmed capacity risks are DRACO-005 (unbounded owner filter) and DRACO-006 (uncancellable blocking work). The only measurements collected for this audit were the 1,500-ID SQLite reproduction and the cancellation harness; no live Qdrant/OpenSearch load benchmark was run. The repository’s documented historical measurements were not treated as new audit measurements.

## 8. Verification matrix

| Check | Command | Result | Limitations |
|---|---|---|---|
| Working-tree state | `git status --short --branch` | PASS | Clean at start and after audit; only this report is added. |
| Recent history | `git log -5 --oneline` | PASS | Audited `de499c3`. |
| Remotes | `git remote -v` | PASS | No remotes changed. |
| Python syntax/bytecode compilation | `python -m compileall -q app src` | PASS | Does not import optional dependencies. |
| Framework unit tests | `python -m pytest tests/draco/unit -q` | PASS — 51 passed | No heavy app integrations. |
| Framework reliability tests | `python -m pytest tests/draco/reliability -q` | PASS — 6 passed | No live services. |
| Focused app tests | `python -m unittest tests.test_registry tests.test_registry_connections tests.test_ingestion_dispatch tests.test_pipeline tests.test_fusion_rrf tests.test_pdf_detector tests.test_pdf_detector_validation tests.test_settings_validation tests.test_calibration_errors tests.test_memory_connections tests.test_qdrant_collection tests.test_worker` | PASS — 77 run, 1 skipped | Focused subset only; one optional test skipped. |
| Full original test suite | `python -m unittest discover -s tests -t . -v` | FAIL — 151 run, 11 errors, 11 skipped | Missing `langchain_core` and `prometheus_client` caused import errors. |
| Full Draco pytest suite | `python -m pytest tests/draco -q` | FAIL during collection — 2 errors, 2 skipped | Missing `langchain_core` and `sentence_transformers`. |
| Deterministic eval runner | `python -m tests.draco.eval.runner` | FAIL — 0/36 | `draco` package not installed; RAG cases also lack `sentence_transformers`. |
| Draco eval pytest | `python -m pytest tests/draco/eval -q` | FAIL — 1 failed, 3 passed | RAG suite crashes on missing `sentence_transformers`. |
| Lint | `ruff check src tests/draco examples` | BLOCKED | `ruff` executable unavailable. |
| Type checking | `mypy src/draco` | BLOCKED | `mypy` executable unavailable. |
| Dependency consistency | `python -m pip check` | FAIL | `rq 2.12.0 requires croniter, which is not installed.` |
| Dependency vulnerability audit | `python -m pip_audit --strict` | BLOCKED | `pip_audit` module unavailable. |
| Docker client | `docker --version` | PASS | Docker 29.8.2 client present. |
| Dockerfile build check | `docker build --check .` | BLOCKED | Docker Desktop Linux daemon was not running. |
| Live startup/readiness | Application startup with Redis/Qdrant/OpenSearch/vLLM | NOT RUN | External services and complete dependency set unavailable. |
| PDF resource reproduction | Docling conversion with adversarial PDF | BLOCKED | `docling` is not installed locally; no production service was contacted. |

Targeted reproductions were run with temporary directories or in-memory harnesses outside tracked files. No production systems, remote branches, or remote resources were modified.

## 9. Prioritized remediation plan

1. **Immediately contain security exposure:** rotate any credentials or tokens found in historical runtime data, coordinate Git history cleanup, and review repository/fork access.
2. **Block SSRF and unsafe TLS defaults:** reject private-host mode in production, enforce egress allowlists, and require OpenSearch certificate verification at startup.
3. **Isolate PDF conversion:** move parsing behind resource limits and a killable boundary before enabling untrusted uploads in production.
4. **Fix retrieval publication:** filter only completed/published versions and make failed multi-backend indexing invisible until both indexes are consistent.
5. **Bound retrieval work:** replace the unbounded owner ID list with a backend-side owner filter or bounded published selector; add capacity limits.
6. **Control cancellation and backpressure:** eliminate uncancellable request-path blocking work or isolate it in bounded, killable workers.
7. **Define outage-safe abuse controls:** fail closed for high-risk rate-limit policies or provide a shared fallback; enforce trusted proxy CIDRs.
8. **Correct approval expiry:** invalidate approvals when their source proposal expires or becomes inactive, and add regression coverage.
9. **Repair verification hygiene:** install the declared development dependencies in a clean environment, run the full CI-equivalent checks, build the container with the daemon running, and test startup against disposable service instances.

## 10. Audit limitations

- No live Redis, Qdrant, OpenSearch, vLLM, RQ worker, or reverse proxy was available.
- The local Python environment does not match `requirements.txt` and does not have the editable `draco` package installed.
- Docling and the ML stack were not available for an adversarial PDF or end-to-end retrieval test.
- OpenSearch TLS construction was code-confirmed but not instantiated because `opensearchpy` is missing locally.
- No remote fetch, GitHub issue/PR mutation, branch change, merge, rebase, or remote write was performed. Local commit history and local fix branches were inspected; live issue/PR status was not queried.
- Historical runtime files were identified by path and commit history without printing their contents. A separate authorized secret/privacy review is required to determine exactly what data was exposed.
- No claim is made that the application has no additional defects in unexecuted integrations, third-party libraries, or deployment-specific proxy/network policies.
