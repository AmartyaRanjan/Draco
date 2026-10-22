# production.md: what was done to make this service production-ready, and why

Written 2026-10-13. Scope: the HTTP platform layer around the existing ingestion, retrieval, chat and agent code. The retrieval pipeline itself
was **not** changed (its measured quality is in [evals.md](evals.md)). Every number below was measured in this repository on this machine unless it says otherwise.

**Read this first: what "production ready" means here.** The service is hardened for a *single-tenant trust domain*: one organisation, one shared document
corpus, authenticated API clients behind a reverse proxy. It is **not** safe for multi-tenant use (no per-user document access control). Section 6 lists every
known gap, in the order they are likely to hurt.

Contents: [1 Summary](#1-summary) · [2 Decisions and reasons](#2-decisions-and-reasons) · [3 Configuration](#3-configuration-reference) · [4 Measured behaviour](#4-measured-behaviour) ·
[5 Operations](#5-operations-runbook) · [6 Not done / known gaps](#6-not-done-and-known-gaps) · [7 Files](#7-files)

---

## 1. Summary

| Area | Before | Now |
|---|---|---|
| Authentication | `/chat`, `/agent/*` only. **`/documents` (upload, status) and `/query` were open to anyone.** | Every data route requires an API key. |
| Brute-force protection | none | 5 failed keys per address (refilling 10/min) locks that address out, **even for a valid key** while locked. |
| Rate limiting | none | Token bucket per user (and per address before auth), shared across processes in Redis; `429` + `Retry-After` + `X-RateLimit-*`. |
| Overload protection | requests queued behind vLLM until they timed out | Bounded LLM admission: wait up to 10 s, then `503` + `Retry-After`. |
| Request size | upload capped at 100 MB, everything else unbounded | 1 MiB for JSON routes (also when chunked, with no `Content-Length`), 101 MiB for uploads, `413` otherwise. |
| Upload validation | any file type accepted, then failed in the worker | `.pdf` suffix **and** `%PDF-` header checked while streaming; `415` before anything is queued. |
| Observability | unstructured logs | JSON logs with a per-request id (also returned as `X-Request-ID`), Prometheus `/metrics`, route-template labels (no id cardinality). |
| Health | `/health` returned `ok` unconditionally | `/health` (liveness) and `/health/ready` (Redis, Qdrant, OpenSearch, vLLM probed; `503` if any is down). |
| Startup | bad config failed on the first request | Settings validated at boot; `APP_ENV=production` refuses open auth, wildcard CORS, disabled rate limiting. |
| Security headers / CORS | none | `nosniff`, `X-Frame-Options: DENY`, `no-referrer`, `no-store`; CORS off unless origins are listed. |
| Persistence | SQLite rollback journal | WAL for the registry and conversation stores (API and worker write concurrently). |
| Agent robustness | two defects found by evaluation (below) | fixed |
| Packaging / CI | none | `Dockerfile` (non-root, healthcheck), `.dockerignore`, GitHub Actions unit-test workflow. **Not built or run here.** |
| Tests | 306 | 343 (+37: limiter, lockout, body caps, health, metrics, admission, settings, upload type, injection, retry) |

Two defects the evaluation (not code review) found in the agent, both fixed: **(1)** instruction-bearing web pages made the agent print an attacker's canary
(`sf-inject-web-plain`); **(2)** a PowerPoint request crashed with a context-window `400` by a single token, because the token budget cannot see chat-template and tool-schema overhead.

---

## 2. Decisions and reasons

Each decision lists what was chosen, what was rejected, and why.

### 2.1 Rate limiting: token bucket, in Redis, in our own code (`app/core/ratelimit.py`)

* **Algorithm: token bucket** (rate + burst), not fixed window or sliding log. A fixed window lets a client send 2x the limit across a window boundary; a
  sliding log costs O(requests) memory per client. A bucket is O(1), allows the short bursts a chat UI produces, and gives an exact `Retry-After`.
* **State in Redis**, one atomic Lua script per decision, using **Redis server time**. Reason: the app will run as more than one process or replica, and per-process
  counters multiply the real limit by the process count. Server time avoids clock skew between replicas. Redis was already a dependency (the RQ ingestion queue), so this adds no new infrastructure.
* **Redis down → fall back to per-process buckets, with a 5 s circuit breaker**, not fail-open and not fail-closed. Fail-open removes protection exactly when the platform is
  degraded; fail-closed turns a Redis blip into a full API outage. The breaker stops every request paying a connect timeout while Redis is away. Verified: an unreachable Redis still limits (unit test), and `rate_limit_backend_errors_total` counts the fallbacks.
* **Own implementation, not `slowapi`.** It keys by IP by default; we need the *authenticated user* as the key, a shared Redis bucket, a peek operation for the lockout, and
  our own metrics. That is about 150 lines, fully tested, with no extra dependency.
* **Different budgets per cost class**, not one global number:

  | Policy | Applies to | Default (per min, burst) | Why this number |
  |---|---|---|---|
  | `heavy` | `/chat`, `/chat/stream`, `/agent/chat`, `/query` | 20, 6 | The GPU serves 2 sequences; a single answer takes about 3 s (p50) to 7 s (p95). 20 per minute per user is more than an interactive human sends, and one user cannot occupy the model. |
  | `upload` | `POST /documents` | 10, 5 | Each upload triggers minutes of GPU conversion in the worker. |
  | `read` | other authenticated routes | 120, 30 | Cheap SQLite reads. |
  | `ip` | everything, before auth | 300, 60 | Flood backstop only; deliberately loose so a shared office address is not punished. |
  | `authfail` | failed key attempts | 10, 5 | See 2.2. |

  These are starting values chosen from measured service times, not tuned against real traffic. Change them with `RATE_LIMIT_*` and watch `rate_limit_rejections_total`.
* **`X-Forwarded-For` is ignored by default.** It is client-controlled; trusting it lets anyone mint a new identity per request. `TRUSTED_PROXY_HOPS=N` takes the N-th entry
  from the right (the one our own proxy appended).

### 2.2 Authentication everywhere, plus a brute-force lockout

`/documents` and `/query` were reachable without a key, so anyone who could reach the port could upload files to the GPU worker and query every document. They now use the same dependency as chat.
A failed key consumes a token from a per-address `authfail` bucket; when it is empty the address gets `429` **before** the key is compared. Rejecting valid keys during lockout is intentional: otherwise
the lockout leaks whether a guess was right. Keys are still compared with `hmac.compare_digest`.

### 2.3 LLM admission gate (`app/core/concurrency.py`)

Rate limits bound *per-user* demand; they do not bound *total* demand on a 2-sequence model. Without a gate, a burst from many users queues in vLLM, every request waits past the client timeout, and
the work is wasted. The gate lets `LLM_MAX_CONCURRENT` (4) requests in flight per process and makes the next wait at most 10 s, then answers `503` with `Retry-After`. Rejected quickly and
explicitly, the client can retry or show "busy". For streaming, the slot is taken **before** the response starts, so a saturated server returns a real `503` rather than a broken stream, and is released when the stream ends or the client disconnects.
Care was taken that the `503` is not swallowed by the routes' catch-all (which would have turned it into a `500`); there is a test.

### 2.4 One pure-ASGI middleware, not `BaseHTTPMiddleware` (`app/core/middleware.py`)

`BaseHTTPMiddleware` buffers responses and mishandles client disconnects, which this service relies on (SSE chat cancels the upstream vLLM request on disconnect). The middleware adds, in order: request id,
per-address flood limit, body-size cap, security headers, structured access log and metrics. The body cap counts bytes as they arrive, so a chunked upload with no `Content-Length` cannot bypass it
(FastAPI turns a failed body read into a `400`; the middleware reports the real cause, `413`). A bug of exactly this kind was caught by a test while writing it.

### 2.5 Observability

* **JSON logs with a request id** in a context variable, so every log line from a request (including the retrieval code's `extra={...}` fields) is correlatable. Request ids from clients are accepted only if alphanumeric, otherwise replaced (log-injection guard).
* **Metrics label by route *template*** (`/documents/{document_id}`), never the raw path. Raw paths make unbounded label cardinality, which takes Prometheus down. A test asserts no document id appears in `/metrics`.
* **Liveness and readiness are separate.** A restart does not fix a dead Qdrant, so liveness must not depend on it; readiness must, so traffic stops. Probes run concurrently with a 2 s timeout each and are cached for 5 s so an orchestrator probe storm cannot hammer the stores.
* **`/metrics` is open unless `METRICS_TOKEN` is set.** Put it behind the proxy or set the token.

### 2.6 Fail fast at boot; refuse unsafe production settings

Misconfiguration found on the first user request is an outage. The lifespan validates platform and inference settings (including the pre-existing "no API keys" check) and, when `APP_ENV=production`, refuses
`APP_AUTH_DISABLED=true`, `CORS_ORIGINS=*` and `RATE_LIMIT_ENABLED=false`. The `on_event("startup")` hook (deprecated) became a lifespan handler.

### 2.7 Upload validation

The worker only converts PDFs, so any other upload used to be queued, picked up, and failed minutes later. The route now rejects by suffix and by the `%PDF-` header of the first chunk (`415`),
before the file touches disk. This is a type check, not malware scanning: a hostile but valid PDF still reaches Docling (see 6.7).

Uploads now create a registry outbox row in the same SQLite transaction as the document. A leased dispatcher publishes deterministic RQ job IDs, retries with capped exponential backoff, and reconciles pending or lost queue jobs every 30 seconds while the API is running. If queue publication fails, the API returns `503` but retains the registered file so a duplicate upload can resume publication. Approved-source claims use a renewable five-minute lease; a later request can reclaim a claim abandoned before PDF submission, and its resulting document uses the same outbox.

### 2.8 SQLite stays, with WAL

PostgreSQL was considered and **not** adopted. The registry, conversation store and agent store are small, single-host and already behind interfaces. The real concurrency problem (API and worker both
writing `documents.db`) is solved by WAL, which lets readers proceed during writes. Moving to Postgres becomes necessary with more than one host, not before.

### 2.9 Single uvicorn worker per container

Each worker loads BGE-M3 and ColBERT onto the GPU (about 1.5 GiB). More workers on one GPU buy no throughput, because the lock in `retrievers.py` serialises GPU use anyway, and they multiply memory use. Scale with replicas (the rate limits are shared through Redis, so this is safe), subject to GPU capacity.

### 2.10 Agent fixes found by evaluation

* **Prompt injection through web pages.** Fetched passages that contain instruction-like text (`ignore previous instructions`, `<system>`, `disregard what the user…`, `call the tool…`) are now **withheld** from the model's evidence and reported
  (`quarantined` count, in the capability summary). Measured effect in [evals.md](evals.md): the plain-text payload that previously made the answer start with the attacker's canary no longer does. **This is a pattern filter; paraphrases get past it**
  (evals.md includes a paraphrased page the filter did not catch; the model happened to ignore it in the one trial run, which is luck, not a defence). It is one layer next to the existing ones (evidence labelled untrusted, no side-effecting capability runs without a human approval record, citation validation); it is not a guarantee.
* **Context-window overflow by a few tokens.** The agent's budget cannot count chat-template and tool-schema tokens, so a drafting call could request `max_tokens` that overflowed the window and the whole request failed. The loop now catches that specific
  `400` and retries up to four times with the output cap shrunk by 30% each time (the server's "at least N input tokens" is only `window + 1 - max_tokens`, so it carries no information to compute from; the first attempt at parsing it failed live, which is why the shrink is geometric).
* **Greetings no longer trigger retrieval and tool calls.** "Hello! How are you today?" used to run two searches, a query rewrite and two validations. A deterministic smalltalk intent now answers with no tools and no model call.

---

## 3. Configuration reference

Full annotated list: [.env.platform.example](.env.platform.example) (platform), [.env.inference.example](.env.inference.example), [.env.agent.example](.env.agent.example).

| Variable | Default | Notes |
|---|---|---|
| `APP_ENV` | `development` | `production` enforces the safety checks in 2.6 |
| `APP_API_KEYS` | none | `key:user,key2:user2`. Generate with `openssl rand -hex 24`. Keys live in the environment / a secret store, never in the repo |
| `RATE_LIMIT_ENABLED` | `true` | |
| `RATE_LIMIT_HEAVY/UPLOAD/READ/IP/AUTHFAIL` | see 2.1 | `per_minute,burst` |
| `REDIS_URL` | `redis://localhost:6379/0` | shared by RQ and the limiter. Give Redis persistence (AOF) if lost rate-limit state matters; losing it only resets buckets |
| `LLM_MAX_CONCURRENT` / `LLM_QUEUE_WAIT_S` | `4` / `10` | match vLLM `--max-num-seqs`: about 2x it, so a request is always ready as one finishes |
| `TRUSTED_PROXY_HOPS` | `0` | set to the number of proxies in front of the app |
| `MAX_BODY_BYTES` / `MAX_UPLOAD_BYTES` | 1 MiB / 101 MiB | |
| `CORS_ORIGINS` | empty (off) | explicit list only |
| `METRICS_TOKEN` | none | bearer token for `/metrics` |
| `LOG_JSON` / `LOG_LEVEL` | `true` / `INFO` | |

---

## 4. Measured behaviour

All runs: one machine, one L4-class GPU (22 GiB) **shared with other tenants**, vLLM serving Qwen3-4B-FP8 with `--max-num-seqs 2`, API and test client on the same host (no network latency). Raw output: `eval_results/load_test.jsonl`, `eval_results/load_metrics.txt`.

**Rate limiting and lockout against real Redis** (production defaults, `scripts/load_test.py`, 8 concurrent clients, 40 `/query` requests): 6 admitted (the burst), **34 answered `429` with `Retry-After: 3`**. Sequential bad keys: `401 401 401 401 401 429`; a valid key during the lockout: `429`. `rate_limit_rejections_total{policy="authfail"}` incremented as expected.

**Capacity with limits raised** (so the model, not the limiter, is the constraint; the same 8 questions repeat, so prefix caching flatters generation):

| Route | Concurrency | Requests | Result | Latency p50 / p95 | Throughput |
|---|---|---|---|---|---|
| `/query` (retrieval only) | 1 | 24 | 24x 200 | 0.34 s / 0.40 s | 2.9 req/s |
| `/query` (retrieval only) | 8 | 48 | 48x 200 | 2.72 s / 2.78 s | 2.9 req/s |
| `/chat` (retrieval + generation) | 1 | 12 | 12x 200 | 1.29 s / 2.45 s | 0.8 req/s |
| `/chat` | 4 | 24 | 24x 200 | 2.53 s / 3.72 s | 1.5 req/s |
| `/chat` | 16 | 48 | 48x 200 | 10.09 s / 11.42 s | 1.5 req/s |

Reading: **retrieval is GPU-serial at about 2.9 req/s** (the BGE-M3 + ColBERT lock), so concurrency only adds queueing: latency grows linearly with clients (0.34 s at 1, 2.7 s at 8) while throughput stays flat. **`/chat` saturates at about 1.5 req/s near concurrency 4**; at 16 clients every request still succeeded but waited about 10 s, right at the 10 s admission limit, so the gate (`llm_slot_rejections_total` was 0 here) was about to start shedding. The practical capacity of this deployment is therefore **roughly 90 grounded answers per minute in total**, shared by all users; the per-user `heavy` limit of 20 per minute means about 4 to 5 users can run flat out before the model is the bottleneck. Not tested: sustained load (minutes to hours), `/agent/chat` under load (each request makes several LLM calls and takes 2 to 60 s), or concurrent ingestion while serving.

**Latency under real contention.** The same hardware is shared: during the first RAG run the host load average was 70 to 185 (33 logged-in users) and one request took 417 s while two timed out; the repeat of the same workload on a quiet host had a p95 of 6.7 s and no errors. Treat the capacity table as an upper bound, and size for the contended case if the GPU is shared in production.

**Resource use.** vLLM about 6.6 GiB (`--gpu-memory-utilization 0.30`, 1.5 GiB fixed KV cache) plus the API process about 1.5 GiB for BGE-M3/ColBERT.

**Tests.** 343 unit tests pass (`python -m unittest discover -s tests -t .`), 15 skipped (integration tests that need live services). The new platform tests include one against real Redis (skipped if it is unreachable) showing two "replicas" share one bucket.

---

## 5. Operations runbook

**Start.** vLLM: `cd deploy/vllm && docker compose up -d`; API: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --no-access-log` (the middleware writes the access log); worker: `rqworker ingestion`
(needs Docling, which is only installed where ingestion runs). Gate traffic on `GET /health/ready`.

**Suggested alerts** (all metrics exist):

| Alert | Expression | Meaning |
|---|---|---|
| Dependency down | `dependency_up == 0` for 2 min | Redis / Qdrant / OpenSearch / vLLM failed readiness |
| Server errors | `sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) > 0.02` | |
| Latency | `histogram_quantile(0.95, sum by (le,route) (rate(http_request_duration_seconds_bucket[5m]))) > 30` on chat routes | the measured p95 is below 10 s when unloaded |
| Model saturated | `rate(llm_slot_rejections_total[5m]) > 0` | users are being shed: add a GPU/replica or raise `LLM_MAX_CONCURRENT` only if vLLM has headroom |
| Limiter degraded | `increase(rate_limit_backend_errors_total[10m]) > 0` | Redis is unreachable; limits are per-process only |
| Brute force | `increase(rate_limit_rejections_total{policy="authfail"}[10m]) > 20` | someone is guessing keys |

**Rotate an API key:** edit `APP_API_KEYS` (add the new, deploy, remove the old, deploy). Keys are checked on every request; no sessions to invalidate.

**Tuning a limit:** change `RATE_LIMIT_*`, restart. Buckets in Redis adopt new numbers on the next request; no flush needed.

**Backups.** Qdrant snapshots, OpenSearch snapshots, and the three SQLite files in `app/storage/` (with WAL, copy using `sqlite3 .backup`, not `cp`). Nothing here automates this.

---

## 6. Not done, and known gaps

Ordered by how likely they are to matter.

1. **No per-user document access control.** Any authenticated user can retrieve any indexed document. The registry has no owner column and the retrieval filter is `document_ids` only (it narrows, it does not authorise). Do not put documents with different audiences in one deployment until this exists.
2. **Prompt injection is mitigated, not solved.** See 2.10 and [evals.md](evals.md) for the measured residual. In particular, a poisoned *private* corpus chunk is still presented to the model; the agent then reports conflicting sources rather than trusting either (a measured behaviour, not a guarantee).
3. **Retrieval quality depends on how users phrase questions.** Questions prefixed with a long document descriptor ("In the 2DOF ... roadmap, ...") retrieve markedly worse (hit@5 0.79 vs 0.97 once the prefix is removed). Tiny chunks that lose their context (a CGPA line without the person's name) are missed by the agent path. Fixes (contextual chunk headers, query rewriting on the plain `/chat` path) change ingestion and need re-indexing; they were not done.
4. **The two large PDFs in `documents/` are not indexed** (a 548-page and a 673-page book are `queued`; Docling is not installed in this environment). All measurements are on the 7 indexed PDFs (280 chunks). Production corpora will be larger and noisier than anything measured here.
5. **TLS, authentication of `/metrics`, WAF and DDoS protection are the reverse proxy's job** and are not configured in this repository. Do not expose the app port directly.
6. **Single-host data stores.** SQLite files, one Qdrant, one OpenSearch, one Redis, one GPU shared with other tenants on this machine (measured load average 15 to 185 during testing; see 4). No replication, failover, or automated backup.
7. **Uploaded PDFs are parsed by Docling with no sandbox or per-file resource limit** beyond a 3600 s RQ timeout. A malicious PDF is an attack on the worker.
8. **Docker image and CI workflow are untested.** Neither was built or run; the Dockerfile pins Python 3.14 and omits Docling and CUDA, which the worker image needs.
9. **Rate limits are starting values**, chosen from measured service times, not from real traffic.
10. **The in-memory fallback limiter is per process**, so with N processes and Redis down the effective limit is N times higher until Redis returns.
11. **Auth is a static API-key list.** No expiry, no scopes, no audit log beyond the access log. Fine for service-to-service clients, thin for people.
12. **Conversation history and agent state have no retention job** except `scripts/agent_maintenance.py` for artifacts and feedback.

---

## 7. Files

New: `app/core/{http_settings,ratelimit,concurrency,observability,middleware}.py`, `app/api/health.py`, `tests/test_platform.py`, `.env.platform.example`, `Dockerfile`, `.dockerignore`, `.github/workflows/ci.yml`,
`scripts/{eval_rag,eval_judge,eval_ragas,eval_agent_suite,load_test}.py`, `tests/eval/{rag_gold,agent_suite}.jsonl`, `evals.md`, this file.

Changed: `app/main.py` (lifespan, middleware, routers), `app/api/{chat,agent,query,routes}.py` (auth, limits, admission, upload checks), `app/db/registry.py` and `app/inference/memory.py` (WAL),
`app/agent/{graph,web}.py` and `app/agent/capabilities/web_research.py` (section 2.10), `app/inference/client.py` (`reduced_output_cap`), `requirements.txt` (`prometheus-client`), `tests/__init__.py` (rate limiting off in unit tests), and the upload and query tests (they now send a PDF header / are authenticated).
