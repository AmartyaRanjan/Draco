# tools.md — hierarchical tool layer: what was done to the project

Written 2026-10-13. Scope: the tool layer, hierarchical orchestration, reasoning-mode control and
context-budget management described in the task brief. Ingestion and retrieval were **reused, not rewritten**.

Contents: [1 Audit](#1-repository-audit) · [2 Gaps](#2-gap-analysis) · [3 Architecture](#3-architecture) ·
[4 Capability status](#4-capability-status) · [5 Key mechanisms](#5-key-mechanisms) · [6 API](#6-http-api) ·
[7 Configuration](#7-configuration) · [8 Tests and evaluation](#8-tests-and-evaluation) ·
[9 Files](#9-files-changed) · [10 Limitations](#10-limitations-and-risks) · [11 Next steps](#11-next-steps)

---

## 1. Repository audit

What existed before this work (verified by reading the code and probing the running services, not from the brief):

| Area | State found | Location |
|---|---|---|
| Ingestion | RQ queue + worker; PDF only (`process_document` raises for other suffixes); Docling → chunk → BGE-M3 embed → Qdrant + OpenSearch | `app/ingestion/*`, `app/api/routes.py`, `app/core/queue.py` |
| Registry | **SQLite** (`app/storage/documents.db`), statuses `uploaded/queued/processing/completed/failed`; content-hash duplicate check; **no user/owner column** | `app/db/registry.py` |
| Retrieval | dense ‖ BM25 → exact `chunk_id` dedup → RRF(k=60) → ColBERT (shared BGE-M3 backbone) → threshold(raw 0.55, provisional)+min-3 fallback | `app/retrieval/*` |
| Inference | Plain `ChatOrchestrator` (hand-rolled loop, two tools), `/chat`, `/chat/stream`, SQLite conversation store, API-key auth | `app/inference/*`, `app/api/chat.py` |
| Model runtime | vLLM v0.31.0 container, **Qwen3-4B-FP8** (not 8B-AWQ), `--reasoning-parser qwen3`, hermes tool parser, 8192 ctx, 2 seqs, 0.30 GPU util | `deploy/vllm/` |
| Infra up | Redis, Qdrant (`rag_chunks`), OpenSearch 2.17.1, vLLM — all reachable | probed |
| LangGraph | **Not installed, no graph existed** | — |
| Artifact libs | openpyxl/matplotlib/pillow present; python-docx, python-pptx, reportlab were missing | — |
| Tests | 137 passing (15 skipped) before this work | `tests/` |

### Differences between the brief and the repository (decisions taken)

| Brief says | Repository reality | Decision |
|---|---|---|
| Qwen3-8B-AWQ | Qwen3-4B-FP8 deployed; GPU has ~0.5–3.5 GB free (shared tenants), so 8B does not fit now | **Did not swap the model.** The runtime adapter is profile-based and was checked against both checkpoints' chat templates (both read `enable_thinking`), so the swap is a config change. 8B-AWQ pin: `4da05a8edb55c6046cce958586c33b61da07bb79`. |
| OpenSearch BM25 | Same | matches |
| PostgreSQL for history/state | Everything is SQLite; the only Postgres on the host belongs to an unrelated project | Kept SQLite in a **separate file** (`app/storage/agent.db`), same pattern as `conversations.db`. Schema is plain SQL, portable. |
| Existing LangGraph graph | None | Built one (new `app/agent/graph.py`); the existing `/chat` loop is untouched. |
| Web search/image providers | Nothing configured | Clean provider interfaces + real implementations (SearXNG, Brave, OpenAI-compatible images); **all report `not_configured`** until env is set. |

## 2. Gap analysis

| Needed by the brief | Status before | Status now |
|---|---|---|
| Capability registry, typed contracts, one result envelope | missing | done (`app/agent/registry.py`, `contracts.py`) |
| Hierarchical supervisor in LangGraph | missing | done (`app/agent/graph.py`) |
| 21 capabilities in 9 groups | 2 ad-hoc tools | done — see §4 |
| Reasoning-mode control with capability detection | hard-coded `enable_thinking` flag in `client.py` | done (`app/agent/runtime.py`), live-probed |
| Context-budget manager + `summarize_evidence` | char/BGE-tokenizer estimate only | done (`budget.py`, `summarize.py`) |
| Approval-gated web ingestion | missing | done (`store.py`, `knowledge_management.py`, `/agent/sources/*`) |
| Artifact generation + storage | missing | done (`artifacts.py`, capability modules) |
| SSRF-safe web fetch | missing | done (`web.py`) |
| Citation validation | marker stripping only | done (`validation.py`) |
| Per-user ownership of documents | missing | **not solved** (registry has no owner) — §10 |
| Persistent LangGraph checkpoints | missing | **not solved** — in-memory saver only — §10 |

## 3. Architecture

```
                         POST /agent/chat  (X-API-Key → user_id)
                                    │
 ┌──────────────────────── LEVEL 1  SUPERVISOR (LangGraph, app/agent/graph.py) ────────────────────────┐
 │ classify_request ─► configure_reasoning_mode ─► plan_capabilities ─► execute_capability ◄──┐        │
 │   (deterministic)      (policy + adapter)         (deterministic)            │             │ more   │
 │                                                                             ▼             │ steps /│
 │                                                                   check_context_budget    │ retry /│
 │                                                                     │ needs compaction?    │ web    │
 │                                                                     ▼                      │        │
 │                                                           summarize_if_required ─► evaluate_evidence │
 │                                                                                         │ sufficient │
 │   finalize ◄─ validate_citations ◄(one bounded revision)─ generate_final_response ◄─────┘            │
 └──────────────────────────────────────────────────────────────────────────────────────────────────────┘
        │ every capability call goes through CapabilityRegistry.execute()  (validate · permission · timeout ·
        │ size cap · error mapping · trace record)
        ▼
 ┌──────────────── LEVEL 2  CAPABILITY GROUPS (app/agent/capabilities/*, 21 capabilities) ─────────────────┐
 │ knowledge(4) web_research(3) knowledge_management(3) artifact_creation(3) visualization(3)              │
 │ file_analysis(1) computation(1) validation(2) context_management(1)       + set_reasoning_mode (control) │
 └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
        ▼
 ┌──────────────── LEVEL 3  ATOMIC SERVICES (Services container, app/agent/services.py) ───────────────────┐
 │ RetrievalService (Qdrant+OpenSearch+RRF+ColBERT) · registry.py · RQ ingestion queue · WebFetcher ·       │
 │ SearchProvider · ImageProvider · ArtifactStore · AgentStore(SQLite) · EvidenceSummarizer ·               │
 │ RuntimeAdapter(vLLM) · TokenCounter(Qwen tokenizer)                                                      │
 └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Design rules that the code enforces:

* **Deterministic first.** Intent classification, planning, routing, retries, budget decisions and query rewriting
  are plain functions. The LLM is used only (a) to pick tool calls inside a bounded, group-filtered loop for
  artifact/visualization/calculation/file tasks, (b) to write the final answer, (c) to summarize when the budget
  manager demands it. A grounded question costs **one** LLM call; "calculate 12*(3+4)" costs **zero**.
* **Task-filtered tool exposure.** The model only sees LLM-exposed capabilities of the selected groups.
  `ingest_approved_source`, `summarize_evidence`, `validate_citations` are orchestrator-only. A tool call outside
  the selected groups is rejected even if the model emits it.
* **Provenance is structural.** Evidence lives in a run-scoped `EvidenceStore`; private items require real
  `document_id` + `chunk_id`, web items require `url` + `passage_id` and are forbidden from carrying chunk ids.
  Citations in the response are built from the store, never from model-written ids.

## 4. Capability status

Legend: **T** = unit/integration tests in repo · **L** = exercised against the live stack in this session ·
**C** = blocked by missing configuration/provider.

| # | Capability | Group | Status | Notes |
|---|---|---|---|---|
| 1 | `search_knowledge_base` | knowledge | **Implemented, T + L** | reuses `RetrievalService` unchanged; whole chunks only within token budget; raw ColBERT score labelled "not a probability" |
| 2 | `get_document_metadata` | knowledge | **Implemented, T + L** | never returns `file_path`/raw error text; chunk count from OpenSearch |
| 3 | `expand_document_context` | knowledge | **Implemented, T; lookup service verified live** | neighbours by `chunk_index`; the OpenSearch lookup was run live, the capability itself only against a fake in tests |
| 4 | `rewrite_search_query` | knowledge | **Implemented, T + L** | deterministic (keywords / identifiers / sub-questions / shortened); no LLM |
| 5 | `web_search` | web_research | **Implemented, T; C** | SearXNG + Brave providers parse-tested with mock transports only; not tested against a live provider |
| 6 | `fetch_webpage` | web_research | **Implemented, T + L** | SSRF/redirect/size/content-type controls; live: metadata-IP, `::1`, localhost ports blocked, example.com fetched |
| 7 | `extract_web_evidence` | web_research | **Implemented, T** | deterministic idf-overlap passage ranking; stores as `W#` with URL + passage id |
| 8 | `propose_knowledge_ingestion` | knowledge_management | **Implemented, T** | preview + heuristic trust indicators; reuses existing proposal for same URL |
| 9 | `ingest_approved_source` | knowledge_management | **Implemented, T; end-to-end not run** | approval-gated, one job per source; enqueue bridge not run against the real queue (would write a test doc into the production index) |
| 10 | `inspect_source_provenance` | knowledge_management | **Implemented, T** | source → approval → job → registry status |
| 11 | `generate_document` | artifact_creation | **Implemented, T + L** | DOCX + PDF; real files re-opened with python-docx / pypdf in tests |
| 12 | `generate_presentation` | artifact_creation | **Implemented, T** | PPTX, notes, tables, chart images, sources slide |
| 13 | `generate_spreadsheet` | artifact_creation | **Implemented, T** | XLSX (allow-listed formulas, provenance sheet) + CSV; formula/CSV-injection blocked |
| 14 | `generate_chart_or_table` | visualization | **Implemented, T + L** | PNG via matplotlib; labels, units and underlying table always returned |
| 15 | `generate_image` | visualization | **Implemented, T; C** | OpenAI-compatible provider; `not_configured` until `IMAGE_*` set |
| 16 | `compose_multimodal_response` | visualization | **Implemented, T + L** | manifest only references artifacts verified to exist for that user |
| 17 | `analyze_uploaded_file` | file_analysis | **Implemented, T + L** | csv/tsv/xlsx/pdf/docx/txt/md; only files under the upload dir |
| 18 | `run_calculation` | computation | **Implemented, T + L** | restricted AST evaluator, not `eval` |
| 19 | `validate_citations` | validation | **Implemented, T + L** | identifier existence (incl. live OpenSearch) + lexical support; says it does not prove truth |
| 20 | `record_feedback` | validation | **Implemented, T** | separate table; `authorizes_ingestion` always false |
| 21 | `summarize_evidence` | context_management | **Implemented, T + L** | hierarchical, provenance-preserving, extractive fallback |
| — | `set_reasoning_mode` | control plane | **Implemented, T + L** | not an LLM tool; see §5.1 |

## 5. Key mechanisms

### 5.1 Reasoning-mode control (`app/agent/runtime.py`)

Verified facts (from the checkpoint files, not memory):

* `Qwen/Qwen3-4B-FP8` (deployed, rev `96b30dc…`) and `Qwen/Qwen3-8B-AWQ` (rev `4da05a8…`) ship a chat template that reads the Jinja
  variable `enable_thinking`. vLLM forwards `chat_template_kwargs` per request ⇒ **no restart/reload to switch**.
* There is **no** generic `thinking=true` API argument. `*-Instruct-2507` is non-thinking only; `*-Thinking-2507` always thinks —
  the adapter models those as *fixed-mode* profiles and reports `fallback_fixed_mode` instead of pretending.
* **Live probe against the running server:** thinking on → reasoning returned, 200 tokens; off → no reasoning, 12 tokens.

Behaviour: `ReasoningMode` enum → `ReasoningController` (bounded mode changes, default 2) → `RuntimeAdapter.build_llm`.
A decision's `applied` stays `None` until the adapter has built a client carrying the config, so "requested" is never
reported as "applied". Policy: non-thinking by default; thinking for artifact/research intents, ≥8 evidence items, or
contradictory evidence; after tool work finishes the closing message drops back to non-thinking. If thinking exhausts
the output budget the answer is retried once without thinking. Reasoning text is never returned to the user.

### 5.2 Context budget and summarization (`budget.py`, `summarize.py`, graph nodes)

* Counts use the **generation model's tokenizer** (Qwen, exact); the response states `token_counts: exact|estimate`.
  Window (`INFERENCE_MAX_MODEL_LEN`) and app evidence budget are separate; reserves for output, tool results and a safety margin are explicit.
* Compaction triggers **before** the next step when evidence passes `AGENT_EVIDENCE_COMPACTION_RATIO` (0.85) of the evidence budget. Ladder, each rung only if the previous one did not restore headroom:
  1. compact formatting → 2. (obsolete tool output is never replayed, nothing to drop) → 3. `summarize_evidence` on all but the top-2 items →
  4. trim older history → 5. omit lowest-ranked items **with an explicit warning**; if it still cannot fit, status `budget_error` rather than silent loss.
* `summarize_evidence`: batches → per-batch JSON summary → bounded merge (depth 3). Deterministic guards independent of the model: unknown source refs removed; claims whose
  **numbers are absent from the cited original text** are dropped and reported; negation/condition flips dropped and reported; contradictions and gaps from any level are
  re-attached if the merge dropped them; truncated model output keeps only complete claims and says so; final size is checked against the target (`over_budget` is an error, not silent).
  Originals stay in the `EvidenceStore`; only items actually covered by a kept claim are marked summarized. LLM failure → extractive fallback (verbatim sentences).

### 5.3 Approval-gated ingestion

`propose_knowledge_ingestion` stores a snapshot + preview + trust indicators → status `pending_approval`. Only
`POST /agent/sources/{id}/approve` (authenticated user) writes an approval row, bound to that user, that source's
URL hash **and** content hash, with an expiry. `ingest_approved_source` re-checks all of that, then claims the single
job for the source (unique constraint ⇒ concurrent/retried calls reuse it; a *failed* job can be retried atomically).
No capability code references `approve_source` (a test greps for it). Feedback is a separate table and cannot approve.
Because the existing worker only accepts PDFs, the approved snapshot is rendered to a PDF with a provenance header and submitted through
the same registry/queue path as `POST /documents` (content-hash dedupe included). Pages in scripts the PDF font cannot draw are refused up front.

### 5.4 Safety controls on web content

Scheme/credential/port allow-list → own DNS resolution → reject any non-public address (private, loopback, link-local incl. 169.254.169.254, CGNAT, multicast, v4-mapped v6) →
connect **to the validated IP** (Host header + SNI preserved) → manual redirects re-validated each hop (max 4) → streamed size cap (counts decoded bytes) → content-type allow-list →
HTML stripped of script/style/forms → instruction-like text flagged (`injection_suspected`) and everything labelled untrusted. Web evidence is `W#` + URL, never a chunk.

## 6. HTTP API

All routes require `X-API-Key` (same keys as `/chat`).

| Route | Purpose |
|---|---|
| `POST /agent/chat` | body `{message, conversation_id?, document_ids?, allow_web?, reasoning?: auto\|thinking\|non_thinking, max_output_tokens?}` → answer, `citations`, `tool_trace`, `artifacts`, `manifest`, `pending_approvals`, `validation`, `reasoning` (requested vs applied), `context` (budget + compaction log), `warnings` |
| `GET /agent/capabilities` | registry (schemas, cost, permissions, approval/untrusted flags), unconfigured providers, runtime reasoning capabilities |
| `POST /agent/sources/{id}/approve` (`{"ingest": true}` optional) / `…/reject` / `…/ingest` | the human-approval step |
| `GET /agent/sources/{id}` | provenance (approval, job, registry status) |
| `GET /agent/artifacts/{id}` | download (owner-only, expiry-checked, `nosniff`) |
| `POST /agent/feedback` | `{target_type: answer\|source\|artifact, target_id, rating: -1..1, comment?}` |

Invoking an individual capability from code: `registry.execute(name, args, CapabilityContext(...))` returns the common envelope
`{tool_name, status: success|empty|pending|error|timeout, data, metadata{trace_id, execution_id, duration_ms, output_chars, extra}, error{code, message, retryable}}`.
Web access is **off unless the request sets `allow_web: true`**.

## 7. Configuration

Template: [.env.agent.example](.env.agent.example). Secrets (`WEB_SEARCH_API_KEY`, `IMAGE_API_KEY`) come from the environment only.
Provider status is visible at `GET /agent/capabilities` → `unconfigured`. Currently unconfigured on this machine: **web search, image generation**.
Maintenance (artifact expiry, feedback retention): `python scripts/agent_maintenance.py` from cron.

## 8. Tests and evaluation

### 8.1 Automated tests

`.venv/bin/python -m unittest discover -s tests -t .` → **276 tests OK, 15 skipped** (was 137 OK / 15 skipped; the 15 skips are pre-existing:
Docling-dependent and live-infra tests — Docling is not installed in this venv, see §10).
**139 new tests**, none needing live services (fakes for retrieval, vLLM client, providers, transports):

| File | Tests | What it pins down |
|---|---|---|
| `tests/test_agent_core.py` | 35 | registry envelope (validation, permissions, timeout, empty-vs-error, no leaked internals, size cap, secret redaction, task-filtered exposure); reasoning adapter (enum validation, mode **reaches the client's `chat_template_kwargs`**, unsupported/fixed-mode/unknown-model reported truthfully, bounded changes, policy); budget (window vs evidence budget, compaction fires *before* overflow, exact/estimate flag, tool-definition tokens); summarizer (ID preservation, invented IDs removed, wrong numbers/negation flips dropped and reported, contradictions preserved through merges, nested merge, JSON retry, extractive fallback, over-budget reported, empty/oversized sets, truncated-output salvage) |
| `tests/test_agent_capabilities.py` | 60 | each capability's contract; SSRF (schemes, creds, ports, private/metadata/loopback/v4-mapped IPs, redirect-to-internal, redirect loops, content type, size cap, DNS-resolved-private); injection flagging; provider parsing; calculator safety; citation validation (invalid ids, numbers, negation, declines, user-supplied numbers, index existence); **approval enforcement, per-user scoping, expiry, content tamper, idempotent/concurrent ingestion, failed-job retry**; artifact generation re-opened with docx/pypdf/pptx/openpyxl, formula/CSV-injection blocked, size limit, ownership, expiry cleanup, path traversal, generator failure not faked, image provider failures, manifest verification; file analysis (path confinement, corrupt/too-large/unsupported) |
| `tests/test_agent_graph.py` | 37 | intent→group routing (web is opt-in); grounded answer with real ids; abstain without an LLM call; bounded retrieval retry; partial failures; one bounded citation revision; mode reaching the client per node; forced tool choice (first step only) and its fallback; artifact flow verified on disk; tools outside the selected groups rejected; tool-call/step limits; echo fallback; compaction triggered by the budget manager (not the model), originals kept, extractive fallback, explicit omission warning; web flow cites URLs not chunks; proposal ends in `awaiting_approval` with nothing ingested; graph compiles with the expected nodes |
| `tests/test_agent_api.py` | 7 | auth on every route, grounded `/agent/chat`, capability listing, approval flow over HTTP (user-scoped, idempotent), reject blocks ingestion, owner-only artifact download, feedback |

Coverage of the brief's §13 list: routing ✔ · retrieval-grounded answers ✔ · argument validation ✔ · unsupported runtime modes ✔ · approval enforcement ✔ ·
idempotent ingestion ✔ · artifact failures ✔ · budget threshold triggering ✔ · summary provenance ✔ · invalid-citation detection ✔ · retry/timeout limits ✔ ·
partial failures ✔ · token & latency measurements ✔ (live eval, below).

### 8.2 Live evaluation against the real stack

`scripts/eval_agent.py` + `tests/eval/agent_eval.jsonl` (9 cases) drive the real vLLM (Qwen3-4B-FP8), Qdrant, OpenSearch, BGE-M3 and ColBERT on the indexed documents.
Output: `eval_results/agent_<tag>.json` (per-case checks, latency, tokens, tool trace, requested-vs-applied mode, budget snapshots, compaction log).
Kept runs: `run3` (first run with the strict answer-quality checks — 7/9) and `run6` / `run6b` / `run6c` (final code).

Final run (`run6`) — 9/9 checks pass, all statuses `ok`:

| case | latency ms | LLM calls | output tokens | capabilities used | mode requested/applied |
|---|---|---|---|---|---|
| qa-fact | 16789* | 1 | 11 | search_knowledge_base, validate_citations | non_thinking/non_thinking |
| qa-abstain | 2189 | 1 | 46 | + rewrite_search_query (one bounded retry) | non_thinking/non_thinking |
| calc | 28 | **0** | 0 | run_calculation | non_thinking/**None** (no client built, so nothing "applied") |
| artifact-pdf | 46191 | 4 | 1599 | + generate_document (single PDF) | thinking/thinking → non_thinking/non_thinking |
| chart | 12328 | 5 | 397 | + generate_chart_or_table, compose_multimodal_response | non_thinking |
| web-no-optin | 1662 | 1 | 46 | no web tool called | non_thinking |
| web-unconfigured | 5972 | 1 | 226 | + web_search → `not_configured`, reported in `warnings` | thinking/thinking |
| compaction | 51616 | 2 | 314 | + summarize_evidence (LLM, orchestrator-invoked) | non_thinking |
| thinking | 14892 | 1 | 570 | search_knowledge_base | thinking/thinking |

\*includes cold-loading BGE-M3/ColBERT. Latency p50 12.3 s, max 51.6 s. **The GPU is shared with other tenants (22–23 of 23 GB in use, OOM warnings from the allocator
during runs), so absolute latencies are noisy** — the compaction case took 18 s in another run. Variance check: `artifact-pdf` and `chart` passed in 3 of 3 final runs.

What the live runs found (every item was invisible to the unit tests, and each now has a regression test):

| Finding | Fix |
|---|---|
| Summarizer fell back to extractive on real chunks: output cap tied to the 435-token *target*, and the SDK's `response_format` path raises `LengthFinishReasonError` on truncation | output cap scales with batch input; JSON mode via `extra_body`; complete claims salvaged from truncated JSON and the cut-off is reported |
| Negation heuristic dropped many valid claims (compared against the best sentence of *every* cited source) | single best sentence, ≥0.5 overlap required, light stemming, only strong conditionals ("unless", "only if"…) enforced |
| Chart answer was the model echoing my internal tool lines | notes are plain sentences labelled "do not quote"; deterministic reply if the model echoes them anyway |
| Model sometimes did not call the artifact tool (artifact-pdf failed once in 3 un-forced runs); it also sometimes made a DOCX *and* a PDF | first step forces the known capability via `tool_choice` (verified supported by this vLLM in both modes); requested format ("PDF", "Word", "CSV"…) overrides the model's default |
| In thinking mode reasoning + a long spec exceeded the 1000-token cap → truncated tool call → loop ended silently | artifact drafting has its own cap (`AGENT_ARTIFACT_MAX_OUTPUT_TOKENS`, 2400; ordinary answers stay at 1000); malformed/truncated call is detected and retried once without thinking; a missing call is a warning, not silence |
| Model wrote `[**S1**]` / `[S1-S3]`, so citations did not resolve | normalized to `[S1][S2][S3]` before validation |
| Honest declines ("the sources do not contain X") and user-supplied numbers were flagged unsupported | declines exempt; request/tool-result numbers count as supported (`user_context`); lenient lexical mode for tool-task replies |
| Compaction answer had no citations at all | mostly-uncited answers trigger the one bounded revision |
| `/chat` budgeted evidence with the BGE-M3 tokenizer, not the generator's | `app/api/chat.py` now passes the Qwen tokenizer counter |

Caveats on this evidence: 9 cases, same machine, structural + simple content checks — not a quality benchmark; no RAGAS (not installed; the local 4B is a weak judge).
Model quality limits are real: in a live chart the 4B model drafted two crossed series (renders faithfully, content is poor); `thinking` answers sometimes embellish
(the validator flagged an invented "section 11" that is not in the cited chunk — correctly).

### 8.3 Live checks outside the test suite (this session)

* Reasoning-mode probe on the running server: thinking on → reasoning returned (200 tokens); off → none (12 tokens). Forced `tool_choice` accepted in both modes.
* Tokenizer: Qwen tokenizer loads from cache and counts exactly.
* OpenSearch: `chunk_exists` (true for a real chunk; false for wrong document / missing chunk), neighbour expansion, per-document chunk counts.
* Network: metadata IP, `::1` blocked as non-public; `localhost:6333`, `127.0.0.1:9200` blocked by the port allow-list; `example.com` fetched.
* Registry: real uploaded file analysed; real documents' metadata (page/OCR/chunk counts).
* The repo's PDF detector classifies the generated web-snapshot PDF as all-digital (no OCR path).
* HTTP smoke of the real app with real keys: 401 without key; `/agent/capabilities` → 21 capabilities, 2 unconfigured; `/agent/chat` answers a calculation.

**Not run:** Docling conversion of the web-snapshot PDF (Docling not installed here); RQ enqueue → worker → index for a web-sourced document (would write a test document into the production index);
`web_search` against a live SearXNG/Brave; image generation against a live provider; load/concurrency testing; a swap to Qwen3-8B-AWQ.

## 9. Files changed

**New** (all untracked; nothing committed — commit when you are happy with it):

| Path | Purpose |
|---|---|
| `app/agent/contracts.py`, `registry.py`, `config.py` | result envelope + error codes; capability registry and the single execution wrapper; settings |
| `app/agent/runtime.py` | reasoning-mode enum, model profiles, capability detection, controller, adapter, policy, live probe |
| `app/agent/budget.py`, `summarize.py` | context-budget manager; hierarchical provenance-preserving summarizer |
| `app/agent/evidence.py`, `store.py` | evidence store (private vs web identity); SQLite state: sources, approvals, jobs, feedback, artifacts |
| `app/agent/web.py`, `artifacts.py` | SSRF-safe fetcher, extraction, search providers; artifact storage + DOCX/PDF/PPTX/XLSX/CSV/chart renderers |
| `app/agent/graph.py`, `schemas.py`, `prompts.py` | the LangGraph supervisor; request/response models; prompts |
| `app/agent/capabilities/*.py` (11 files) | the 21 capabilities by group |
| `app/agent/services.py`, `wiring.py`, `catalog.py` | Level-3 service container; production wiring (OpenSearch lookups, RQ bridge, vLLM client); registry builder |
| `app/api/agent.py` | `/agent/*` routes |
| `scripts/eval_agent.py`, `scripts/agent_maintenance.py`, `tests/eval/agent_eval.jsonl` | live evaluation; cron cleanup; eval cases |
| `tests/test_agent_{core,capabilities,graph,api}.py`, `tests/agent_helpers.py` | 139 tests + fakes |
| `.env.agent.example`, `tools.md`, `eval_results/agent_run{3,6,6b,6c}.json` | config template; this document; eval output |

**Modified existing files (small, behaviour-preserving):**

| File | Change |
|---|---|
| `app/main.py` | include the agent router (+2 lines) |
| `app/api/chat.py` | `/chat` now budgets prompt tokens with the **generation model's** tokenizer (was BGE-M3's via the retrieval counter) |
| `app/retrieval/context.py` | `TokenCounter.exact` property + loader refactor (`count()` behaviour unchanged) — lets budgets say "exact" vs "estimate" |
| `requirements.txt` | + `langgraph`, `python-docx`, `python-pptx`, `reportlab`, `openpyxl`, `matplotlib`, `beautifulsoup4`, `lxml`, `pillow` (pinned to the versions installed and tested) |
| `.gitignore` | `app/storage/agent.db*`, `app/storage/artifacts/` |

**Not modified:** ingestion (`app/ingestion/*`, `routes.py`, worker), the registry, the retrieval pipeline stages (dense/BM25/dedup/RRF/ColBERT/selection), the `/chat` orchestrator.
No semantic deduplication was introduced; exact `chunk_id` dedup, RRF and ColBERT are exactly as before.

## 10. Limitations and risks

Ordered by how likely they are to bite.

1. **No per-user ownership of documents.** The registry has no owner column, so any authenticated user can retrieve, analyse or read metadata of any registry document;
   `document_ids` only *narrows*. Conversations, sources, approvals, artifacts and feedback *are* user-scoped. Do not expose this multi-tenant until the registry gets an owner and retrieval filters use it.
2. **4B model quality.** Tool calling is unreliable without forcing (one failure in three un-forced artifact runs); model-drafted specs can be semantically poor (crossed chart series);
   `thinking` answers can embellish. Mitigations are in place (forced first call, validation, one bounded revision) but they reduce, not remove, the risk. Qwen3-8B-AWQ is the intended upgrade (checked: same template flag).
3. **Latency/capacity.** Thinking-mode artifact runs take 46–53 s, grounded QA 2–17 s on a GPU shared with other tenants (allocator OOM warnings during runs). vLLM is configured for 2 sequences and ~1.3× concurrency at 8k;
   artifact loops (up to 2400 output tokens, several calls) can queue. No load test was run.
4. **Graph state is not durable.** The checkpointer is `InMemorySaver`: a process restart loses an in-flight run (approvals, jobs, artifacts, feedback *are* durable in SQLite). A SQLite/Postgres checkpointer is an extra dependency I did not add.
5. **Human approval is not a LangGraph `interrupt`.** A run that proposes a source ends with status `awaiting_approval`; approval is a separate authenticated HTTP call; ingestion is then triggered by `/approve {"ingest": true}` or `/ingest`. The approval is a verifiable durable record, but the original graph run is not resumed.
6. **Web ingestion is lossy and PDF-shaped.** The existing worker accepts only PDFs, so an approved page's text snapshot is rendered to a PDF (tables flattened, original HTML not kept). Pages in scripts DejaVu cannot draw (e.g. Devanagari) are refused rather than ingested as blanks.
   The enqueue → worker → index path for these PDFs was not exercised end to end, and **Docling is not installed in this venv** (its model caches exist, so it was installed once) — the worker cannot run from here as-is.
7. **Providers unconfigured / untested live:** web search (SearXNG/Brave parsing is covered only by mock transports) and image generation. Both report `not_configured` and are visible at `/agent/capabilities`.
8. **Citation validation is lexical + identifiers + numbers + negation**, not entailment. Paraphrases can be marked "weak". An `entailment` hook exists in `Services` but is not wired to any model. A passing validation does not mean a claim is true.
9. **Token accounting** is exact for text (Qwen tokenizer) but chat-template overhead (6/message, 8 per prompt) and tool-schema size (JSON dump) are approximations covered by a 128-token safety margin.
10. **SSRF defences** resolve DNS themselves and connect to the validated IP; that is strong but not a substitute for egress firewalling the host. Redirects are re-validated; there is no domain allow-list or robots.txt handling.
11. **Observability** is structured logging (`rag.agent`, no document text, secrets redacted) plus the per-request trace in the response. There is no metrics exporter (none existed). Retention (`purge_feedback`, artifact expiry) is implemented but only runs via `scripts/agent_maintenance.py`; there is no HTTP endpoint for deleting a user's feedback.
12. `/agent/chat` does not stream (the existing `/chat/stream` is unchanged).

**Things found in the existing repo, deliberately not changed:**
* `app/storage/uploads/*.pdf` (personal documents) and `app/storage/documents.db` are **tracked in git**. Consider `git rm --cached` + ignoring `app/storage/uploads/` and `*.db`.
* `app/inference/client.py` still hard-codes the Qwen `enable_thinking` mapping that `app/agent/runtime.py` now owns as a model profile — a duplicate to consolidate when `/chat` is next touched.
* `documents.db` lists completed `.md` documents although `worker.py` currently rejects non-PDF files (probably ingested by earlier code).
* The relevance threshold (raw ColBERT 0.55) is still provisional, as recorded in `RETRIEVAL_LOG.md`.

## 11. Next steps

1. **Add an owner column to the registry** and thread `user_id` into retrieval filters (`resolve_filters` is already the single enforcement point) — unblocks multi-user use.
2. **Run the real ingestion path once for a web source** in a scratch index: install Docling in the worker environment, enqueue via `/agent/sources/{id}/approve {"ingest": true}`, confirm chunks land with the expected `document_id`, then re-ask the question and check the citation is a chunk, not a `W#`.
3. **Move to Qwen3-8B-AWQ** when GPU headroom allows (pin `4da05a8…`): set `INFERENCE_MODEL`, restart vLLM with the same flags, run `scripts/eval_agent.py --probe --tag 8b` and compare with `run6`. Expect better tool calling and spec quality.
4. **Durable checkpoints + real `interrupt()` approval** (LangGraph SQLite/Postgres saver) so a run can resume after the user approves.
5. **Configure and test a search provider** (`WEB_SEARCH_PROVIDER=searxng` is the self-hosted option), then add a live web case to `agent_eval.jsonl`.
6. **A better validator:** an entailment judge (stronger model) behind the existing `Services.entailment` hook; RAGAS faithfulness once a judge is available. Grow `agent_eval.jsonl` from 9 to 50+ labelled cases and calibrate the 0.55 retrieval threshold on them.
7. **Streaming** for `/agent/chat` (reuse the `emit` pattern from `/chat/stream`), per-user budgets/rate limits for web, image and artifact tools, and a metrics exporter.
8. Housekeeping: untrack uploads/DBs from git (see §10), consolidate the `enable_thinking` mapping, schedule `scripts/agent_maintenance.py`, and commit.
