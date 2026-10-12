# Retrieval implementation log (2026-10-12)

Ingestion code (`app/ingestion/*`, `app/db/*`, worker, routes) is **unchanged**.

## Added
| File | Purpose |
|---|---|
| `app/retrieval/settings.py` | Env-driven, validated settings (`get_settings()`), calibration loader |
| `app/retrieval/timing.py` | `Trace` (perf_counter_ns stage timer, records on exception), cold-start dict |
| `app/retrieval/schema.py` | `Candidate` dataclass, `RetrievalError(stage, status_code)` |
| `app/retrieval/retrievers.py` | `DenseRetriever` (existing `embed_texts` + Qdrant `query_points`), `LexicalRetriever` (OpenSearch `multi_match` on `text`, `section_path^0.5`); `GPU_LOCK`; sync clients run via `asyncio.to_thread` + `wait_for` timeout |
| `app/retrieval/fusion.py` | Exact `chunk_id` dedup (keeps both ranks/scores; text/version conflict flagged + logged, Qdrant record kept) and RRF (1-based, tie-break: best source rank, then chunk_id) |
| `app/retrieval/rerank.py` | ColBERT = BGE-M3 multi-vector head (`colbert_linear.pt` from `BAAI/bge-m3`), score = mean MaxSim; shares the already-loaded BGE-M3 backbone (no 2nd copy), length-bucketed batches, lazy load |
| `app/retrieval/selection.py` | Threshold + min-evidence fallback (`passed_threshold` / `fallback_below_threshold`), optional `Calibrator` |
| `app/retrieval/context.py` | Token counter (BGE-M3 tokenizer, char-estimate fallback), whole-chunk budgeting, `[S#]` evidence blocks with file/doc/pages/section/relevance |
| `app/retrieval/generator.py` | OpenAI-compatible streaming generator (httpx), TTFT/duration/usage, citation extraction from `[S#]` markers (unknown labels reported, never invented) |
| `app/retrieval/service.py` | `RetrievalService.retrieve()` / `.answer()` (FastAPI-independent), degraded mode, `resolve_filters()` access hook, `get_service()` singleton |
| `app/api/query.py` | `POST /query` |
| `tests/test_retrieval.py` | 18 unit tests (fusion, selection, context, timing, service, API) with fakes |
| `scripts/benchmark_retrieval.py` | p50/p95 latency; Recall@K/MRR for dense, BM25, RRF, ColBERT, selected when labels exist |
| `RETRIEVAL_LOG.md` | this file |

## Modified
- `app/main.py`: import + `include_router(query_router)` (2 lines).
- `requirements.txt`: added `httpx==0.28.1` (used directly by the generator).

## Removed
Nothing.

## API
`POST /query` `{query (1-2000 chars), document_ids?: [≤50], generate?: true, include_diagnostics?: false}`
→ `request_id, answer, abstained, evidence_sufficient, degraded_retrievers, citations[], sources[], diagnostics?`.
Errors: 422 invalid, 503 both retrievers down, 502/504 generator, 500 rerank/unexpected (generic message, no stack trace).

## Config (env; defaults)
`RETRIEVAL_DENSE_TOP_K=30`, `RETRIEVAL_BM25_TOP_K=30`, `RETRIEVAL_RRF_K=60`, `RETRIEVAL_DENSE_TIMEOUT_S/BM25_TIMEOUT_S=10`,
`COLBERT_MODEL=BAAI/bge-m3`, `COLBERT_DEVICE` (=EMBEDDING_DEVICE), `COLBERT_SHARE_BACKBONE=true`, `COLBERT_BATCH_SIZE=8`, `COLBERT_MAX_CANDIDATES=60`, `COLBERT_MAX_LENGTH=512`,
`RETRIEVAL_SCORE_THRESHOLD=0.55` (raw) / `0.8` when `RETRIEVAL_CALIBRATION_FILE` is set (JSON `[[raw_score, prob], ...]`, monotone),
`RETRIEVAL_MIN_SELECTED=3`, `RETRIEVAL_MAX_SELECTED=20`, `RETRIEVAL_MAX_CONTEXT_TOKENS=6000`, `RETRIEVAL_TOKENIZER`,
`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_TIMEOUT_S=60`, `LLM_MAX_OUTPUT_TOKENS=800`, `LLM_TEMPERATURE=0`, `RETRIEVAL_LOG_LEVEL`, `RETRIEVAL_DIAGNOSTICS_VERBOSE`.
Set `COLBERT_SHARE_BACKBONE=false` + `COLBERT_DEVICE=cpu` to move reranking off the GPU.

## Verification (actually run)
- `python -m unittest discover -s tests -t .`: **97 tests OK (11 pre-existing skips)**, includes 18 new.
- Live smoke on real Qdrant (280 pts) + OpenSearch (280 docs) in `rag_chunks`: dense 30 / BM25 30 → ~11-14 overlap → 46-49 unique → ColBERT → selection works; Hindi query ran (BM25 returned 0 hits; no Hindi docs in corpus, so Hindi quality is **unverified**).
- Steady-state latency (4 queries × 10 runs, L4, shared GPU): dense (embed+Qdrant) p50 55 / p95 84 ms; BM25 p50 17 / p95 31 ms; fusion ~0.2 ms; ColBERT (≈46 cands) p50 364 / p95 490 ms; retrieval total p50 464 / p95 556 ms. Cold start: first request ~11 s (BGE-M3 load), ColBERT head load 0.3 s.
- Peak torch allocation in a process doing embed+rerank: ~1.17 GiB (budget 6 GiB).

## Not done / limitations
- **Not verified:** LLM generation (no provider configured; code is unit-tested with fakes only), streaming TTFT against a real endpoint, `generate` path end-to-end, GPU-memory contention with a co-resident local LLM.
- **Threshold 0.55 is provisional/uncalibrated** (raw MaxSim ≠ probability). On the broad query "reinforcement learning" 20 chunks passed (cap hit), so it is loose; no labelled data exists to calibrate. Provide labels (benchmark format above), fit an isotonic map, supply `RETRIEVAL_CALIBRATION_FILE`, then use 0.8.
- No quality claims: no labelled dataset exists, so Recall/MRR/nDCG were not measured (script ready).
- No language filter (payload has no language field); no auth/tenancy in the app, so `resolve_filters()` only narrows by document_id — it is the place to add real access control.
- Query endpoint does not yet exclude documents that are not in `ready` registry status (stale versions are already deleted by ingestion).
- Token counting uses the BGE-M3 tokenizer, an approximation for the generator's tokenizer; set `RETRIEVAL_TOKENIZER` to match.
- Timed-out DB threads cannot be cancelled (client-level timeouts also set); no retries added deliberately.
- Models load lazily on first request (no startup warm-up); no Prometheus/OTel export yet (stage names/trace dict are export-ready).
