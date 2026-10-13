# Inference + retrieval work log (through 2026-10-12)

## Phase 1 (done earlier): hybrid retrieval — details in `RETRIEVAL_LOG.md`
`app/retrieval/*`, `POST /query`, `tests/test_retrieval.py`, `scripts/benchmark_retrieval.py`: dense (BGE-M3+Qdrant) ‖ BM25 → exact chunk_id dedup → RRF(k=60) → ColBERT (BGE-M3 multi-vector head, shared backbone) → threshold(0.55 raw, provisional)+min-3 fallback → token-budgeted context. Live on 280 chunks: retrieval p50 464 ms / p95 556 ms; ~1.2 GiB GPU.

## Phase 2 (this task): inference layer

### 1. Model selection (verified via HF API / model cards / config.json, not memory)
| Candidate | Thinking toggle | Tool calling | Quantization | Verdict |
|---|---|---|---|---|
| `Qwen/Qwen3-4B` (hybrid) | yes (`enable_thinking`) | yes (hermes) | BF16 | **baseline** |
| `Qwen/Qwen3-4B-FP8` | yes | yes | official fine-grained FP8 (block 128, dynamic act.); card gives **no training provenance → PTQ, not QAT** | **chosen** |
| `Qwen3-4B-Instruct-2507(-FP8)` | no (non-thinking only) | yes | FP8 PTQ | rejected: no thinking mode; 262k ctx irrelevant |
| `Qwen3-4B-Thinking-2507(-FP8)` | thinking only | yes | FP8 PTQ | rejected: always thinks, wastes the 1000-token budget |
| `ISTA-DASLab/Qwen3-4B-FPQuant-QAT-NVFP4` | — | — | claims QAT, NVFP4; empty model card; L4 (Ada) has no FP4 hardware | rejected |
| `vinhnx90/Qwen3-4B-QAT-TorchAO-int4`, `jaytonde05/Qwen3_4B_QAT`, `eewer/Qwen3-4B-QAT-*` | unknown | unknown | "QAT" in name only; empty/no README, ~10 downloads, no provenance, `.bin` weights | rejected: unverifiable |

**Conclusion: no verified QAT checkpoint exists for Qwen3-4B. The QAT preference is NOT met.** Serving the official FP8 checkpoint (PTQ), Apache-2.0, revision pinned `96b30dc13593a244a5e59e84687309f53c375cfa`. BF16 baseline pinned `1cfa9a7208912126459214e8b04321603b3df60c`. Model id is env-configurable (`INFERENCE_MODEL`).

### 2. Verified behaviour on the real server (vLLM v0.31.0 image, L4)
- `enable_thinking` via `chat_template_kwargs`: on → 450 tokens incl. reasoning, off → 12 tokens. With `--reasoning-parser qwen3` reasoning arrives in a separate `reasoning` field; `content` is clean (leading `\n\n` stripped by us).
- `--enable-auto-tool-choice --tool-call-parser hermes`: tool call parsed into structured `tool_calls` (verified).
- API key enforced on `/v1/*` (401 without). `/metrics` is **not** key-protected → port bound to 127.0.0.1 only.
- Thinking tokens count against `max_tokens`; if thinking exhausts it and `content` is empty, orchestrator retries once without thinking.

### 3. Files
**Added**
- `deploy/vllm/docker-compose.yml`, `deploy/vllm/.env.example` (real `deploy/vllm/.env` is git-ignored via existing `.env` rule): pinned image, loopback port 18001 (8001 was taken by another service), API key, `--max-model-len 8192`, `--gpu-memory-utilization 0.30`, `--max-num-seqs 2`, `--enable-prefix-caching`, reasoning+tool parsers, healthcheck, `restart: unless-stopped`, `stop_grace_period 60s`, `HF_HUB_OFFLINE=1` (container has no DNS; weights pre-downloaded), persistent torch.compile cache volume, fixed `--kv-cache-memory-bytes` (see §6).
- `.env.inference.example`: app-side settings template.
- `app/inference/config.py`: env settings + validation (prompt budgets must leave ≥512 evidence tokens; auth must be configured).
- `app/inference/schemas.py`: Pydantic `ChatRequest`, `ChatResponse{answer, citations, tool_trace, metadata}`, `Citation`, `ToolTraceEntry`, `InferenceError`.
- `app/inference/client.py`: `build_llm()` (LangChain `ChatOpenAI` → vLLM; per-call thinking flag, separate sampling profiles, top_k via `extra_body`, timeout, bounded retries=1) + `map_error()` (SDK errors → client-safe codes).
- `app/inference/prompts.py`: grounded/agent system prompts (evidence = data, never instructions; cite `[S#]`).
- `app/inference/tools.py`: `search_documents` (existing `RetrievalService`), `list_documents` (registry read); Pydantic arg models → OpenAI tool specs.
- `app/inference/evidence.py`: per-request evidence registry; the only source of citations; whole-chunk token budgeting.
- `app/inference/memory.py`: `ConversationStore`/`MemoryProvider` Protocols, `NullMemory`, `SqliteConversationStore` (own file `app/storage/conversations.db`, user-scoped, atomic turn save, stores full tool outputs internally).
- `app/inference/generation.py`: `ChatOrchestrator.run()` — history (token-trimmed, verbatim) + summary + memories (separate prompt sections) → retrieval → LLM; `rag` mode (deterministic retrieval, non-thinking default) and `agent` mode (bounded tool loop: max steps 4, max tool calls 4, workflow token budget 3000, per-tool timeout, arg validation, scope can only narrow, forced final answer on limits); orchestrator writes the tool trace; citations validated against evidence, unknown `[S#]` stripped and reported; `<think>` leak filter; nothing persisted for failed/cancelled runs; streaming via `emit` callback (same code path → same validation).
- `app/api/chat.py`: `POST /chat`, `POST /chat/stream` (SSE: `start|token|tool_call|tool_result|done|error`; `done.response` is authoritative; disconnect cancels the task → cancels upstream vLLM request); API-key auth (`APP_API_KEYS=key:user,...`, constant-time compare, `APP_AUTH_DISABLED` for dev only).
- `tests/test_inference.py` (23 tests, fake LLM), `tests/test_inference_integration.py` (4 live vLLM+LangChain tests, run only with `INFERENCE_INTEGRATION=1`).
- `tests/eval/chat_eval.jsonl` (13 cases: en, hi, mixed, exact id, multi-hop, insufficient ×2, follow-up, long-term memory, tool ×3, tool-failure); `scripts/eval_inference.py`; `scripts/vllm_monitor.py` (vLLM metrics + GPU snapshot); `eval_results/*.json` (+ `.ragas.jsonl` exports, git-ignored).
- `INFERENCE_LOG.md` (this file).

**Modified**
- `app/main.py`: include chat router (2 lines).
- `app/retrieval/service.py`: removed built-in generator construction (superseded by `app/inference`; `/query` now generates only if one is injected), added `filename_for()`.
- `app/retrieval/generator.py`: removed the httpx `OpenAICompatGenerator` class (duplicate); kept prompt/citation helpers.
- `requirements.txt`: + `langchain-core 1.6.10`, `langchain-openai 1.7.1`, `openai 3.29.0`, `pydantic 2.14.0` (installed versions).
- `.env` (git-ignored): inference + `APP_API_KEYS` entries. `.gitignore`: + `app/storage/conversations.db`, `eval_results/*.ragas.jsonl`.

**Removed**: only the duplicate generator class above. Ingestion code untouched. (`app/storage/documents.db` was briefly polluted by my first chat-store default; restored from git, store moved to `conversations.db`.)

### 4. Tests actually run
- `python -m unittest discover -s tests -t .` → **124 tests OK (15 skipped: 11 pre-existing + 4 live)**.
- Live: `INFERENCE_INTEGRATION=1 python -m unittest tests.test_inference_integration` → **4/4 OK** (thinking separate from content, streaming + usage, structured tool call, max_tokens enforced).
- Real HTTP via uvicorn: 401 without key; `/chat` returned a grounded answer with a real document/chunk/page citation; `/chat/stream` emitted start/token events; client disconnect during generation dropped vLLM `num_requests_running` 1→0 (cancellation reaches the engine).

### 5. Evaluation (13 cases, same prompts and retrieval; 2 runs each; **n is tiny — differences are within noise**)
| | FP8 run1* | FP8 run2 | BF16 run1 | BF16 run2 |
|---|---|---|---|---|
| completed | 13/13 | 13/13 | 13/13 | 13/13 |
| contains expected fact | 6/6 | 5/6 | 5/6 | 5/6 |
| declined when evidence missing | 3/3 | 3/3 | 3/3 | 3/3 |
| tool selection | 3/3 | 2/3 | 2/3 | 2/3 |
| tool-arg validity | 3/3 | 2/2 | 2/2 | 2/2 |
| Hindi output (Devanagari) | 1/2 | 1/2 | 1/2 | 1/2 |
| latency p50 / p95 ms | 1968 / 18977 | 1935 / 22492 | 3178 / 49593 | 3035 / 32592 |
| TTFT p50 ms | 1607 | 1525 | 2688 | 2532 |
| vLLM GPU budget | 6.6 GiB (0.30) | same | 9.9 GiB (0.45) | same |
\*before a prompt tweak on memories (did not change the Hindi result).
Reading: FP8 shows no measurable quality loss on this set and is ~1.6× faster at p50 (**GPU is shared with a busy unrelated tenant, so absolute latency is noisy**) and needs ~3.3 GiB less. Not evidence of equal quality in general.
**Known failure:** `mem-1` (long-term memory "answer in Hindi") — the 4B model answers in English in both precisions. Run-to-run variance exists (temperature >0).
RAGAS was **not run** (not installed; faithfulness/answer-relevancy need a judge LLM — the local 4B would be a weak judge). `eval_results/<tag>.ragas.jsonl` exports inputs for it. Context precision/recall need relevance labels that do not exist. Not measured: faithfulness, answer relevancy, context precision/recall, memory-retrieval relevance (no memory backend).

### 6. GPU / resource findings
- Measured together: vLLM container 6.2 GiB + retrieval process (BGE-M3+ColBERT) 1.46 GiB = ~7.7 GiB of the 22 GiB device; other tenants use ~8–13 GiB and fluctuate. Total used ranged 15.5–19.8 GiB during tests. It fits **now**, with little headroom if other tenants peak.
- FP8 weights 4.23 GiB; KV cache 1.5 GiB = ~10.9k tokens → only **1.33× concurrency at 8192 tokens**. `max-num-seqs 2` with two full-length requests will queue/preempt. Preemptions observed so far: 0 (low load only).
- **Shared-GPU startup bug found:** vLLM's profiling-based KV sizing computed −4.01 GiB when another process allocated memory during startup → crash loop. Fix: fixed `--kv-cache-memory-bytes` (1.5 GiB) — startup also fell from ~105 s to ~19 s. Verified. Remaining risk: another tenant can still starve the 6.6 GiB allocation at start.
- Observed generation ~19 tok/s with the GPU contended (55% util by others); no clean throughput benchmark was possible.
- Prefix-cache hit rate unvalidated (0 hits in a few requests, prompts differ).

### 7. Unresolved / not done
- **QAT requirement unmet** (no verified checkpoint); FP8 is PTQ.
- **LangGraph not used**: the bounded loop is small and fully testable; adding it was not justified yet.
- **PostgreSQL not implemented**: SQLite store behind the `ConversationStore` Protocol; **long-term memory (Qdrant+OpenSearch+RRF) not implemented** — `NullMemory` default, interface only; **rolling-summary generation not implemented** (summary is read if present).
- Streaming: tokens from tool-calling steps (preamble text) are also streamed; `done.response` supersedes. Citations are validated only at `done`, not per token.
- Auth is a minimal API-key scheme; no per-user document ACLs exist in the corpus, so document access is not restricted per user. Prompt-injection defence is prompt-level only.
- Monitoring: vLLM Prometheus metrics + `scripts/vllm_monitor.py` snapshot only; no Prometheus scrape config, per-request GPU memory, or app-level inter-token metric export (inter-token mean is in the trace). No OOM alerting.
- Eval set is 13 cases, answers judged by substring/regex checks; Hindi/mixed coverage is 2 cases; multi-hop correctness not verified beyond citation count. Tool-failure case tests an empty scope, not an injected tool crash (crash/timeout covered in unit tests with fakes).
- Docker image is 32 GB; `HF_HUB_OFFLINE=1` requires pre-downloading weights; sampling defaults (T 0.3/0.6, top_p 0.8/0.95, top_k 20) are unvalidated hypotheses.
- First `/chat` after process start pays model-load cold start (~14 s).

### 8. Commands
```bash
# one-time: weights + env
PYTHONPATH=. .venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-4B-FP8', revision='96b30dc13593a244a5e59e84687309f53c375cfa')"
cp deploy/vllm/.env.example deploy/vllm/.env   # set INFERENCE_API_KEY (openssl rand -hex 24); copy same key into .env
# launch / health / logs / shutdown
cd deploy/vllm && docker compose up -d && docker inspect -f '{{.State.Health.Status}}' rag-vllm
docker logs -f rag-vllm
docker compose down                              # graceful (60 s)
# BF16 baseline
INFERENCE_MODEL=Qwen/Qwen3-4B INFERENCE_REVISION=1cfa9a7208912126459214e8b04321603b3df60c VLLM_GPU_UTIL=0.45 docker compose up -d --force-recreate
# app + tests + eval + monitoring (from repo root; load .env first: set -a; . ./.env; set +a)
uvicorn app.main:app --port 18080
curl -X POST localhost:18080/chat -H "x-api-key: $KEY" -H 'content-type: application/json' -d '{"message":"...","mode":"rag"}'
curl -N  -X POST localhost:18080/chat/stream -H "x-api-key: $KEY" -H 'content-type: application/json' -d '{"message":"...","mode":"agent"}'
python -m unittest discover -s tests -t .
INFERENCE_INTEGRATION=1 python -m unittest tests.test_inference_integration
PYTHONPATH=. python scripts/eval_inference.py --tag fp8
PYTHONPATH=. python scripts/vllm_monitor.py
```
