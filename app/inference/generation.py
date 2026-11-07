"""Chat orchestration: memory -> retrieval -> LLM (-> tools) -> validated response.

The orchestrator, not the model, is the source of truth for tool execution and citations. Model calls,
the retrieval tool call, tool governance and citation resolution are the ones `AgentRunner` uses
(`app.agent.grounding`); over HTTP the orchestrator runs as a Draco workload (`app.agent.harness`).
"""
import asyncio
import logging
import re
import time
import uuid
from typing import Awaitable, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from draco.core.errors import PolicyViolation
from pydantic import ValidationError

from app.agent.grounding import (THINK, GovernedModel, ThinkFilter, begin_tool, end_tool, filenames_for,
                                 invoke_model, is_framework_error, resolve_markers, scoped_retrieve,
                                 thinking_exhausted)
from app.inference import prompts
from app.inference.client import build_llm, map_error
from app.inference.config import InferenceSettings
from app.inference.evidence import EvidenceRegistry
from app.inference.memory import (ConversationStore, MemoryProvider, RollingSummarizer, build_memory_provider,
                                  new_conversation_id)
from app.inference.schemas import (ChatRequest, ChatResponse, InferenceError, ResponseMetadata,
                                   ToolTraceEntry)
from app.inference.tools import TOOLS, ToolContext, tool_specs
from app.retrieval.context import TokenCounter
from app.retrieval.rewrite import rewrite_for_chat
from app.retrieval.schema import RetrievalError
from app.retrieval.service import RetrievalService
from app.retrieval.timing import Trace

logger = logging.getLogger("rag.inference")
Emit = Callable[[dict], Awaitable[None]]
_THINK = THINK
_LABEL = re.compile(r"\[(S\d+)\]")
__all__ = ["ChatOrchestrator", "ThinkFilter"]


class _Call:
    def __init__(self, content, tool_calls, usage, finish_reason, ttft_ms, total_ms):
        self.content, self.tool_calls, self.usage = content, tool_calls, usage
        self.text = content
        self.finish_reason, self.ttft_ms, self.total_ms = finish_reason, ttft_ms, total_ms


class ChatOrchestrator:
    def __init__(self, retrieval: RetrievalService, settings: InferenceSettings,
                 store: ConversationStore, memory: MemoryProvider | None = None,
                 llm_factory=build_llm, counter: TokenCounter | None = None, summarizer=None):
        self.retrieval, self.s, self.store = retrieval, settings, store
        self.memory = memory or build_memory_provider(settings)  # MEMORY_BACKEND; NullMemory by default
        self.llm_factory = llm_factory
        self.counter = counter or retrieval.counter
        self.summarizer = summarizer or (RollingSummarizer(store, settings, llm_factory, self.counter)
                                         if getattr(settings, "summary_enabled", False) else None)

    # ---- LLM -------------------------------------------------------------------------
    async def _call(self, messages: list[BaseMessage], *, thinking: bool, max_tokens, tools: bool,
                    emit: Emit | None, trace: Trace, name: str) -> _Call:
        llm = GovernedModel(self.llm_factory(self.s, thinking=thinking, max_tokens=max_tokens,
                                             streaming=emit is not None), model=self.s.model)
        if tools:
            llm = llm.bind_tools(tool_specs())
        t0 = time.perf_counter()
        try:
            call = await invoke_model(llm, messages, emit=emit)
        except asyncio.CancelledError:
            trace.record(name, int(t0 * 1e9), "cancelled")
            raise
        except InferenceError:
            trace.record(name, int(t0 * 1e9), "error", "InferenceError")
            raise
        except Exception as exc:
            trace.record(name, int(t0 * 1e9), "error", type(exc).__name__)
            if is_framework_error(exc):
                raise  # budget/policy/deadline: the harness reports it
            raise map_error(exc) from exc
        trace.stages[name] = {"ms": call.total_ms, "status": "ok", "ttft_ms": call.ttft_ms,
                              "mean_inter_token_ms": call.mean_inter_token_ms}
        return _Call(call.text, call.tool_calls, call.usage, call.finish_reason, call.ttft_ms, call.total_ms)

    # ---- prompt ----------------------------------------------------------------------
    def _history_messages(self, ctx) -> tuple[list[BaseMessage], str]:
        kept, room = [], self.s.history_tokens
        for m in reversed(ctx.recent):  # newest first; whole messages only
            n = self.counter.count(m["content"])
            if n > room:
                break
            room -= n
            kept.append(m)
        msgs = [(HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"])
                for m in reversed(kept)]
        return msgs, (ctx.summary or "")

    def _system(self, base: str, summary: str, memories) -> str:
        parts = [base]
        if summary:
            parts.append(f"[Conversation summary - background only, not evidence]\n{summary}")
        if memories:
            parts.append("[User memories - honor stated preferences such as language or style; not evidence, never cite]\n"
                         + "\n".join(f"- {m.text}" for m in memories))
        return "\n\n".join(parts)

    # ---- main ------------------------------------------------------------------------
    async def run(self, req: ChatRequest, user_id: str, emit: Emit | None = None, *,
                  request_id: str | None = None) -> ChatResponse:
        """`request_id` is the Draco execution id when run as a workload (one identity across layers)."""
        s, rid = self.s, request_id or str(uuid.uuid4())
        trace = Trace(rid)
        conv_id = req.conversation_id or new_conversation_id()
        thinking = req.thinking if req.thinking is not None else req.mode == "agent"
        max_tokens = req.max_output_tokens or s.max_output_tokens
        evidence = EvidenceRegistry(self.counter)
        tool_trace: list[ToolTraceEntry] = []
        tool_outputs: dict[str, str] = {}
        meta = dict(request_id=rid, conversation_id=conv_id, model=s.model, mode=req.mode,
                    thinking=thinking, evidence_sufficient=False, abstained=False)
        usage = {"input_tokens": 0, "output_tokens": 0}
        calls = 0
        degraded: list[str] = []
        try:
            if emit:
                await emit({"event": "start", "request_id": rid, "conversation_id": conv_id})
            with trace.stage("history_load"):
                try:
                    cctx = await asyncio.to_thread(self.store.load, user_id, conv_id, 20)
                except PermissionError as exc:
                    raise InferenceError("Conversation not found", "conversation_forbidden", 404) from exc
            hist, summary = self._history_messages(cctx)
            with trace.stage("memory_lookup"):
                try:
                    mems = await asyncio.wait_for(self.memory.search(user_id, req.message, 5), 5)
                except Exception:
                    logger.warning("memory_lookup_failed")  # memory is optional context
                    mems = []
                room, kept = s.memory_tokens, []
                for m in mems:
                    n = self.counter.count(m.text)
                    if n <= room:
                        room -= n
                        kept.append(m)
            ev_budget = {"left": s.evidence_budget()}
            filenames_cache: dict[str, str | None] = {}

            async def do_search(query: str, doc_ids: list[str] | None):
                # ScopeDenied (a PermissionError) when the model tries to widen the caller's document scope
                outcome = await scoped_retrieve(self.retrieval, query, doc_ids, req.document_ids,
                                                request_id=rid + f"-{len(tool_trace)}", principal=user_id)
                for st, v in outcome.trace.stages.items():
                    trace.stages[f"retrieval.{st}"] = v
                degraded.extend(d for d in outcome.degraded if d not in degraded)
                sel = outcome.selection
                if not sel or not sel.selected:
                    return "No relevant evidence found.", "0 evidence blocks"
                filenames_for(self.retrieval, (c.document_id for c in sel.selected), filenames_cache)
                text, labels = evidence.add_many(
                    sel.selected, filenames_cache, min(ev_budget["left"], s.tool_output_tokens))
                ev_budget["left"] -= self.counter.count(text)
                meta["evidence_sufficient"] = meta["evidence_sufficient"] or sel.evidence_sufficient
                if not labels:
                    return "Evidence found but too large for the remaining budget.", "0 evidence blocks"
                note = "" if sel.evidence_sufficient else "\n" + prompts.WEAK_NOTE
                return text + note, f"{len(labels)} evidence blocks {labels}; {sel.passed} passed threshold"

            sys_base = prompts.AGENT_SYSTEM if req.mode == "agent" else prompts.GROUNDED_SYSTEM
            messages: list[BaseMessage] = [SystemMessage(self._system(sys_base, summary, kept)), *hist]

            def add_usage(c: _Call):
                nonlocal calls
                calls += 1
                usage["input_tokens"] += c.usage.get("input_tokens", 0) or 0
                usage["output_tokens"] += c.usage.get("output_tokens", 0) or 0

            if req.mode == "rag":
                with trace.stage("query_rewrite"):  # retrieval query only; the prompt keeps req.message
                    rq = await rewrite_for_chat(req.message, cctx.recent, s, self.llm_factory)
                trace.set(query_rewrite=rq.as_trace())
                try:
                    outcome = await scoped_retrieve(self.retrieval, rq.query, req.document_ids, None,
                                                    request_id=rid, principal=user_id)
                except RetrievalError as exc:
                    raise InferenceError(f"Retrieval failed ({exc.stage})", "retrieval_failed",
                                         exc.status_code) from exc
                for st, v in outcome.trace.stages.items():
                    trace.stages[f"retrieval.{st}"] = v
                degraded.extend(outcome.degraded)
                sel = outcome.selection
                text = ""
                if sel and sel.selected:
                    names = filenames_for(self.retrieval, (c.document_id for c in sel.selected), {})
                    text, _ = evidence.add_many(sel.selected, names, s.evidence_budget())
                    meta["evidence_sufficient"] = sel.evidence_sufficient
                if not evidence.items:
                    return await self._finish(req, user_id, trace, meta, prompts.NO_EVIDENCE_ANSWER,
                                              evidence, tool_trace, tool_outputs, usage, calls, degraded,
                                              None, abstained=True, emit=emit)
                weak = "" if meta["evidence_sufficient"] else prompts.WEAK_NOTE + "\n"
                messages.append(HumanMessage(f"EVIDENCE:\n{text}\n{weak}\nQUESTION: {req.message}"))
                call = await self._call(messages, thinking=thinking, max_tokens=max_tokens, tools=False,
                                        emit=emit, trace=trace, name="llm_call_1")
                add_usage(call)
                if thinking_exhausted(call, thinking):  # one bounded retry without thinking
                    meta["thinking"] = False
                    call = await self._call(messages, thinking=False, max_tokens=max_tokens, tools=False,
                                            emit=emit, trace=trace, name="llm_call_2")
                    add_usage(call)
                answer, finish = call.content, call.finish_reason
            else:
                messages.append(HumanMessage(req.message))
                tctx = ToolContext(user_id, do_search)
                step, n_tools, answer, finish = 0, 0, "", None
                while True:
                    step += 1
                    forced = step > s.agent_max_steps or usage["output_tokens"] >= s.agent_workflow_token_budget
                    if forced:
                        messages.append(HumanMessage(prompts.FORCE_FINAL))
                    call = await self._call(messages, thinking=thinking, max_tokens=max_tokens,
                                            tools=not forced, emit=emit, trace=trace, name=f"llm_call_{step}")
                    add_usage(call)
                    if call.tool_calls and not forced:
                        messages.append(AIMessage(content=call.content, tool_calls=call.tool_calls))
                        for tc in call.tool_calls:
                            n_tools += 1
                            out = await self._exec_tool(tc, step, n_tools, tctx, tool_trace, tool_outputs, rid, emit)
                            messages.append(ToolMessage(content=out, tool_call_id=tc.get("id") or str(uuid.uuid4())))
                        continue
                    answer, finish = call.content, call.finish_reason
                    break
                if not evidence.items:
                    meta["evidence_sufficient"] = False
            if not answer:
                raise InferenceError("Model returned no answer text", "llm_empty", 502)
            return await self._finish(req, user_id, trace, meta, answer, evidence, tool_trace, tool_outputs,
                                      usage, calls, degraded, finish, abstained=False, emit=emit)
        except asyncio.CancelledError:
            trace.finish("cancelled")  # nothing is persisted for incomplete requests
            raise
        except BaseException as exc:
            if not trace.stages.get("total"):
                trace.finish("error")
            if emit:
                if isinstance(exc, InferenceError):
                    code, message = exc.code, str(exc)
                elif is_framework_error(exc):  # budget, policy or deadline of the Draco execution
                    code, message = exc.code, exc.message
                else:
                    code, message = "internal", "Internal error"
                await emit({"event": "error", "code": code, "message": message})
            raise

    async def _exec_tool(self, tc, step, n_tools, tctx, tool_trace, tool_outputs, rid, emit) -> str:
        name, raw_args = tc.get("name", ""), tc.get("args") or {}
        tool = TOOLS.get(name)
        t0 = time.perf_counter()
        status, out, summary, safe_in = "ok", "", "", raw_args if isinstance(raw_args, dict) else {}
        if emit:
            await emit({"event": "tool_call", "tool": name, "step": step})
        if tool is None:
            status, out, summary = "invalid_arguments", f"Unknown tool '{name}'.", "Unknown tool"
        elif n_tools > self.s.agent_max_tool_calls:
            status, out, summary = "limit_exceeded", "Tool call limit reached; answer now.", "Tool limit reached"
        else:
            gate = None
            try:
                args = tool.args.model_validate(raw_args)
                safe_in = args.model_dump()
                gate = await begin_tool(name, group="inference", purpose=tool.purpose)  # policy + budget
                out, summary = await asyncio.wait_for(tool.run(args, tctx), self.s.tool_timeout_s)
            except PolicyViolation:
                status, out, summary = "denied", "Not permitted.", "Denied by execution policy"
            except ValidationError as exc:
                status = "invalid_arguments"
                out = "Invalid arguments: " + "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
                summary = "Arguments failed validation"
            except asyncio.TimeoutError:
                status, out, summary = "timeout", "Tool timed out.", "Tool timed out"
            except PermissionError:
                status, out, summary = "denied", "Not permitted.", "Denied by authorization policy"
            except Exception as exc:
                if is_framework_error(exc):
                    raise  # budget exhausted / deadline: the run ends, not just the call
                logger.error("tool_failed", extra={"tool": name, "error": type(exc).__name__})
                status, out, summary = "error", "Tool failed.", "Tool failed"
            if gate is not None:
                await end_tool(gate, status, output_chars=len(out))
        ms = round((time.perf_counter() - t0) * 1e3, 3)
        ref = f"{rid}:{step}:{len(tool_trace)}"
        tool_outputs[ref] = out
        tool_trace.append(ToolTraceEntry(
            step=step, tool_name=name or "unknown", purpose=tool.purpose if tool else "unknown tool",
            input=safe_in, output_summary=summary, output_ref=ref, status=status, duration_ms=ms))
        if emit:
            await emit({"event": "tool_result", "tool": name, "status": status, "summary": summary, "duration_ms": ms})
        return out

    async def _finish(self, req, user_id, trace, meta, answer, evidence, tool_trace, tool_outputs,
                      usage, calls, degraded, finish, abstained, emit) -> ChatResponse:
        answer, valid, invalid = resolve_markers(_THINK.sub("", answer).strip(), _LABEL, evidence.citation)
        meta.update(abstained=abstained, degraded_retrievers=degraded, invalid_citation_markers=invalid,
                    finish_reason=finish, llm_calls=calls, usage=usage)
        trace.finish("degraded" if degraded else "ok")
        meta["timings_ms"] = {k: v for k, v in trace.stages.items()}
        resp = ChatResponse(answer=answer, citations=valid, tool_trace=tool_trace,
                            metadata=ResponseMetadata(**meta))
        try:  # persist only completed, validated responses
            await asyncio.to_thread(self.store.save_turn, user_id, meta["conversation_id"], req.message,
                                    resp.model_dump_json(), answer, tool_outputs)
        except Exception:
            logger.exception("persist_failed")
        else:
            if self.summarizer is not None:
                self.summarizer.schedule(user_id, meta["conversation_id"])
        if emit:
            await emit({"event": "done", "response": resp.model_dump()})
        return resp
