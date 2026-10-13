"""Chat orchestration: memory -> retrieval -> LLM (-> tools) -> validated response.

The orchestrator, not the model, is the source of truth for tool execution and citations.
"""
import asyncio
import json
import logging
import re
import time
import uuid
from typing import Awaitable, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

from app.inference import prompts
from app.inference.client import build_llm, map_error
from app.inference.config import InferenceSettings
from app.inference.evidence import EvidenceRegistry
from app.inference.memory import ConversationStore, MemoryProvider, NullMemory, new_conversation_id
from app.inference.schemas import (ChatRequest, ChatResponse, InferenceError, ResponseMetadata,
                                   ToolTraceEntry)
from app.inference.tools import TOOLS, ToolContext, tool_specs
from app.retrieval.context import TokenCounter
from app.retrieval.schema import RetrievalError
from app.retrieval.service import RetrievalService
from app.retrieval.timing import Trace

logger = logging.getLogger("rag.inference")
Emit = Callable[[dict], Awaitable[None]]
_THINK = re.compile(r"<think>.*?</think>", re.S)
_LABEL = re.compile(r"\[(S\d+)\]")


class ThinkFilter:
    """Defensive: drop <think>…</think> from streamed text if the server parser is misconfigured."""

    def __init__(self):
        self.buf, self.inside = "", False

    def feed(self, text: str) -> str:
        self.buf += text
        out = ""
        while True:
            if self.inside:
                j = self.buf.find("</think>")
                if j < 0:
                    self.buf = self.buf[-8:]
                    return out
                self.buf, self.inside = self.buf[j + 8:], False
            else:
                j = self.buf.find("<think>")
                if j >= 0:
                    out += self.buf[:j]
                    self.buf, self.inside = self.buf[j + 7:], True
                    continue
                keep = 6 if "<think>"[:1] in self.buf[-6:] else 0  # hold a possible partial tag
                out += self.buf[:len(self.buf) - keep]
                self.buf = self.buf[len(self.buf) - keep:]
                return out


class _Call:
    def __init__(self, content, tool_calls, usage, finish_reason, ttft_ms, total_ms):
        self.content, self.tool_calls, self.usage = content, tool_calls, usage
        self.finish_reason, self.ttft_ms, self.total_ms = finish_reason, ttft_ms, total_ms


class ChatOrchestrator:
    def __init__(self, retrieval: RetrievalService, settings: InferenceSettings,
                 store: ConversationStore, memory: MemoryProvider | None = None,
                 llm_factory=build_llm, counter: TokenCounter | None = None):
        self.retrieval, self.s, self.store = retrieval, settings, store
        self.memory = memory or NullMemory()
        self.llm_factory = llm_factory
        self.counter = counter or retrieval.counter

    # ---- LLM -------------------------------------------------------------------------
    async def _call(self, messages: list[BaseMessage], *, thinking: bool, max_tokens, tools: bool,
                    emit: Emit | None, trace: Trace, name: str) -> _Call:
        llm = self.llm_factory(self.s, thinking=thinking, max_tokens=max_tokens, streaming=emit is not None)
        if tools:
            llm = llm.bind_tools(tool_specs())
        t0, first, last, acc, gaps = time.perf_counter(), None, None, None, []
        flt = ThinkFilter()
        try:
            if emit is None:
                acc = await llm.ainvoke(messages)
                first = time.perf_counter()
            else:
                async for chunk in llm.astream(messages):
                    acc = chunk if acc is None else acc + chunk
                    text = chunk.content if isinstance(chunk.content, str) else ""
                    text = flt.feed(text) if text else ""
                    if text:
                        now = time.perf_counter()
                        if first is None:
                            first = now
                        elif last is not None:
                            gaps.append(now - last)
                        last = now
                        await emit({"event": "token", "text": text})
        except asyncio.CancelledError:
            trace.record(name, int(t0 * 1e9), "cancelled")
            raise
        except Exception as exc:
            trace.record(name, int(t0 * 1e9), "error", type(exc).__name__)
            raise map_error(exc) from exc
        end = time.perf_counter()
        if acc is None:
            raise InferenceError("Empty response from model", "llm_empty", 502)
        usage = getattr(acc, "usage_metadata", None) or {}
        content = _THINK.sub("", acc.content if isinstance(acc.content, str) else "").strip()
        trace.stages[name] = {"ms": round((end - t0) * 1e3, 3), "status": "ok",
                              "ttft_ms": round((first - t0) * 1e3, 3) if first else None,
                              "mean_inter_token_ms": round(sum(gaps) / len(gaps) * 1e3, 3) if gaps else None}
        return _Call(content, list(acc.tool_calls or []), usage,
                     (acc.response_metadata or {}).get("finish_reason"),
                     trace.stages[name]["ttft_ms"], trace.stages[name]["ms"])

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
    async def run(self, req: ChatRequest, user_id: str, emit: Emit | None = None) -> ChatResponse:
        s, rid = self.s, str(uuid.uuid4())
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
                allowed = req.document_ids
                if doc_ids and allowed and not set(doc_ids) <= set(allowed):
                    raise PermissionError("document outside the permitted set")
                scope = doc_ids or allowed
                outcome = await self.retrieval.retrieve(query, scope, rid + f"-{len(tool_trace)}", user_id)
                for st, v in outcome.trace.stages.items():
                    trace.stages[f"retrieval.{st}"] = v
                degraded.extend(d for d in outcome.degraded if d not in degraded)
                sel = outcome.selection
                if not sel or not sel.selected:
                    return "No relevant evidence found.", "0 evidence blocks"
                for c in sel.selected:
                    if c.document_id not in filenames_cache:
                        filenames_cache[c.document_id] = self.retrieval.filename_for(c.document_id)
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
                try:
                    outcome = await self.retrieval.retrieve(req.message, req.document_ids, rid, user_id)
                except RetrievalError as exc:
                    raise InferenceError(f"Retrieval failed ({exc.stage})", "retrieval_failed",
                                         exc.status_code) from exc
                for st, v in outcome.trace.stages.items():
                    trace.stages[f"retrieval.{st}"] = v
                degraded.extend(outcome.degraded)
                sel = outcome.selection
                text = ""
                if sel and sel.selected:
                    names = {c.document_id: self.retrieval.filename_for(c.document_id) for c in sel.selected}
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
                if thinking and not call.content and call.finish_reason == "length":
                    # Thinking consumed the whole output budget: one bounded retry without thinking.
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
            err = exc if isinstance(exc, InferenceError) else None
            if emit:
                await emit({"event": "error", "code": err.code if err else "internal",
                            "message": str(err) if err else "Internal error"})
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
            try:
                args = tool.args.model_validate(raw_args)
                safe_in = args.model_dump()
                out, summary = await asyncio.wait_for(tool.run(args, tctx), self.s.tool_timeout_s)
            except ValidationError as exc:
                status = "invalid_arguments"
                out = "Invalid arguments: " + "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
                summary = "Arguments failed validation"
            except asyncio.TimeoutError:
                status, out, summary = "timeout", "Tool timed out.", "Tool timed out"
            except PermissionError:
                status, out, summary = "denied", "Not permitted.", "Denied by authorization policy"
            except Exception as exc:
                logger.error("tool_failed", extra={"tool": name, "error": type(exc).__name__})
                status, out, summary = "error", "Tool failed.", "Tool failed"
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
        answer = _THINK.sub("", answer).strip()
        valid, invalid, seen = [], [], set()
        for label in _LABEL.findall(answer):
            if label in seen:
                continue
            seen.add(label)
            cit = evidence.citation(label)
            (valid if cit else invalid).append(cit or label)
        for label in invalid:  # never let an unverifiable reference reach the client
            answer = answer.replace(f"[{label}]", "")
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
        if emit:
            await emit({"event": "done", "response": resp.model_dump()})
        return resp
