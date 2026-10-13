"""Level 1 - the supervisor: an explicit LangGraph workflow.

  classify_request -> configure_reasoning_mode -> plan_capabilities -> execute_capability
     -> check_context_budget -> [summarize_if_required] -> evaluate_evidence
        -> (more capabilities | rewrite/retry | web research) ... -> generate_final_response
     -> validate_citations (-> one revision) -> finalize

Deterministic functions decide routing, budgets and retries. The LLM is used only to (a) choose tool calls
inside a bounded, group-filtered agent loop for artifact/visualization/computation tasks, (b) write the final
answer, (c) summarize evidence when the budget manager demands it. State is plain serializable data; live
objects (evidence store, clients) live in the per-run context.
"""
import asyncio
import json
import logging
import operator
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent import prompts
from app.agent.capabilities.validation import normalize_citations
from app.agent.budget import BudgetConfig, ContextBudget
from app.agent.contracts import new_id
from app.agent.evidence import EvidenceStore
from app.agent.registry import CapabilityContext, CapabilityRegistry, TraceRecord
from app.agent.runtime import ModePolicy, ReasoningController, ReasoningMode, RuntimeAdapter
from app.agent.schemas import AgentChatRequest, AgentChatResponse, AgentTraceEntry, CitationOut
from app.agent.summarize import EvidenceSummary, render_summary
from app.inference.client import map_error, reduced_output_cap
from app.inference.memory import ConversationStore
from app.inference.schemas import InferenceError

logger = logging.getLogger("rag.agent")
_THINK = re.compile(r"<think>.*?</think>", re.S)
_MARK = re.compile(r"\[([SW]\d+)\]")

# ---- intent classification (deterministic) ------------------------------------------------
_ARTIFACT_VERB = r"(create|generate|make|draft|write|prepare|build|export|produce|compile)"
_KINDS = {"presentation": r"(presentation|slides?|deck|pptx?)", "spreadsheet": r"(spreadsheet|xlsx|excel|csv)",
          "document": r"(report|document|docx|pdf|memo|summary document|brief)"}
_CHART = r"(chart|plot|graph|visuali[sz]e|bar chart|pie chart|line chart)"
_IMAGE = r"(image|picture|illustration|photo|draw)"
_CALC = r"(calculate|compute|what is [\d(]|how much is [\d(]|\d+\s*[-+*/^%]\s*\d+|average of|mean of|sum of|percent(age)? of)"
_WEB = r"(search the web|web search|google|look up online|online|latest|news|today|currently|current|https?://)"
_FILE = r"(analy[sz]e|columns?|statistics|summari[sz]e)\b.*\b(file|csv|spreadsheet|xlsx|upload|data)|\b(uploaded|attached)\b"
_SAVE = r"(save|add (it |this )?to (the )?(knowledge|kb|database)|ingest|index this|remember this source|store this source)"
_PROV = r"(where did .* come from|provenance|source of (this|that) document|ingestion (status|history))"


_FMT = [(r"\bpdf\b", "pdf"), (r"\b(docx|word)\b", "docx"), (r"\bcsv\b", "csv"), (r"\b(xlsx|excel)\b", "xlsx")]


def requested_format(message: str) -> str | None:
    m = message.lower()
    return next((f for p, f in _FMT if re.search(p, m)), None)


_GREET = r"(hi+|hello+|hey+|namaste|good (morning|afternoon|evening)|thanks?( you)?|thank you( so much)?|bye|goodbye|how are you( doing)?( today)?|who are you|what can you do|help)"
_SMALLTALK = re.compile(rf"^\W*{_GREET}(\W+{_GREET})?\W*$", re.I)
SMALLTALK_REPLY = ("Hello! I can answer questions about the indexed documents (with citations), run calculations, and create "
                   "reports, spreadsheets, presentations and charts from them. What would you like to know?")


def classify(message: str, has_docs: bool, allow_web: bool) -> dict:
    m = message.lower()
    if _SMALLTALK.match(m):  # a greeting needs no retrieval, tools or model call
        return {"intent": "smalltalk", "wants_web": False, "wants_ingest": False, "has_numbers": False}
    flags = {"wants_web": bool(allow_web and re.search(_WEB, m)), "wants_ingest": bool(allow_web and re.search(_SAVE, m)),
             "has_numbers": bool(re.search(r"\d", m))}
    kind = next((k for k, p in _KINDS.items() if re.search(p, m)), None)
    if re.search(_ARTIFACT_VERB, m) and kind:
        return {"intent": "artifact", "artifact_kind": kind, "format": requested_format(message), **flags}
    if re.search(_IMAGE, m) and re.search(r"(generate|create|make|draw)", m):
        return {"intent": "image", **flags}
    if re.search(_CHART, m):
        return {"intent": "visualization", **flags}
    if has_docs and re.search(_FILE, m):
        return {"intent": "file_analysis", **flags}
    if re.search(_PROV, m):
        return {"intent": "provenance", **flags}
    if re.search(_CALC, m) and flags["has_numbers"]:
        return {"intent": "calculation", **flags}
    if flags["wants_web"]:
        return {"intent": "research", **flags}
    return {"intent": "knowledge_qa", **flags}


FORCED_TOOL = {"document": "generate_document", "presentation": "generate_presentation", "spreadsheet": "generate_spreadsheet"}
FORCED_BY_INTENT = {"visualization": "generate_chart_or_table", "image": "generate_image"}

TOOL_INTENTS = {"artifact", "visualization", "image", "file_analysis", "calculation", "provenance"}

GROUPS_BY_INTENT = {
    "knowledge_qa": {"knowledge"},
    "research": {"knowledge", "web_research"},
    "artifact": {"knowledge", "artifact_creation", "visualization", "computation"},
    "visualization": {"knowledge", "visualization", "computation"},
    "image": {"visualization"},
    "calculation": {"computation"},
    "file_analysis": {"file_analysis", "computation", "visualization"},
    "provenance": {"knowledge_management"},
    "smalltalk": set(),
}


def select_groups(c: dict, allow_web: bool) -> list[str]:
    g = set(GROUPS_BY_INTENT.get(c["intent"], {"knowledge"}))
    if allow_web and (c["wants_web"] or c["wants_ingest"]):
        g |= {"web_research"}
    if c["wants_ingest"]:
        g |= {"knowledge_management"}
    return sorted(g)


# ---- state --------------------------------------------------------------------------------
def _keep(a, b):  # last write wins
    return b


class AgentState(TypedDict, total=False):
    request_id: str
    classification: dict
    groups: list[str]
    plan: list[dict]
    counters: dict
    queries_tried: list[str]
    evidence_sufficient: bool
    web_done: bool
    needs_compaction: bool
    status: str
    answer: str
    finish_reason: str | None
    validation: dict | None
    abstain: bool
    revise_problems: str
    notes: Annotated[list, operator.add]
    warnings: Annotated[list, operator.add]
    results: Annotated[list, operator.add]       # compact per-capability outcomes for the final prompt
    compaction_log: Annotated[list, operator.add]
    budget_log: Annotated[list, operator.add]
    node_log: Annotated[list, operator.add]


@dataclass
class Run:
    """Per-request live objects (not part of checkpointed state)."""
    req: AgentChatRequest
    user_id: str
    request_id: str
    evidence: EvidenceStore
    cap_ctx: CapabilityContext
    controller: ReasoningController
    budget: ContextBudget
    history: list = field(default_factory=list)
    trace: list[TraceRecord] = field(default_factory=list)
    decisions: list = field(default_factory=list)
    summary: EvidenceSummary | None = None
    summary_rendered: str = ""
    compact_render: bool = False
    history_cap: int | None = None
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0})
    artifacts: dict = field(default_factory=dict)
    manifest: dict | None = None
    pending: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)


class AgentRunner:
    def __init__(self, services, registry: CapabilityRegistry, adapter: RuntimeAdapter, inference_settings,
                 conv_store: ConversationStore | None = None, policy: ModePolicy | None = None):
        self.svc, self.reg, self.adapter, self.inf = services, registry, adapter, inference_settings
        self.s = services.settings
        self.conv, self.policy = conv_store, policy or ModePolicy()
        self.counter = services.counter

    # ================= public ===============================================================
    async def run(self, req: AgentChatRequest, user_id: str) -> AgentChatResponse:
        rid = str(uuid.uuid4())
        inf = self.inf
        cfg = BudgetConfig(model_max_len=inf.max_model_len, output_reserve=req.max_output_tokens or inf.max_output_tokens,
                           tool_reserve=self.s.tool_reserve_tokens, safety_margin=self.s.safety_margin_tokens,
                           compaction_ratio=self.s.evidence_compaction_ratio, evidence_budget=inf.evidence_budget())
        evidence = EvidenceStore()
        perms = {"read", "write", "generate"} | ({"web"} if req.allow_web else set())
        cctx = CapabilityContext(user_id=user_id, trace_id=rid, services=self.svc, evidence=evidence,
                                 permissions=frozenset(perms), document_scope=req.document_ids)
        run = Run(req=req, user_id=user_id, request_id=rid, evidence=evidence, cap_ctx=cctx,
                  controller=ReasoningController(self.adapter.caps, max_changes=self.s.max_mode_changes),
                  budget=ContextBudget(self.counter, cfg))
        conv_id = req.conversation_id or str(uuid.uuid4())
        run.conv_id = conv_id
        if self.conv is not None:
            try:
                ctx = await asyncio.to_thread(self.conv.load, user_id, conv_id, 20)
            except PermissionError as exc:
                raise InferenceError("Conversation not found", "conversation_forbidden", 404) from exc
            run.history = ctx.recent
        graph = self._build(run)
        state: AgentState = {"request_id": rid, "counters": {"tool_calls": 0, "retrieval_attempts": 0, "rewrites": 0,
                                                              "llm_steps": 0, "summarizations": 0, "answer_retries": 0},
                             "queries_tried": [], "plan": [], "status": "ok", "notes": [], "warnings": [], "results": [],
                             "compaction_log": [], "budget_log": [], "node_log": []}
        t0 = time.perf_counter()
        try:
            final = await graph.ainvoke(state, config={"configurable": {"thread_id": rid}, "recursion_limit": 80})
        except asyncio.CancelledError:
            raise
        except InferenceError:
            raise
        except Exception as exc:
            logger.exception("agent_graph_failed")
            final = {**state, "status": "failed", "answer": "The request could not be completed.",
                     "warnings": [f"graph error: {type(exc).__name__}"]}
        resp = self._response(run, final, conv_id, (time.perf_counter() - t0) * 1e3)
        if self.conv is not None and resp.status != "failed":
            try:
                await asyncio.to_thread(self.conv.save_turn, user_id, conv_id, req.message, resp.model_dump_json(),
                                        resp.answer, {})
            except Exception:
                logger.exception("persist_failed")
        return resp

    # ================= graph construction ===================================================
    def _build(self, run: Run):
        g = StateGraph(AgentState)

        def node(name, fn):
            async def wrapped(state):
                t0 = time.perf_counter()
                out = await fn(run, state)
                ms = round((time.perf_counter() - t0) * 1e3, 2)
                out = dict(out or {})
                out["node_log"] = [{"node": name, "ms": ms}]
                return out
            g.add_node(name, wrapped)

        node("classify_request", self.n_classify)
        node("configure_reasoning_mode", self.n_mode)
        node("plan_capabilities", self.n_plan)
        node("execute_capability", self.n_execute)
        node("check_context_budget", self.n_budget)
        node("summarize_if_required", self.n_summarize)
        node("evaluate_evidence", self.n_evaluate)
        node("generate_final_response", self.n_generate)
        node("validate_citations", self.n_validate)
        node("finalize", self.n_finalize)

        g.add_edge(START, "classify_request")
        g.add_edge("classify_request", "configure_reasoning_mode")
        g.add_edge("configure_reasoning_mode", "plan_capabilities")
        g.add_conditional_edges("plan_capabilities", lambda s: "execute_capability" if s["plan"] else "evaluate_evidence",
                                {"execute_capability": "execute_capability", "evaluate_evidence": "evaluate_evidence"})
        g.add_edge("execute_capability", "check_context_budget")
        g.add_conditional_edges("check_context_budget", lambda s: "summarize_if_required" if s.get("needs_compaction") else "evaluate_evidence",
                                {"summarize_if_required": "summarize_if_required", "evaluate_evidence": "evaluate_evidence"})
        g.add_edge("summarize_if_required", "evaluate_evidence")
        g.add_conditional_edges("evaluate_evidence", self._route_eval,
                                {"execute_capability": "execute_capability", "generate_final_response": "generate_final_response"})
        g.add_edge("generate_final_response", "validate_citations")
        g.add_conditional_edges("validate_citations", lambda s: "generate_final_response" if s.get("status") == "revise" else "finalize",
                                {"generate_final_response": "generate_final_response", "finalize": "finalize"})
        g.add_edge("finalize", END)
        return g.compile(checkpointer=InMemorySaver())

    @staticmethod
    def _route_eval(s):
        if s.get("status") == "budget_error":
            return "generate_final_response"
        return "execute_capability" if s.get("plan") else "generate_final_response"

    # ================= helpers ==============================================================
    async def _exec(self, run: Run, state, name, args, purpose) -> tuple:
        """Run one capability through the registry (validation, timeout, trace). Returns (result, counters)."""
        c = dict(state["counters"])
        if c["tool_calls"] >= self.s.max_tool_calls:
            return None, c
        c["tool_calls"] += 1
        res = await self.reg.execute(name, args, run.cap_ctx, purpose=purpose, sink=run.trace)
        return res, c

    def _record(self, name, res) -> dict:
        return {"tool": name, "status": res.status, "summary": (res.data or {}).get("summary") if res.data else (res.error.message if res.error else ""),
                "error": res.error.code.value if res.error else None, "extra": res.metadata.extra,
                "expression": (res.data or {}).get("expression") if name == "run_calculation" else None}

    # ================= nodes ================================================================
    async def n_classify(self, run: Run, state):
        c = classify(run.req.message, bool(run.req.document_ids), run.req.allow_web)
        groups = select_groups(c, run.req.allow_web)
        if c["intent"] in ("artifact", "visualization", "image", "file_analysis"):   # drafting calls get a larger output cap
            run.budget.cfg.output_reserve = max(run.budget.cfg.output_reserve, self.s.artifact_output_tokens)
        return {"classification": c, "groups": groups}

    async def n_mode(self, run: Run, state):
        c = state["classification"]
        if run.req.reasoning != "auto":
            mode, why = ReasoningMode(run.req.reasoning), "user preference"
        else:
            mode, why = self.policy.initial(c["intent"])
        run.decisions.append(run.controller.request(mode, why))
        return {}

    async def n_plan(self, run: Run, state):
        c, q = state["classification"], run.req.message
        plan, i = [], c["intent"]
        if i in ("knowledge_qa", "research", "artifact", "visualization"):
            plan.append({"cap": "search_knowledge_base", "args": {"query": q, "document_ids": run.req.document_ids}, "purpose": "retrieve private evidence"})
        if i == "research" or (i in ("knowledge_qa",) and c["wants_web"]):
            plan.append({"cap": "web_search", "args": {"query": q, "max_results": 4}, "purpose": "find external sources"})
        if i == "calculation":
            m = re.search(r"([-+*/().\d\s^%]{3,}|[a-z]+\([^)]*\))", q)
            expr = re.sub(r"\s+", " ", m.group(1)).replace("^", "**").strip() if m else ""
            if expr and re.search(r"\d", expr) and re.search(r"[-+*/%]|\*\*|\(", expr):
                plan.append({"cap": "run_calculation", "args": {"expression": expr}, "purpose": "deterministic calculation",
                             "fallback": {"cap": "@agent_loop", "args": {}, "purpose": "calculation needs tool selection"}})
            else:
                plan.append({"cap": "@agent_loop", "args": {}, "purpose": "choose and call computation tools"})
        elif i == "provenance":
            u = re.search(r"https?://\S+", q)
            plan.append({"cap": "@agent_loop", "args": {"url": u.group(0) if u else None}, "purpose": "inspect provenance"})
        elif i in ("artifact", "visualization", "image", "file_analysis"):
            force = FORCED_TOOL.get(c.get("artifact_kind")) if i == "artifact" else FORCED_BY_INTENT.get(i)
            plan.append({"cap": "@agent_loop", "args": {"force": force}, "purpose": f"produce {i} via tools"})
        return {"plan": plan}

    async def n_execute(self, run: Run, state):
        plan = list(state["plan"])
        if not plan:
            return {}
        step, plan = plan[0], plan[1:]
        c, notes, results, warns = dict(state["counters"]), [], [], []
        ev_sufficient = state.get("evidence_sufficient", False)
        web_done = state.get("web_done", False)
        tried = list(state.get("queries_tried", []))
        if step["cap"] == "@agent_loop":
            out = await self._agent_loop(run, state, c, step["args"].get("force"))
            return {"plan": plan, "counters": out["counters"], "results": out["results"], "warnings": out["warnings"]}
        run.cap_ctx.scratch["evidence_tokens_left"] = max(0, run.budget.evidence_limit() - run.budget.used_by["evidence"] - run.budget.used_by["evidence_summary"])
        res, c = await self._exec(run, state, step["cap"], step["args"], step["purpose"])
        if res is None:
            return {"plan": [], "counters": c, "warnings": ["tool-call limit reached; remaining steps skipped"], "status": "partial"}
        results.append(self._record(step["cap"], res))
        if not res.ok and step.get("fallback"):
            plan.insert(0, step["fallback"])
        upd = {"plan": plan, "counters": c, "results": results, "queries_tried": tried}
        if step["cap"] == "search_knowledge_base":
            c["retrieval_attempts"] += 1
            tried.append(step["args"]["query"])
            if res.ok and res.data:
                upd["evidence_sufficient"] = bool(res.data.get("evidence_sufficient")) or ev_sufficient
            if res.status in ("error", "timeout"):
                warns.append(f"private retrieval failed: {res.error.code.value}")
        elif step["cap"] == "web_search" and res.status == "success":
            for hit in res.data["results"][:2]:
                plan.append({"cap": "fetch_webpage", "args": {"url": hit["url"]}, "purpose": "read search result"})
            upd["web_done"] = True
        elif step["cap"] == "web_search":
            upd["web_done"] = True
            if res.status != "empty":
                warns.append(f"web search failed: {res.error.code.value if res.error else 'unknown'}")
        elif step["cap"] == "fetch_webpage" and res.status == "success":
            plan.insert(0, {"cap": "extract_web_evidence", "args": {"page_id": res.data["page_id"], "query": run.req.message},
                            "purpose": "extract relevant passages"})
            if state["classification"]["wants_ingest"]:
                plan.append({"cap": "propose_knowledge_ingestion", "args": {"page_id": res.data["page_id"],
                             "reason": f"requested by user: {run.req.message[:200]}"}, "purpose": "propose saving this source (needs approval)"})
        elif step["cap"] == "propose_knowledge_ingestion" and res.status == "pending":
            run.pending.append({"source_id": res.data["source_id"], "title": res.data["preview"].get("title"), "url": res.data["preview"].get("url"),
                                "trust": res.data["trust"]["level"], "approve": res.data["how_to_approve"]})
        elif step["cap"] in ("fetch_webpage", "extract_web_evidence") and not res.ok:
            warns.append(f"{step['cap']} failed: {res.error.code.value}")
        upd["plan"] = plan
        upd["warnings"] = warns
        return upd

    # ---- bounded LLM tool loop (artifact / visualization / computation / file analysis) ---------
    async def _agent_loop(self, run: Run, state, counters, force: str | None = None) -> dict:
        c = dict(counters)
        groups = set(state["groups"])
        specs = self.reg.specs_for(groups)
        results, warns = [], []
        if not specs:
            return {"counters": c, "results": results, "warnings": ["no tools available for this task"]}
        ev_text = self._evidence_text(run)
        sys = SystemMessage(prompts.AGENT_TOOLS)
        fmt = state["classification"].get("format")
        msgs = [sys, HumanMessage((f"EVIDENCE:\n{ev_text}\n\n" if ev_text else "") + f"TASK: {run.req.message}" +
                                  (f"\nRequested output format: {fmt}." if fmt else "") +
                                  (f"\nUploaded file document_ids: {run.req.document_ids}" if run.req.document_ids else ""))]
        run.budget.set("tools", run.budget.tools_tokens(specs))
        for step in range(self.s.max_llm_steps):
            if c["tool_calls"] >= self.s.max_tool_calls:
                warns.append("tool-call limit reached")
                break
            d = run.decisions[-1]
            cap = self.s.artifact_output_tokens
            llm = self.adapter.build_llm(d, max_tokens=cap, cap=cap)
            # The target capability is known for artifact/chart/image requests: force that call on the first
            # step instead of hoping a small model chooses to call it (live runs showed it sometimes does not).
            choice = force if (step == 0 and force and any(sp["function"]["name"] == force for sp in specs)) else None
            try:
                shrinks = 0
                while True:
                    try:
                        ai = await (llm.bind_tools(specs, tool_choice=choice) if choice else llm.bind_tools(specs)).ainvoke(msgs)
                        break
                    except Exception as exc:
                        # prompt + max_tokens overran the window by the template/tool-schema overhead our counter cannot see;
                        # the server does not say by how much, so shrink the output cap geometrically (bounded).
                        smaller = reduced_output_cap(exc, cap) if shrinks < 4 else None
                        if smaller is not None and smaller < cap:
                            shrinks += 1
                            warns.append(f"output cap reduced from {cap} to {smaller} tokens to fit the context window")
                            cap = smaller
                            llm = self.adapter.build_llm(d, max_tokens=cap, cap=cap)
                        elif choice is not None and "BadRequest" in type(exc).__name__:
                            warns.append("runtime rejected a forced tool choice; fell back to automatic tool choice")
                            choice = None
                        else:
                            raise
            except Exception as exc:
                raise map_error(exc) from exc
            c["llm_steps"] += 1
            self._usage(run, ai)
            calls = list(ai.tool_calls or [])
            if not calls and force and step == 0 and (getattr(ai, "invalid_tool_calls", None) or
                                                      (ai.response_metadata or {}).get("finish_reason") == "length"):
                # A malformed/truncated tool call (thinking can eat the output budget): retry once without thinking.
                warns.append("tool call was malformed or truncated; retried once without thinking")
                d2 = run.controller.request(ReasoningMode.non_thinking, "tool call truncated or malformed")
                run.decisions.append(d2)
                if d2.planned == ReasoningMode.non_thinking and d2.planned != d.planned:
                    llm = self.adapter.build_llm(d2, max_tokens=cap, cap=cap)
                    ai = await llm.bind_tools(specs, tool_choice=choice).ainvoke(msgs) if choice else await llm.bind_tools(specs).ainvoke(msgs)
                    c["llm_steps"] += 1
                    self._usage(run, ai)
                    calls = list(ai.tool_calls or [])
            if not calls:
                if force and step == 0 and not run.artifacts:
                    warns.append("the model did not produce the requested tool call")
                break
            msgs.append(AIMessage(content=ai.content or "", tool_calls=calls))
            for tc in calls:
                if c["tool_calls"] >= self.s.max_tool_calls:
                    msgs.append(ToolMessage(content="Tool-call limit reached.", tool_call_id=tc.get("id") or new_id()))
                    continue
                c["tool_calls"] += 1
                name = tc.get("name", "")
                cap = self.reg.get(name)
                if cap is None or cap.group not in groups or not cap.llm_exposed:   # never trust the model's choice
                    res_txt = f"Tool '{name}' is not available for this task."
                    results.append({"tool": name, "status": "error", "summary": "not available", "error": "not_found", "extra": {}})
                else:
                    run.cap_ctx.scratch["evidence_tokens_left"] = max(0, run.budget.evidence_limit() - run.budget.used_by["evidence"])
                    targs = dict(tc.get("args") or {})
                    if fmt and name in ("generate_document", "generate_spreadsheet") and targs.get("format") != fmt \
                            and fmt in (("docx", "pdf") if name == "generate_document" else ("xlsx", "csv")):
                        targs["format"] = fmt          # an explicit user format request overrides the model's default
                    res = await self.reg.execute(name, targs, run.cap_ctx, purpose=f"agent step {step + 1}", sink=run.trace)
                    results.append(self._record(name, res))
                    res_txt = self._tool_message(run, name, res)
                msgs.append(ToolMessage(content=res_txt, tool_call_id=tc.get("id") or new_id()))
        else:
            warns.append("agent loop step limit reached")
        return {"counters": c, "results": results, "warnings": warns}

    def _tool_message(self, run: Run, name: str, res) -> str:
        """What the model sees of a tool result: small and without bulky payloads."""
        if not res.ok:
            return f"ERROR {res.error.code.value}: {res.error.message}"
        d = res.data or {}
        if "artifact" in d and d["artifact"]:
            a = d["artifact"]
            run.artifacts[a["artifact_id"]] = a
            return json.dumps({"artifact_id": a["artifact_id"], "filename": a["filename"], "status": a["status"], "format": a["format"]})
        if name in ("search_knowledge_base", "expand_document_context"):
            return json.dumps({"added_evidence_refs": [i["ref"] for i in d.get("items", [])]})
        if name == "extract_web_evidence":
            return json.dumps({"added_evidence_refs": [i["ref"] for i in d.get("evidence", [])]})
        if name == "compose_multimodal_response":
            run.manifest = d
            return json.dumps({"blocks": len(d.get("blocks", []))})
        return json.dumps(d, default=str)[:1500]

    def _usage(self, run: Run, ai):
        u = getattr(ai, "usage_metadata", None) or {}
        run.usage["llm_calls"] += 1
        run.usage["input_tokens"] += u.get("input_tokens", 0) or 0
        run.usage["output_tokens"] += u.get("output_tokens", 0) or 0

    # ---- budget -------------------------------------------------------------------------
    def _evidence_text(self, run: Run) -> str:
        parts = []
        if run.summary_rendered:
            parts.append(prompts.SUMMARY_NOTE + "\n" + run.summary_rendered)
        parts += [i.render(compact=run.compact_render) for i in run.evidence.items.values() if i.status == "raw"]
        return "\n".join(parts)

    def _history_msgs(self, run: Run):
        room = run.history_cap if run.history_cap is not None else self.inf.history_tokens
        kept = []
        for m in reversed(run.history):
            n = run.budget.message_tokens(m["content"])
            if n > room:
                break
            room -= n
            kept.append(m)
        return [(HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"]) for m in reversed(kept)], len(kept)

    def _account(self, run: Run, state):
        b = run.budget
        sys_tokens = b.message_tokens(prompts.GROUND)
        hist, _ = self._history_msgs(run)
        b.set("system", sys_tokens)
        b.set("history", sum(b.message_tokens(m.content) for m in hist))
        b.set("request", b.message_tokens(run.req.message))
        raw = sum(b.tokens(i.render(compact=run.compact_render)) for i in run.evidence.items.values() if i.status == "raw")
        b.set("evidence", raw)
        b.set("evidence_summary", b.tokens(run.summary_rendered) if run.summary_rendered else 0)

    async def n_budget(self, run: Run, state):
        self._account(run, state)
        d = run.budget.check()
        snap = run.budget.snapshot("after_execute")
        snap["compaction_needed"] = d.compaction_needed
        return {"needs_compaction": d.compaction_needed, "budget_log": [snap]}

    async def n_summarize(self, run: Run, state):
        """Compaction ladder; each rung is tried only if the previous did not restore headroom."""
        b, log, warns, c = run.budget, [], [], dict(state["counters"])

        def fits():
            self._account(run, state)
            return not b.check().compaction_needed

        def rung(name, before, **kw):
            self._account(run, state)
            log.append({"rung": name, "tokens_before": before, "tokens_after": b.used_by["evidence"] + b.used_by["evidence_summary"], **kw})

        before = b.used_by["evidence"] + b.used_by["evidence_summary"]
        run.compact_render = True                                   # 1. drop metadata formatting
        if fits():
            rung("compact_formatting", before)
            return {"needs_compaction": False, "compaction_log": log, "budget_log": [b.snapshot("after_compaction")]}
        rung("compact_formatting", before, restored=False)
        # 2. obsolete intermediate messages: tool outputs are never replayed into the final prompt (nothing to drop)
        raw = sorted(run.evidence.raw(), key=lambda i: -(i.colbert_score if i.colbert_score is not None else 0))
        keep_n = 2
        if c["summarizations"] < 2 and len(raw) > keep_n:           # 3. summarize evidence, keeping provenance
            refs = [i.ref for i in raw[keep_n:]]
            target = max(128, min(self.s.summary_target_tokens, b.evidence_limit() // 2))
            before = b.used_by["evidence"] + b.used_by["evidence_summary"]
            res, c = await self._exec(run, state, "summarize_evidence",
                                      {"query": run.req.message, "task": "answer the user's question",
                                       "refs": refs, "target_tokens": target,
                                       "prior_summaries": [run.summary.model_dump()] if run.summary else []},
                                      "compress evidence to fit context budget")
            c["summarizations"] = c.get("summarizations", 0) + 1
            if res is not None and res.ok and res.data and res.data.get("rendered"):
                out = res.data
                run.summary = EvidenceSummary.model_validate(out["summary"])
                run.summary_rendered = out["rendered"]
                covered = [r for r in refs if r in out["sources"]]
                run.evidence.mark_summarized(covered)                # originals stay in the store
                if out["coverage"].get("uncovered_refs"):
                    warns.append(f"{len(out['coverage']['uncovered_refs'])} evidence items could not be summarized and stay raw")
                if out["method"] != "llm":
                    warns.append("evidence summarized with the extractive fallback")
                rung("summarize_evidence", before, covered=len(covered), method=out["method"], status=out["status"],
                     contradictions=len(run.summary.contradictions))
                if run.summary.contradictions:                         # contradictory evidence justifies thinking mode
                    esc = self.policy.escalate(contradictions=len(run.summary.contradictions))
                    if esc:
                        run.decisions.append(run.controller.request(esc[0], esc[1]))
                if fits():
                    return {"needs_compaction": False, "counters": c, "compaction_log": log, "warnings": warns,
                            "budget_log": [b.snapshot("after_compaction")]}
            else:
                err = res.error.code.value if res is not None and res.error else "skipped"
                warns.append(f"evidence summarization unavailable ({err})")
                rung("summarize_evidence", before, failed=err)
        run.history_cap = (run.history_cap if run.history_cap is not None else self.inf.history_tokens) // 2   # 4. older history
        before = b.used_by["history"]
        if fits():
            log.append({"rung": "trim_history", "tokens_before": before, "tokens_after": b.used_by["history"]})
            return {"needs_compaction": False, "counters": c, "compaction_log": log, "warnings": warns,
                    "budget_log": [b.snapshot("after_compaction")]}
        omitted = []                                               # 5. reduce breadth: lowest-ranked raw items leave the prompt
        for it in reversed(sorted(run.evidence.raw(), key=lambda i: -(i.colbert_score if i.colbert_score is not None else 0))):
            if fits() or len(run.evidence.raw()) <= 1:
                break
            run.evidence.mark_omitted([it.ref])
            omitted.append(it.ref)
        fits()
        log.append({"rung": "reduce_breadth", "omitted_refs": omitted, "fits": b.check().ok})
        if omitted:
            warns.append(f"{len(omitted)} lower-ranked evidence items were left out of the prompt for budget reasons: {omitted}")
        out = {"needs_compaction": False, "counters": c, "compaction_log": log, "warnings": warns, "budget_log": [b.snapshot("after_compaction")]}
        if not b.check().ok:
            out.update(status="budget_error", warnings=warns + ["context budget cannot be met even after compaction; narrow the question"])
        return out

    async def n_evaluate(self, run: Run, state):
        """Decide: run more capabilities, retry retrieval with a rewritten query, escalate to web, or answer."""
        if state.get("status") == "budget_error":
            return {}
        plan, c, i = list(state["plan"]), dict(state["counters"]), state["classification"]["intent"]
        if plan:
            return {}
        needs_evidence = i in ("knowledge_qa", "research")
        have = any(e.status != "omitted" for e in run.evidence.items.values())
        sufficient = state.get("evidence_sufficient", False)
        if needs_evidence and (not have or not sufficient) and c["retrieval_attempts"] < self.s.max_retrieval_attempts and c["tool_calls"] < self.s.max_tool_calls:
            rw, c = await self._exec(run, state, "rewrite_search_query",
                                     {"query": run.req.message, "previous_queries": state.get("queries_tried", [])}, "rewrite query for a retry")
            if rw is not None and rw.status == "success" and rw.data["variants"]:
                c["rewrites"] += 1
                return {"plan": [{"cap": "search_knowledge_base", "args": {"query": rw.data["variants"][0], "document_ids": run.req.document_ids},
                                  "purpose": f"retry retrieval ({rw.data['strategy'][rw.data['variants'][0]]})"}],
                        "counters": c, "notes": ["retrieval retry with rewritten query"]}
        if needs_evidence and not have and state["classification"]["wants_web"] is False and run.req.allow_web and not state.get("web_done") \
                and c["tool_calls"] < self.s.max_tool_calls:
            return {"plan": [{"cap": "web_search", "args": {"query": run.req.message, "max_results": 4},
                              "purpose": "private evidence insufficient; searching the web"}],
                    "notes": ["escalated to web research"]}
        return {"counters": c}

    # ---- final answer -------------------------------------------------------------------
    async def n_generate(self, run: Run, state):
        c, i = dict(state["counters"]), state["classification"]["intent"]
        results = state.get("results", [])
        ok_results = [r for r in results if r["status"] in ("success", "pending")]
        if i == "smalltalk":
            return {"answer": SMALLTALK_REPLY, "finish_reason": "deterministic", "abstain": False}
        # Pure deterministic answers need no model call.
        calc = next((r for r in results if r["tool"] == "run_calculation" and r["status"] == "success"), None)
        if i == "calculation" and calc and not run.evidence.items:
            return {"answer": f"{calc['expression']} {calc['summary']}" if calc.get("expression") else calc["summary"],
                    "finish_reason": "deterministic", "abstain": False}
        have_evidence = any(e.status != "omitted" for e in run.evidence.items.values())
        if i in ("knowledge_qa", "research") and not have_evidence:
            return {"answer": prompts.ABSTAIN, "finish_reason": "no_evidence", "abstain": True, "status": "abstained" if state.get("status") == "ok" else state["status"]}
        if i == "provenance" and not ok_results:
            return {"answer": "No provenance record was found for that source.", "finish_reason": "no_results", "abstain": True}
        if state.get("status") == "budget_error":
            return {"answer": "The collected evidence is too large to fit the model's context even after compaction. Please narrow the question or restrict it to specific documents.",
                    "finish_reason": "budget_error", "abstain": True}
        self._account(run, state)
        ev = self._evidence_text(run)
        facts = self._results_text(run, results)
        msgs = [SystemMessage(prompts.GROUND), *self._history_msgs(run)[0],
                HumanMessage((f"EVIDENCE:\n{ev}\n" if ev else "EVIDENCE: (none)\n")
                             + (f"\nINTERNAL NOTES (background for you; do not quote or list them):\n{facts}\n" if facts else "")
                             + f"\nREQUEST: {run.req.message}"
                             + ("\nReply to the user in 2-4 plain sentences about what was done and how to get it." if i in TOOL_INTENTS else ""))]
        if state.get("status") == "revise":
            msgs.append(HumanMessage(prompts.REVISE.format(problems=state.get("revise_problems", "unsupported claims"))))
        d = run.decisions[-1]
        if i in TOOL_INTENTS and results and d.planned == ReasoningMode.thinking:   # the work is done; the closing message needs no deliberation
            d = run.controller.request(ReasoningMode.non_thinking, "tool work finished; reporting the outcome")
            run.decisions.append(d)
        max_tokens = min(run.req.max_output_tokens or self.inf.max_output_tokens, self.inf.max_output_tokens)
        try:
            ai = await self.adapter.build_llm(d, max_tokens=max_tokens).ainvoke(msgs)
            self._usage(run, ai)
            text = _THINK.sub("", ai.content if isinstance(ai.content, str) else "").strip()
            finish = (ai.response_metadata or {}).get("finish_reason")
            if not text and finish == "length" and d.planned == ReasoningMode.thinking:   # thinking ate the budget: retry once, non-thinking
                d2 = run.controller.request(ReasoningMode.non_thinking, "thinking exhausted the output budget")
                run.decisions.append(d2)
                if d2.planned == ReasoningMode.non_thinking:
                    ai = await self.adapter.build_llm(d2, max_tokens=max_tokens).ainvoke(msgs)
                    self._usage(run, ai)
                    text, finish = _THINK.sub("", ai.content if isinstance(ai.content, str) else "").strip(), (ai.response_metadata or {}).get("finish_reason")
        except Exception as exc:
            raise map_error(exc) from exc
        if i in TOOL_INTENTS and (not text or any(r["tool"] in text for r in results) or "INTERNAL NOTES" in text):
            text, finish = self._template_answer(run, results), "template_fallback"   # model echoed internal notes
        if not text:
            raise InferenceError("Model returned no answer text", "llm_empty", 502)
        text = normalize_citations(text)
        c["llm_steps"] += 1
        return {"answer": text, "finish_reason": finish, "counters": c, "abstain": False, "status": "ok" if state.get("status") in ("revise", None) else state["status"]}

    def _results_text(self, run: Run, results) -> str:
        """Plain-sentence notes about work done (never bracketed tool lines the model could echo)."""
        lines = []
        for r in results:
            if r["tool"] in ("search_knowledge_base", "web_search", "fetch_webpage", "extract_web_evidence", "rewrite_search_query",
                             "summarize_evidence", "compose_multimodal_response"):
                continue
            what = r["tool"].replace("_", " ")
            if r["status"] in ("success", "pending"):
                lines.append(f"The {what} step succeeded" + (f" ({r['summary']})." if r["summary"] else "."))
            else:
                lines.append(f"The {what} step did not succeed ({r['error']}); tell the user plainly.")
        for a in run.artifacts.values():
            lines.append(f"A {a['format']} file named '{a['filename']}' was created and verified; the user can download it from the artifacts list.")
        for p in run.pending:
            lines.append(f"The web source '{p['title']}' ({p['url']}) was NOT saved; it is waiting for the user's explicit approval.")
        return "\n".join(lines)

    def _template_answer(self, run: Run, results) -> str:
        """Deterministic reply used when the model echoes internal notes or returns nothing after tool work."""
        parts = []
        for a in run.artifacts.values():
            parts.append(f"I created {a['filename']} ({a['format']}).")
        for r in results:
            if r["status"] not in ("success", "pending") and r["tool"] not in ("search_knowledge_base", "web_search", "fetch_webpage",
                                                                                 "extract_web_evidence", "rewrite_search_query", "summarize_evidence"):
                parts.append(f"The {r['tool'].replace('_', ' ')} step failed ({r['error']}).")
        for p in run.pending:
            parts.append(f"I prepared '{p['title']}' for saving but it is not saved until you approve it.")
        return " ".join(parts) or "I could not complete that request with the available tools."

    # ---- validation ---------------------------------------------------------------------
    async def n_validate(self, run: Run, state):
        ans = state.get("answer", "")
        if state.get("finish_reason") in ("deterministic", "no_evidence", "no_results", "budget_error") or not _MARK.search(ans) and not run.evidence.items:
            return {"validation": None, "status": state.get("status", "ok") if state.get("status") != "revise" else "ok"}
        user_ctx = run.req.message + "\n" + "\n".join(str(r.get("summary") or "") for r in state.get("results", []) if r["status"] == "success")
        mode = "tool_task" if state["classification"]["intent"] in TOOL_INTENTS else "strict"
        res = await self.reg.execute("validate_citations", {"answer": ans, "user_context": user_ctx[:8000], "mode": mode}, run.cap_ctx,
                                     purpose="verify citations before returning", sink=run.trace)
        if not res.ok:
            return {"validation": {"status": "unavailable", "error": res.error.code.value}, "warnings": ["citation validation unavailable"],
                    "status": "partial" if state.get("status") in ("ok", "revise") else state.get("status")}
        v = res.data
        c = dict(state["counters"])
        cov = v["coverage"]
        mostly_uncited = cov["uncited_factual_sentences"] + cov["cited_sentences"] >= 2 and cov["citation_coverage"] < 0.5
        if (v["status"] == "fail" or mostly_uncited) and c["answer_retries"] < 1:
            c["answer_retries"] += 1
            problems = [f"unknown labels {v['invalid_refs']}"] * bool(v["invalid_refs"]) + \
                       [f"'{x['sentence'][:80]}' ({x['reason'] or x['verdict']})" for x in v["claims"] if x["verdict"] in ("unsupported", "invalid_reference")][:3]
            if mostly_uncited:
                problems.append(f"{cov['uncited_factual_sentences']} factual sentence(s) have no [S#]/[W#] citation")
            return {"validation": v, "counters": c, "status": "revise", "revise_problems": "; ".join(problems)}
        if v["invalid_refs"]:                                      # never let an unverifiable reference reach the client
            for r in v["invalid_refs"]:
                ans = ans.replace(f"[{r}]", "")
        out = {"answer": ans, "validation": v, "status": state.get("status", "ok") if state.get("status") != "revise" else "ok"}
        if v["status"] in ("fail", "warn") or mostly_uncited:
            out["warnings"] = [f"citation validation {v['status']}: {v['coverage']['unsupported']} unsupported, "
                               f"{len(v['uncited_sentences'])} uncited sentences"]
            if v["status"] == "fail":
                out["answer"] = ans + "\n\nNote: some statements above could not be verified against the cited sources."
                out["status"] = "partial"
        return out

    async def n_finalize(self, run: Run, state):
        st = state.get("status", "ok")
        if st in ("revise", "budget_error"):
            st = "partial"
        if state.get("abstain") and st == "ok":
            st = "abstained"
        if run.pending and st in ("ok", "partial"):
            st = "awaiting_approval" if st == "ok" else st
        return {"status": st}

    # ================= response ==============================================================
    def _response(self, run: Run, final, conv_id, total_ms) -> AgentChatResponse:
        ans = final.get("answer", "")
        refs = [r for r in dict.fromkeys(_MARK.findall(ans)) if run.evidence.get(r)]
        cits = []
        for r in refs:
            it = run.evidence.get(r)
            cits.append(CitationOut(ref=r, kind=it.kind, document_id=it.document_id, chunk_id=it.chunk_id, filename=it.filename,
                                    page_start=it.page_start, page_end=it.page_end, url=it.url, passage_id=it.passage_id, title=it.title))
        trace = [AgentTraceEntry(step=i + 1, tool_name=t.tool_name, capability_group=t.group, purpose=t.purpose, status=t.status,
                                 input=t.input_summary, output_summary=t.output_summary, duration_ms=t.duration_ms,
                                 execution_id=t.execution_id, output_chars=t.output_chars, error_code=t.error_code,
                                 retry_count=t.retry_count, extra=t.extra) for i, t in enumerate(run.trace)]
        status = final.get("status", "failed")
        if status not in ("ok", "partial", "abstained", "awaiting_approval", "failed"):
            status = "partial"
        run.budget.snapshot("final")
        return AgentChatResponse(
            answer=ans, status=status, citations=cits, tool_trace=trace, artifacts=list(run.artifacts.values()), manifest=run.manifest,
            pending_approvals=run.pending, validation=final.get("validation"),
            reasoning=[d.as_dict() for d in run.decisions],
            context={"budget": final.get("budget_log", []), "compaction": final.get("compaction_log", []),
                     "summary_in_prompt": bool(run.summary_rendered)},
            warnings=list(dict.fromkeys(final.get("warnings", []))),
            metadata={"request_id": run.request_id, "conversation_id": conv_id, "model": self.inf.model,
                      "intent": final.get("classification", {}).get("intent"), "groups": final.get("groups", []),
                      "counters": final.get("counters", {}), "usage": run.usage, "total_ms": round(total_ms, 1),
                      "nodes": final.get("node_log", []), "finish_reason": final.get("finish_reason"),
                      "mode_changes": run.controller.changes,
                      "token_counts": "exact" if run.budget.exact else "estimate"})
