import asyncio
import tempfile
import unittest
from pathlib import Path

import httpx

from app.agent.graph import classify, select_groups
from app.agent.schemas import AgentChatRequest
from app.agent.summarize import EvidenceSummarizer
from app.inference.memory import SqliteConversationStore
from app.inference.schemas import InferenceError
from tests.agent_helpers import (CharCounter, FakeChat, FakeRetrieval, ai, asettings, cand, isettings, make_runner, scripted_llm, tc)
from tests.test_agent_capabilities import HTML, fetcher


def run(coro):
    return asyncio.run(coro)


def chat(runner, msg, **kw):
    return run(runner.run(AgentChatRequest(message=msg, **kw), "alice"))


class ClassifyTests(unittest.TestCase):
    def test_routing(self):
        cases = [("What was revenue in 2023?", False, False, "knowledge_qa", {"knowledge"}),
                 ("Create a PowerPoint presentation about Q3", False, False, "artifact", {"knowledge", "artifact_creation", "visualization", "computation"}),
                 ("calculate 12*(3+4)", False, False, "calculation", {"computation"}),
                 ("search the web for latest inflation news", False, True, "research", {"knowledge", "web_research"}),
                 ("search the web for latest inflation news", False, False, "knowledge_qa", {"knowledge"}),   # web is opt-in
                 ("analyze the uploaded csv columns", True, False, "file_analysis", {"file_analysis", "computation", "visualization"}),
                 ("plot a bar chart of sales", False, False, "visualization", {"knowledge", "visualization", "computation"}),
                 ("generate an image of a lighthouse", False, False, "image", {"visualization"})]
        for msg, docs, web, intent, groups in cases:
            c = classify(msg, docs, web)
            self.assertEqual(c["intent"], intent, msg)
            self.assertEqual(set(select_groups(c, web)), groups, msg)

    def test_ingest_intent_needs_web_opt_in(self):
        self.assertTrue(classify("search the web and save this to the knowledge base", False, True)["wants_ingest"])
        self.assertFalse(classify("search the web and save this to the knowledge base", False, False)["wants_ingest"])


class GraphQATests(unittest.TestCase):
    def test_grounded_answer_with_real_ids_and_non_thinking_mode_reaching_client(self):
        r, svc = make_runner([ai("Revenue was 5 million dollars [S1].")],
                             retrieval=FakeRetrieval([cand(1, "Revenue was 5 million dollars in 2023.")]))
        out = chat(r, "What was the revenue?")
        self.assertEqual(out.status, "ok")
        self.assertEqual([(c.ref, c.kind, c.document_id, c.chunk_id) for c in out.citations], [("S1", "private", "d1", "c1")])
        self.assertEqual(out.validation["status"], "pass")
        self.assertEqual(out.reasoning[0]["requested"], "non_thinking")
        self.assertEqual(out.reasoning[0]["applied"], "non_thinking")
        self.assertEqual(FakeChat.log[0]["kw"]["extra_body"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(len(FakeChat.log), 1)                                    # one LLM call, no planning/classify calls
        names = [t.tool_name for t in out.tool_trace]
        self.assertEqual(names, ["search_knowledge_base", "validate_citations"])
        self.assertTrue(all(t.execution_id and t.duration_ms >= 0 for t in out.tool_trace))
        self.assertEqual(out.metadata["intent"], "knowledge_qa")
        self.assertEqual(out.metadata["token_counts"], "exact")

    def test_no_evidence_abstains_without_calling_the_model_and_retries_are_bounded(self):
        fr = FakeRetrieval([])
        r, svc = make_runner([], retrieval=fr)
        out = chat(r, "What was the revenue of Acme Corp in 2023 and the profit margin?")
        self.assertEqual(out.status, "abstained")
        self.assertEqual(FakeChat.log, [])
        self.assertLessEqual(len(fr.calls), svc.settings.max_retrieval_attempts)
        self.assertEqual(len(fr.calls), 2)                                          # original + exactly one rewrite
        self.assertEqual(out.citations, [])
        self.assertTrue(any(t.tool_name == "rewrite_search_query" for t in out.tool_trace))

    def test_retrieval_failure_is_a_partial_failure_not_a_crash(self):
        from app.retrieval.schema import RetrievalError
        r, svc = make_runner([], retrieval=FakeRetrieval(fail=RetrievalError("x", "hybrid_retrieval")))
        out = chat(r, "What was the revenue?")
        self.assertEqual(out.status, "abstained")
        self.assertTrue(any("retrieval failed" in w for w in out.warnings))
        self.assertTrue(any(t.status == "error" for t in out.tool_trace))

    def test_invalid_citation_triggers_one_revision_then_is_stripped(self):
        r, _ = make_runner([ai("Revenue was 5 million dollars [S7]."), ai("Revenue was 5 million dollars [S1].")],
                           retrieval=FakeRetrieval([cand(1, "Revenue was 5 million dollars in 2023.")]))
        out = chat(r, "What was the revenue?")
        self.assertEqual((out.status, [c.ref for c in out.citations]), ("ok", ["S1"]))
        self.assertEqual(len(FakeChat.log), 2)
        self.assertIn("citation problems", FakeChat.log[1]["messages"][-1].content)
        r, _ = make_runner([ai("Revenue was 9 million dollars [S1]."), ai("Revenue was 9 million dollars [S1].")],
                           retrieval=FakeRetrieval([cand(1, "Revenue was 5 million dollars in 2023.")]))
        out = chat(r, "What was the revenue?")
        self.assertEqual(len(FakeChat.log), 2)                                      # bounded: one revision only
        self.assertEqual(out.status, "partial")
        self.assertIn("could not be verified", out.answer)
        self.assertEqual(out.validation["status"], "fail")

    def test_model_failure_surfaces_as_controlled_error(self):
        import openai
        r, _ = make_runner([openai.APIConnectionError(request=httpx.Request("POST", "http://x"))], retrieval=FakeRetrieval([cand(1)]))
        with self.assertRaises(InferenceError) as cm:
            chat(r, "What is alpha?")
        self.assertEqual(cm.exception.code, "llm_unavailable")

    def test_history_is_persisted_and_scoped_to_user(self):
        store = SqliteConversationStore(Path(tempfile.mkdtemp()) / "c.db")
        r, _ = make_runner([ai("Revenue was 5 million dollars [S1]."), ai("Still 5 million dollars [S1].")], conv_store=store,
                           retrieval=FakeRetrieval([cand(1, "Revenue was 5 million dollars in 2023.")]))
        a = chat(r, "What was the revenue?")
        cid = a.metadata["conversation_id"]
        chat(r, "And again?", conversation_id=cid)
        self.assertIn("What was the revenue?", [m.content for m in FakeChat.log[1]["messages"]])
        with self.assertRaises(InferenceError):
            run(r.run(AgentChatRequest(message="x", conversation_id=cid), "mallory"))

    def test_thinking_exhausting_budget_retries_once_without_thinking(self):
        r, _ = make_runner([ai("", finish="length"), ai("Revenue was 5 million dollars [S1].")],
                           retrieval=FakeRetrieval([cand(1, "Revenue was 5 million dollars in 2023.")]))
        out = chat(r, "What was the revenue?", reasoning="thinking")
        self.assertEqual(out.status, "ok")
        self.assertEqual([FakeChat.log[0]["kw"]["extra_body"]["chat_template_kwargs"], FakeChat.log[1]["kw"]["extra_body"]["chat_template_kwargs"]],
                         [{"enable_thinking": True}, {"enable_thinking": False}])
        self.assertEqual([d["applied"] for d in out.reasoning], ["thinking", "non_thinking"])


class ReasoningModeTests(unittest.TestCase):
    def test_artifact_task_requests_and_applies_thinking(self):
        r, _ = make_runner([ai("no tool needed"), ai("Done [S1].")], retrieval=FakeRetrieval([cand(1, "Done.")]))
        out = chat(r, "Create a PDF report about revenue")
        self.assertEqual(out.reasoning[0]["requested"], "thinking")
        self.assertEqual(out.reasoning[0]["applied"], "thinking")

    def test_unsupported_runtime_is_reported_truthfully(self):
        r, _ = make_runner([ai("Revenue was 5 million dollars [S1].")], model="Qwen/Qwen3-4B-Instruct-2507",
                           retrieval=FakeRetrieval([cand(1, "Revenue was 5 million dollars in 2023.")]))
        out = chat(r, "What was the revenue?", reasoning="thinking")
        d = out.reasoning[0]
        self.assertEqual((d["requested"], d["applied"], d["status"]), ("thinking", "non_thinking", "fallback_fixed_mode"))
        self.assertNotIn("chat_template_kwargs", FakeChat.log[0]["kw"]["extra_body"])
        r, _ = make_runner([ai("Revenue was 5 million dollars [S1].")], model="vendor/unknown-model",
                           retrieval=FakeRetrieval([cand(1, "Revenue was 5 million dollars in 2023.")]))
        d = chat(r, "What was the revenue?").reasoning[0]
        self.assertEqual((d["status"], d["applied"]), ("unsupported", None))


class CalcAndArtifactTests(unittest.TestCase):
    def test_calculation_is_deterministic_without_llm(self):
        r, _ = make_runner([])
        out = chat(r, "calculate 12*(3+4)")
        self.assertEqual(out.answer, "12*(3+4) = 84")
        self.assertEqual(FakeChat.log, [])
        self.assertEqual(out.tool_trace[0].tool_name, "run_calculation")

    def test_failed_calculation_falls_back_to_tool_loop(self):
        r, _ = make_runner([ai("", [tc("run_calculation", {"expression": "0.15*200"})]), ai("It is 30.")])
        out = chat(r, "what is 15% of 200?")
        self.assertEqual([t.tool_name for t in out.tool_trace][:2], ["run_calculation", "run_calculation"])
        self.assertEqual(out.tool_trace[0].status, "error")
        self.assertEqual(out.tool_trace[1].status, "success")
        self.assertEqual(FakeChat.log[0]["tools"], ["run_calculation"])            # task-filtered tool set

    def test_pdf_artifact_flow_is_verified_and_tools_are_filtered(self):
        script = [ai("", [tc("generate_document", {"title": "Revenue Report", "format": "pdf",
                                                    "sections": [{"heading": "Findings", "paragraphs": ["Revenue was 5 million."], "source_refs": ["S1"]}]})]),
                  ai("done"), ai("I created the report.")]
        r, svc = make_runner(script, retrieval=FakeRetrieval([cand(1, "Revenue was 5 million dollars in 2023.")]))
        out = chat(r, "Create a PDF report about revenue")
        self.assertEqual(out.status, "ok")
        self.assertEqual(len(out.artifacts), 1)
        art = out.artifacts[0]
        _, path = svc.artifact_store.resolve(art["artifact_id"], "alice")      # file really exists for this owner
        self.assertTrue(path.read_bytes().startswith(b"%PDF"))
        self.assertNotIn("web_search", FakeChat.log[0]["tools"])
        self.assertNotIn("ingest_approved_source", FakeChat.log[0]["tools"])
        self.assertNotIn("summarize_evidence", FakeChat.log[0]["tools"])
        self.assertIn("generate_document", FakeChat.log[0]["tools"])
        self.assertIn("created and verified", FakeChat.log[-1]["messages"][-1].content)

    def test_closing_message_after_tool_work_drops_to_non_thinking_and_bounds_mode_changes(self):
        script = [ai("", [tc("generate_document", {"title": "R", "format": "pdf", "sections": [{"heading": "H", "paragraphs": ["x"], "source_refs": ["S1"]}]})]),
                  ai("stop"), ai("I created the report for you.")]
        r, svc = make_runner(script, retrieval=FakeRetrieval([cand(1)]))
        out = chat(r, "Create a PDF report about alpha")
        kw = [m["kw"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] for m in FakeChat.log]
        self.assertEqual(kw, [True, True, False])                    # drafting in thinking mode, closing note without
        self.assertEqual([(d["requested"], d["applied"]) for d in out.reasoning], [("thinking", "thinking"), ("non_thinking", "non_thinking")])
        self.assertEqual(out.metadata["mode_changes"], 1)

    def test_known_artifact_capability_is_forced_on_the_first_tool_step_only(self):
        spec = {"title": "R", "format": "pdf", "sections": [{"heading": "H", "paragraphs": ["x"], "source_refs": ["S1"]}]}
        r, _ = make_runner([ai("", [tc("generate_document", spec)]), ai("stop"), ai("I created the report.")], retrieval=FakeRetrieval([cand(1)]))
        chat(r, "Create a PDF report about alpha")
        self.assertEqual([m["tool_choice"] for m in FakeChat.log[:2]], ["generate_document", None])
        r, _ = make_runner([ai("", [tc("generate_chart_or_table", {"title": "T", "categories": ["a"], "series": [{"name": "s", "values": [1]}]})]),
                            ai("stop"), ai("Here is the chart.")], retrieval=FakeRetrieval([cand(1)]))
        chat(r, "plot a bar chart of sales")
        self.assertEqual(FakeChat.log[0]["tool_choice"], "generate_chart_or_table")
        r, _ = make_runner([ai("", [tc("run_calculation", {"expression": "1+1"})]), ai("stop"), ai("2")])
        chat(r, "what is 15% of 200?")
        self.assertIsNone(FakeChat.log[0]["tool_choice"])                       # no single known capability -> model chooses

    def test_runtime_rejecting_forced_choice_falls_back_to_auto_with_warning(self):
        class BadRequestError(Exception):
            pass
        spec = {"title": "R", "format": "pdf", "sections": [{"heading": "H", "source_refs": ["S1"]}]}
        r, _ = make_runner([BadRequestError("tool_choice unsupported"), ai("", [tc("generate_document", spec)]), ai("stop"), ai("Done.")],
                           retrieval=FakeRetrieval([cand(1)]))
        out = chat(r, "Create a PDF report about alpha")
        self.assertEqual(len(out.artifacts), 1)
        self.assertTrue(any("forced tool choice" in w for w in out.warnings))
        self.assertEqual([m["tool_choice"] for m in FakeChat.log[:2]], ["generate_document", None])

    def test_user_supplied_numbers_do_not_trigger_the_citation_revision(self):
        r, _ = make_runner([ai("Revenue was 5 million dollars [S1] versus your baseline of 10000 units.")],
                           retrieval=FakeRetrieval([cand(1, "Revenue was 5 million dollars in 2023.")]))
        out = chat(r, "Compare revenue against a baseline of 10000 units")
        self.assertEqual((out.status, len(FakeChat.log)), ("ok", 1))

    def test_truncated_tool_call_in_thinking_mode_is_retried_without_thinking_with_a_larger_cap(self):
        spec = {"title": "R", "format": "pdf", "sections": [{"heading": "H", "paragraphs": ["x"], "source_refs": ["S1"]}]}
        bad = {"name": "generate_document", "args": '{"title": "R", "sections": [{"head', "id": "t1", "error": "bad json", "type": "invalid_tool_call"}
        r, svc = make_runner([ai("", finish="length", invalid=[bad]), ai("", [tc("generate_document", spec)]), ai("stop"), ai("Created.")],
                             retrieval=FakeRetrieval([cand(1)]))
        out = chat(r, "Create a PDF report about alpha")
        self.assertEqual(len(out.artifacts), 1)
        self.assertTrue(any("malformed or truncated" in w for w in out.warnings))
        kw = [(m["kw"]["extra_body"]["chat_template_kwargs"]["enable_thinking"], m["kw"]["max_tokens"]) for m in FakeChat.log[:2]]
        self.assertEqual(kw, [(True, 2400), (False, 2400)])        # artifact drafting may exceed the 1000-token answer limit
        self.assertEqual([(d["requested"], d["applied"]) for d in out.reasoning][:2], [("thinking", "thinking"), ("non_thinking", "non_thinking")])

    def test_missing_forced_tool_call_is_reported_not_silent(self):
        r, svc = make_runner([ai("I will do that."), ai("Sorry, I could not produce it.")], retrieval=FakeRetrieval([cand(1)]))
        out = chat(r, "Create a PDF report about alpha")
        self.assertEqual(out.artifacts, [])
        self.assertTrue(any("did not produce the requested tool call" in w for w in out.warnings))

    def test_ordinary_answers_keep_the_1000_token_cap(self):
        r, _ = make_runner([ai("Revenue was 5 million dollars [S1].")], retrieval=FakeRetrieval([cand(1, "Revenue was 5 million dollars in 2023.")]))
        chat(r, "What was the revenue?")
        self.assertEqual(FakeChat.log[0]["kw"]["max_tokens"], 1000)
        r, _ = make_runner([ai("Revenue was 5 million dollars [S1].")], retrieval=FakeRetrieval([cand(1, "Revenue was 5 million dollars in 2023.")]))
        chat(r, "What was the revenue?", max_output_tokens=300)
        self.assertEqual(FakeChat.log[0]["kw"]["max_tokens"], 300)

    def test_echoed_internal_notes_are_replaced_by_a_deterministic_reply(self):
        script = [ai("", [tc("generate_document", {"title": "R", "format": "pdf", "sections": [{"heading": "H", "source_refs": ["S1"]}]})]),
                  ai("stop"), ai("[generate_document: success - pdf 1234 bytes]")]
        r, svc = make_runner(script, retrieval=FakeRetrieval([cand(1)]))
        out = chat(r, "Create a PDF report about alpha")
        self.assertNotIn("generate_document", out.answer)
        self.assertIn("I created", out.answer)
        self.assertEqual(out.metadata["finish_reason"], "template_fallback")

    def test_mostly_uncited_answer_gets_one_bounded_revision(self):
        ev = [cand(1, "Revenue was 5 million dollars in 2023. Costs fell by 3 percent in 2023. Headcount reached 120 people in 2023.")]
        uncited = "Revenue was 5 million dollars in 2023. Costs fell by 3 percent in 2023. Headcount reached 120 people in 2023."
        r, _ = make_runner([ai(uncited), ai(uncited.replace("2023.", "2023 [S1].").replace("people in 2023 [S1].", "people in 2023 [S1]."))], retrieval=FakeRetrieval(ev))
        out = chat(r, "Summarize the company year")
        self.assertEqual(len(FakeChat.log), 2)
        self.assertIn("no [S#]/[W#] citation", FakeChat.log[1]["messages"][-1].content)
        self.assertEqual((out.status, [c.ref for c in out.citations]), ("ok", ["S1"]))
        r, _ = make_runner([ai(uncited), ai(uncited)], retrieval=FakeRetrieval(ev))
        out = chat(r, "Summarize the company year")
        self.assertEqual(len(FakeChat.log), 2)                                   # still only one revision
        self.assertTrue(any("citation validation" in w for w in out.warnings))

    def test_requested_file_format_overrides_the_models_default(self):
        spec = {"title": "R", "sections": [{"heading": "H", "paragraphs": ["x"], "source_refs": ["S1"]}]}      # model omitted format -> docx default
        r, svc = make_runner([ai("", [tc("generate_document", spec)]), ai("stop"), ai("Created.")], retrieval=FakeRetrieval([cand(1)]))
        out = chat(r, "Create a PDF report about alpha")
        self.assertEqual([a["format"] for a in out.artifacts], ["pdf"])
        self.assertIn("Requested output format: pdf", FakeChat.log[0]["messages"][1].content)
        self.assertEqual(out.tool_trace[1].input["format"], "pdf")
        r, svc = make_runner([ai("", [tc("generate_document", {**spec, "format": "docx"})]), ai("stop"), ai("Created.")], retrieval=FakeRetrieval([cand(1)]))
        out = chat(r, "Create a Word report about alpha")
        self.assertEqual([a["format"] for a in out.artifacts], ["docx"])

    def test_model_citation_styles_are_normalized_before_validation(self):
        r, _ = make_runner([ai("Revenue was 5 million dollars [**S1**].")], retrieval=FakeRetrieval([cand(1, "Revenue was 5 million dollars in 2023.")]))
        out = chat(r, "What was the revenue?")
        self.assertEqual(([c.ref for c in out.citations], out.answer), (["S1"], "Revenue was 5 million dollars [S1]."))

    def test_model_tool_choices_outside_the_selected_groups_are_rejected(self):
        script = [ai("", [tc("web_search", {"query": "x"}), tc("ingest_approved_source", {"source_id": "abcdefgh"}, id="t2")]), ai("stop"), ai("ok")]
        r, _ = make_runner(script, retrieval=FakeRetrieval([cand(1)]))
        out = chat(r, "Create a PDF report about alpha")
        self.assertFalse(any(t.tool_name in ("web_search", "ingest_approved_source") for t in out.tool_trace))
        self.assertTrue(any("not available" in (w or "") for w in [out.answer]) or out.status in ("ok", "partial"))

    def test_invalid_tool_arguments_and_generation_failure_are_reported(self):
        script = [ai("", [tc("generate_document", {"title": "T", "sections": []})]), ai("", [tc("generate_document", {"title": "T", "sections": [{"heading": "H", "source_refs": ["S99"]}]})]), ai("stop"), ai("Could not create it.")]
        r, svc = make_runner(script, retrieval=FakeRetrieval([cand(1)]))
        out = chat(r, "Create a PDF report about alpha")
        statuses = [(t.tool_name, t.status, t.error_code) for t in out.tool_trace if t.tool_name == "generate_document"]
        self.assertEqual(statuses, [("generate_document", "error", "invalid_input")] * 2)
        self.assertEqual(out.artifacts, [])
        self.assertIn("did not succeed", FakeChat.log[-1]["messages"][-1].content)

    def test_tool_call_and_step_limits(self):
        loop = [ai("", [tc("run_calculation", {"expression": "1+1"}, id=f"t{i}")]) for i in range(10)]
        r, svc = make_runner(loop + [ai("done")])
        svc.settings = asettings(max_tool_calls=3, max_llm_steps=3)
        r.s = svc.settings
        out = chat(r, "compute with the tools: sum of values 1 and 2 using the calculator")
        self.assertLessEqual(len([t for t in out.tool_trace if t.tool_name == "run_calculation"]), 3)
        self.assertLessEqual(out.metadata["counters"]["tool_calls"], 3)


class WebFlowTests(unittest.TestCase):
    def web_runner(self, script, **kw):
        class P:
            name = "fake"
            async def search(self, q, n):
                from app.agent.web import SearchHit
                return [SearchHit("https://example.org/report", "Report", "snippet")]
        ingest = []
        r, svc = make_runner(script, retrieval=FakeRetrieval([]), search_provider=P(),
                             fetcher=fetcher(lambda q: httpx.Response(200, headers={"content-type": "text/html"}, text=HTML)),
                             ingest_pdf=lambda *a: ingest.append(a) or {"document_id": "x", "job_id": "j", "status": "queued"}, **kw)
        return r, svc, ingest

    def test_web_research_cites_urls_not_chunks(self):
        r, svc, _ = self.web_runner([ai("Quarterly revenue grew 12 percent [W1].")])
        out = chat(r, "search the web for quarterly revenue growth", allow_web=True)
        self.assertEqual(out.status, "ok")
        c = out.citations[0]
        self.assertEqual((c.ref, c.kind, c.url, c.document_id, c.chunk_id), ("W1", "web", "https://example.org/report", None, None))
        self.assertTrue(c.passage_id)
        self.assertEqual([t.tool_name for t in out.tool_trace][:4], ["search_knowledge_base", "web_search", "fetch_webpage", "extract_web_evidence"])
        self.assertIn("untrusted", FakeChat.log[0]["messages"][0].content)

    def test_web_is_never_used_without_opt_in(self):
        r, svc, _ = self.web_runner([])
        out = chat(r, "search the web for quarterly revenue growth", allow_web=False)
        self.assertNotIn("web_search", [t.tool_name for t in out.tool_trace])

    def test_proposal_awaits_approval_and_nothing_is_ingested(self):
        r, svc, ingest = self.web_runner([ai("Quarterly revenue grew 12 percent [W1]. The page is prepared for saving but is not saved yet.")])
        out = chat(r, "search the web for quarterly revenue growth and save this to the knowledge base", allow_web=True)
        self.assertEqual(out.status, "awaiting_approval")
        self.assertEqual(len(out.pending_approvals), 1)
        self.assertIn("approve", out.pending_approvals[0]["approve"])
        self.assertEqual(ingest, [])
        self.assertIn("NOT saved", FakeChat.log[0]["messages"][-1].content)
        sid = out.pending_approvals[0]["source_id"]
        self.assertEqual(svc.store.get_source(sid, "alice")["status"], "pending_approval")
        self.assertIsNone(svc.store.get_approval(sid, "alice"))

    def test_web_provider_failure_degrades_gracefully(self):
        class Bad:
            name = "bad"
            async def search(self, q, n): raise RuntimeError("down")
        r, svc = make_runner([], retrieval=FakeRetrieval([]), search_provider=Bad())
        out = chat(r, "search the web for quarterly revenue growth", allow_web=True)
        self.assertEqual(out.status, "abstained")
        self.assertTrue(any("web search failed" in w for w in out.warnings))


class ContextBudgetTests(unittest.TestCase):
    def big_runner(self, script, summarizer_responses, n=8, **kw):
        cands = [cand(i, f"Fact {i}: the measured value for sample {i} was {i * 10} units. " + "Context sentence. " * 28, score=.9 - i * .01) for i in range(1, n + 1)]
        inf = isettings(max_model_len=4096)
        r, svc = make_runner(script, inference=inf, retrieval=FakeRetrieval(cands), **kw)
        svc.settings = asettings(evidence_compaction_ratio=.6)   # evidence (~1k tokens) passes 60% of the 1.3k evidence budget
        r.s = svc.settings
        svc.summarizer = EvidenceSummarizer(scripted_llm(summarizer_responses) if summarizer_responses is not None else None, CharCounter())
        return r, svc

    def test_compaction_triggers_before_overflow_and_keeps_originals(self):
        claims = [{"text": f"The measured value for sample {i} was {i * 10} units", "sources": [f"S{i}"]} for i in range(3, 9)]
        r, svc = self.big_runner([ai("Sample 1 was 10 units [S1]; summary says sample 3 was 30 units [S3].")], [{"claims": claims}])
        out = chat(r, "What were the measured values?")
        comp = out.context["compaction"]
        self.assertTrue(any(c.get("rung") == "summarize_evidence" for c in comp), comp)
        self.assertIn("summarize_evidence", [t.tool_name for t in out.tool_trace])        # invoked by the orchestrator, not the model
        self.assertNotIn("summarize_evidence", FakeChat.log[0]["tools"] or [])
        before_after = [b for b in out.context["budget"] if b["label"] in ("after_execute", "after_compaction")]
        self.assertTrue(before_after[0]["used"] > before_after[-1]["used"])
        final_prompt = FakeChat.log[0]["messages"][-1].content
        self.assertIn("EVIDENCE SUMMARY", final_prompt)
        self.assertIn("[S3]", final_prompt)                                                 # provenance survives in the summary
        self.assertEqual({c.ref for c in out.citations}, {"S1", "S3"})                      # citations still resolve to real chunks
        self.assertEqual(out.citations[1].chunk_id, "c3")
        self.assertEqual(r.svc.store is not None, True)

    def test_summarizer_failure_falls_back_to_extractive_and_still_answers(self):
        r, svc = self.big_runner([ai("Sample 1 was 10 units [S1].")], [RuntimeError("llm down")])
        out = chat(r, "What were the measured values?")
        self.assertTrue(any("extractive" in w for w in out.warnings), out.warnings)
        self.assertEqual(out.status, "ok")

    def test_no_summarizer_configured_reduces_breadth_with_explicit_warning(self):
        r, svc = self.big_runner([ai("Sample 1 was 10 units [S1].")], None)
        out = chat(r, "What were the measured values?")
        self.assertTrue(any("left out of the prompt" in w or "extractive" in w for w in out.warnings), out.warnings)
        prompt = FakeChat.log[0]["messages"][-1].content
        from app.agent.budget import ContextBudget
        self.assertLess(r.counter.count(prompt), 4096 - 1000)                                 # the prompt really fits the window

    def test_small_evidence_does_not_trigger_compaction(self):
        r, svc = make_runner([ai("Revenue was 5 million dollars [S1].")], retrieval=FakeRetrieval([cand(1, "Revenue was 5 million dollars in 2023.")]))
        out = chat(r, "What was the revenue?")
        self.assertEqual(out.context["compaction"], [])
        self.assertNotIn("summarize_evidence", [t.tool_name for t in out.tool_trace])


class GraphStructureTests(unittest.TestCase):
    def test_graph_compiles_with_expected_nodes(self):
        r, _ = make_runner([])
        from app.agent.graph import Run
        from app.agent.budget import BudgetConfig, ContextBudget
        from app.agent.evidence import EvidenceStore
        from app.agent.registry import CapabilityContext
        from app.agent.runtime import ReasoningController
        g = r._build(Run(AgentChatRequest(message="x"), "u", "r", EvidenceStore(), CapabilityContext("u", "r", None, None),
                         ReasoningController(r.adapter.caps), ContextBudget(CharCounter(), BudgetConfig(8192, 1000, 400, 128))))
        nodes = set(g.get_graph().nodes) - {"__start__", "__end__"}
        self.assertEqual(nodes, {"classify_request", "configure_reasoning_mode", "plan_capabilities", "execute_capability", "check_context_budget",
                                 "summarize_if_required", "evaluate_evidence", "generate_final_response", "validate_citations", "finalize"})


if __name__ == "__main__":
    unittest.main()
