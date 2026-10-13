"""Live vLLM + LangChain tests. Skipped unless INFERENCE_INTEGRATION=1 and the server answers."""
import asyncio
import os
import unittest

from langchain_core.messages import HumanMessage

from app.inference.client import build_llm
from app.inference.config import get_inference_settings
from app.inference.tools import tool_specs

LIVE = os.getenv("INFERENCE_INTEGRATION") == "1"
LOOP = asyncio.new_event_loop()  # one loop: the SDK caches its HTTP client per process


def run(coro):
    return LOOP.run_until_complete(coro)


@unittest.skipUnless(LIVE, "set INFERENCE_INTEGRATION=1 with a running vLLM server")
class LiveVllmTests(unittest.TestCase):
    def setUp(self):
        self.s = get_inference_settings()

    def test_thinking_is_separate_from_content(self):
        llm = build_llm(self.s, thinking=True)
        r = run(llm.ainvoke([HumanMessage("What is 17*23? Answer briefly.")]))
        self.assertIn("391", r.content)
        self.assertNotIn("<think>", r.content)

    def test_non_thinking_is_short_and_streams_with_usage(self):
        llm = build_llm(self.s, thinking=False, streaming=True)

        async def go():
            chunks = [c async for c in llm.astream([HumanMessage("Say hello in one word.")])]
            return chunks
        chunks = run(go())
        self.assertGreater(len(chunks), 1)
        self.assertTrue(any(getattr(c, "usage_metadata", None) for c in chunks))

    def test_tool_call_is_parsed_into_structured_call(self):
        llm = build_llm(self.s, thinking=False).bind_tools(tool_specs())
        r = run(llm.ainvoke([HumanMessage("Search the documents for 'census data'.")]))
        self.assertEqual(r.tool_calls[0]["name"], "search_documents")
        self.assertIn("query", r.tool_calls[0]["args"])

    def test_max_tokens_is_enforced(self):
        llm = build_llm(self.s, thinking=False, max_tokens=16)
        r = run(llm.ainvoke([HumanMessage("Write a long essay about the sea.")]))
        self.assertEqual(r.response_metadata.get("finish_reason"), "length")


if __name__ == "__main__":
    unittest.main()
