"""Regression tests: inference timeouts must be positive.

RetrievalSettings.validate() already rejects non-positive timeouts; the
inference settings omitted the same check, so INFERENCE_TIMEOUT_S=0 or
AGENT_TOOL_TIMEOUT_S<=0 passed validation and failed later (immediate
LLM/tool timeouts). This mirrors the colbert_max_length gap fixed in PR #3.
"""
import unittest

from app.inference.config import InferenceSettings


def isettings(**kw):
    base = dict(base_url="http://x/v1", api_key="k", model="m", timeout_s=5, max_retries=0,
                max_model_len=8192, max_output_tokens=1000, grounded_temperature=.3, grounded_top_p=.8,
                reasoning_temperature=.6, reasoning_top_p=.95, top_k=20, agent_max_steps=3,
                agent_max_tool_calls=2, agent_workflow_token_budget=3000, tool_timeout_s=2,
                tool_output_tokens=2500, history_tokens=800, memory_tokens=400,
                prompt_overhead_tokens=600, api_keys={"key1": "alice"}, auth_disabled=False, store_path=None)
    return InferenceSettings(**{**base, **kw})


class InferenceTimeoutValidationTests(unittest.TestCase):
    def test_non_positive_inference_timeout_rejected(self):
        for bad in (0, -1, -0.5):
            with self.assertRaises(ValueError, msg=f"timeout_s={bad}"):
                isettings(timeout_s=bad).validate()

    def test_non_positive_tool_timeout_rejected(self):
        for bad in (0, -1, -2.5):
            with self.assertRaises(ValueError, msg=f"tool_timeout_s={bad}"):
                isettings(tool_timeout_s=bad).validate()

    def test_positive_timeouts_accepted(self):
        isettings(timeout_s=0.1, tool_timeout_s=0.1).validate()
        isettings().validate()


if __name__ == "__main__":
    unittest.main()
