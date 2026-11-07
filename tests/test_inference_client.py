"""/chat client: the Qwen thinking flag comes from the model profile in app.agent.runtime."""
import unittest
from unittest.mock import patch

from app.agent.runtime import PROFILES
from app.inference import client

from tests.test_inference import isettings


class ThinkingProfileTests(unittest.TestCase):
    def test_hybrid_qwen_gets_the_profile_flag(self):
        for model in ("Qwen/Qwen3-4B-FP8", "Qwen/Qwen3-8B-AWQ", "Qwen/Qwen3-4B"):
            self.assertEqual(client.thinking_template_kwargs(model, True), ({"enable_thinking": True}, True))
            self.assertEqual(client.thinking_template_kwargs(model, False), ({"enable_thinking": False}, False))

    def test_fixed_mode_checkpoints_get_no_flag_and_report_their_mode(self):
        self.assertEqual(client.thinking_template_kwargs("Qwen/Qwen3-4B-Instruct-2507", True), ({}, False))
        self.assertEqual(client.thinking_template_kwargs("Qwen/Qwen3-4B-Thinking-2507-FP8", False), ({}, True))

    def test_unknown_model_gets_no_flag(self):
        self.assertEqual(client.thinking_template_kwargs("meta-llama/Llama-3.1-8B", True), ({}, True))

    def test_flag_name_is_owned_by_the_profile(self):
        hybrid = next(p for p in PROFILES if p.supports_toggle)
        renamed = type(hybrid)(hybrid.name, hybrid.pattern, True, None, "thinking_on")
        with patch.object(client, "profile_for", return_value=renamed):
            self.assertEqual(client.thinking_template_kwargs("Qwen/Qwen3-4B", True), ({"thinking_on": True}, True))


class BuildLlmTests(unittest.TestCase):
    def _build(self, model, thinking):
        seen = {}

        def fake(**kw):
            seen.update(kw)
            return kw
        with patch.object(client, "ChatOpenAI", side_effect=fake):
            client.build_llm(isettings(model=model), thinking=thinking, max_tokens=50)
        return seen

    def test_extra_body_and_sampling_follow_the_profile(self):
        kw = self._build("Qwen/Qwen3-4B-FP8", True)
        self.assertEqual(kw["extra_body"], {"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}})
        self.assertEqual((kw["temperature"], kw["top_p"], kw["max_tokens"]), (.6, .95, 50))
        kw = self._build("Qwen/Qwen3-4B-FP8", False)
        self.assertEqual(kw["extra_body"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(kw["temperature"], .3)

    def test_fixed_mode_model_sends_no_template_kwargs(self):
        kw = self._build("Qwen/Qwen3-4B-Instruct-2507-FP8", True)
        self.assertEqual(kw["extra_body"], {"top_k": 20})
        self.assertEqual(kw["temperature"], .3)  # non-thinking checkpoint: grounded sampling


if __name__ == "__main__":
    unittest.main()
