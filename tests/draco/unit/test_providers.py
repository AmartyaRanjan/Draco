"""HTTP providers against httpx.MockTransport: request shape, usage, error mapping, no secret leakage."""
import json

import httpx
import pytest

from draco.adapters.model_provider import ModelRequest, generate
from draco.adapters.providers.anthropic import AnthropicProvider
from draco.adapters.providers.openai_compat import OpenAICompatProvider
from draco.core import ConfigurationError, DracoSettings, ProviderError, Status
from draco.harness import Harness
from draco.security import Secret

KEY = Secret("sk-test-very-secret-123456")


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_openai_compat_request_shape_and_usage():
    seen = {}

    def handler(req):
        seen.update(url=str(req.url), auth=req.headers["authorization"], body=json.loads(req.content))
        return httpx.Response(200, json={"model": "m1", "choices": [{"message": {"content": "hi"}}],
                                         "usage": {"prompt_tokens": 3, "completion_tokens": 4}})
    p = OpenAICompatProvider("http://x/v1/", api_key=KEY, model="m1", client=client(handler))
    r = await p.generate(ModelRequest([{"role": "user", "content": "yo"}], params={"temperature": 0}))
    assert (r.text, r.total_tokens, r.model) == ("hi", 7, "m1")
    assert seen["url"] == "http://x/v1/chat/completions" and seen["auth"] == f"Bearer {KEY.reveal()}"
    assert seen["body"]["temperature"] == 0 and seen["body"]["messages"][0]["content"] == "yo"


async def test_anthropic_request_shape_splits_system_and_sums_text_blocks():
    seen = {}

    def handler(req):
        seen.update(url=str(req.url), key=req.headers["x-api-key"], ver=req.headers["anthropic-version"], body=json.loads(req.content))
        return httpx.Response(200, json={"model": "claude-x", "content": [{"type": "text", "text": "a"}, {"type": "tool_use"}, {"type": "text", "text": "b"}],
                                         "usage": {"input_tokens": 5, "output_tokens": 2}})
    p = AnthropicProvider(KEY, model="claude-x", client=client(handler))
    r = await p.generate(ModelRequest([{"role": "system", "content": "be brief"}, {"role": "user", "content": "q"}]))
    assert (r.text, r.input_tokens, r.output_tokens) == ("ab", 5, 2)
    assert seen["url"].endswith("/v1/messages") and seen["ver"] == "2023-06-01" and seen["body"]["system"] == "be brief"
    assert seen["body"]["messages"] == [{"role": "user", "content": "q"}] and seen["body"]["max_tokens"] == 1024


@pytest.mark.parametrize("status,retryable", [(429, True), (503, True), (400, False), (401, False)])
async def test_http_errors_are_typed_and_leak_nothing(status, retryable):
    def handler(req):
        return httpx.Response(status, text=f"echo of prompt and key {KEY.reveal()}")
    p = OpenAICompatProvider("http://x/v1", api_key=KEY, model="m", client=client(handler))
    with pytest.raises(ProviderError) as e:
        await p.generate(ModelRequest([]))
    assert e.value.retryable is retryable and KEY.reveal() not in e.value.message and str(status) in e.value.message


async def test_timeout_transport_error_and_bad_payloads():
    def timeout(req):
        raise httpx.ReadTimeout("slow", request=req)

    def refused(req):
        raise httpx.ConnectError("no", request=req)
    for h in (timeout, refused):
        with pytest.raises(ProviderError) as e:
            await OpenAICompatProvider("http://x/v1", model="m", client=client(h)).generate(ModelRequest([]))
        assert e.value.retryable
    for payload in ({"choices": []}, {"nothing": 1}, [1]):
        with pytest.raises(ProviderError) as e:
            await OpenAICompatProvider("http://x/v1", model="m", client=client(lambda r, p=payload: httpx.Response(200, json=p))).generate(ModelRequest([]))
        assert not e.value.retryable
    with pytest.raises(ProviderError):
        await AnthropicProvider(KEY, model="m", client=client(lambda r: httpx.Response(200, text="<html>"))).generate(ModelRequest([]))
    with pytest.raises(ConfigurationError):
        await AnthropicProvider(KEY, client=client(lambda r: httpx.Response(200, json={}))).generate(ModelRequest([]))


async def test_provider_through_harness_charges_tokens_and_errors_are_retryable_results():
    ok = OpenAICompatProvider("http://x/v1", model="m", client=client(lambda r: httpx.Response(
        200, json={"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})))
    bad = OpenAICompatProvider("http://x/v1", model="m", client=client(lambda r: httpx.Response(503)))
    h = Harness(DracoSettings())

    async def good(ctx, _):
        return (await generate(ctx, ok, ModelRequest([]))).text

    async def failing(ctx, _):
        return await generate(ctx, bad, ModelRequest([]))
    h.register("good", good)
    h.register("bad", failing)
    r = await h.run("good")
    assert r.output == "hi" and r.usage.tokens == 15 and r.usage.model_calls == 1
    r = await h.run("bad")
    assert r.status == Status.failed and r.error.retryable and r.error.code == "provider"
