# Call a model provider

Draco does not call models by itself, and does not care which one you use. It gives you a thin, provider-neutral contract and a wrapper that makes a model call **count against the budget, respect cancellation, and show up in the trace**. This guide shows the wrapper, the two providers that ship with Draco, and how to plug in your own.

## The contract

A provider is any object with a `name` and an async `generate(request) -> response`:

```python
from draco.adapters.model_provider import ModelProvider, ModelRequest, ModelResponse

class MyProvider:
    name = "my-model-server"

    async def generate(self, request: ModelRequest) -> ModelResponse:
        ...   # call your model; return ModelResponse(text, model, input_tokens, output_tokens)
```

`ModelRequest` has `messages` (a list of `{"role", "content"}` dicts), an optional `model`, and a free-form `params` dict (temperature, max tokens, ...) that your provider interprets. `ModelResponse` has `text`, the `model` name, token counts, and `raw` for the provider's own payload, which is **never exported to traces**.

## Make the call count

Call the provider through `generate(ctx, provider, request)`, not directly:

```python
from draco.adapters.model_provider import generate

response = await generate(ctx, provider, request)
```

This does what a bare call would not:

- Checks cancellation and the deadline first.
- **Charges one model call** to the run's budget and, afterwards, the reported tokens.
- Emits `model.before`, then `model.after` (with latency and tokens) or `model.failed`.

So `BudgetLimits(max_model_calls=..., max_tokens=...)` really bounds what the run can spend, as [budgets and limits](../concepts/budgets.md) describes. For a model call that does not go through a `ModelProvider`, the lower-level `call_model(ctx, fn, ..., provider=, model=, usage_of=)` does the same job.

## Retries

Provider exceptions propagate unchanged. Have your provider raise `ProviderError` with `retryable` set honestly, and wrap the call in `call_with_retry` if repeating it is safe:

```python
from draco import call_with_retry

response = await call_with_retry(ctx, lambda: generate(ctx, provider, request))
```

`call_with_retry` retries only errors that say they are retryable, backs off between attempts, gives up after `max_attempts` (default 3), and **charges each retry** to the budget. A failed attempt also counts as a model call, which is accurate: you did call the model.

## The shipped providers

Both need `pip install "draco-harness[providers]"` (they use `httpx`).

| Provider | For |
|---|---|
| `OpenAICompatProvider(base_url, api_key=Secret(...), model=...)` | OpenAI, vLLM, llama.cpp server, and anything else speaking the chat-completions API. `base_url` includes the version prefix, such as `https://host/v1`. |
| `AnthropicProvider(Secret(...), model=...)` | The Anthropic Messages API |

API keys are `Secret` values, so they never appear in logs, reprs or traces. Both providers map HTTP failures to `ProviderError`: **429 and 5xx and timeouts are retryable, other 4xx are not**, and the error message **never includes the response body**, which could echo back a key or an internal detail.

You can pass your own `httpx.AsyncClient` as `client=` to control transport, proxies or testing. The example below uses that to run a real provider against a fake server, so it works offline:

<!-- doctest -->
```python
import asyncio
import httpx
from draco import BudgetLimits, Harness, call_with_retry
from draco.adapters.model_provider import ModelRequest, generate
from draco.adapters.providers.openai_compat import OpenAICompatProvider
from draco.security import Secret

hits = {"n": 0}

def fake_server(request: httpx.Request) -> httpx.Response:
    hits["n"] += 1
    if hits["n"] == 1:      # the first request is rate limited, and the body mentions a key
        return httpx.Response(429, json={"error": "rate limited, key sk-leaky-abcdefgh"})
    return httpx.Response(200, json={"model": "demo-1", "choices": [{"message": {"content": "Hello!"}}],
                                     "usage": {"prompt_tokens": 12, "completion_tokens": 3}})

async def main():
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake_server))
    provider = OpenAICompatProvider("https://models.example/v1", api_key=Secret("sk-test-123456"),
                                    model="demo-1", client=client)
    h = Harness()

    async def ask(ctx, question):
        request = ModelRequest([{"role": "user", "content": question}])
        response = await call_with_retry(ctx, lambda: generate(ctx, provider, request))
        return response.text

    h.register("ask", ask)

    r = await h.run("ask", "Say hello", limits=BudgetLimits(max_tokens=100))
    print(r.status, r.output)
    print("model calls:", r.usage.model_calls, "| retries:", r.usage.retries, "| tokens:", r.usage.tokens)
    print([e["event"] for e in h.tracer.trace(r.execution_id) if e["event"].startswith(("model", "retry"))])

    hits["n"] = 0
    r = await h.run("ask", "Again", limits=BudgetLimits(max_tokens=10))
    print(r.status, r.error.code, r.error.details)

asyncio.run(main())
```

<!-- output -->
```text
completed Hello!
model calls: 2 | retries: 1 | tokens: 15
['model.before', 'model.failed', 'retry', 'model.before', 'model.after']
budget_exhausted budget {'resource': 'tokens', 'limit': 10}
```

What happened: the first request got a 429, which the provider reported as a retryable `ProviderError` (`model.failed`); `call_with_retry` retried; the second request succeeded with 15 tokens. That cost two model calls and one retry. In the second run, 15 tokens exceeded `max_tokens=10`, so the run ended `budget_exhausted`.

!!! note "The token limit is checked when usage is reported"
    A provider reports tokens after the call finishes, so the call that crosses `max_tokens` has already happened. The limit stops the **next** step, not the one in flight. For a hard per-call cap, also pass the provider a `max_tokens` parameter in `ModelRequest.params`.

## Your own provider

Implement the contract, raise `ProviderError(message, retryable=...)` for upstream failures, and never put response bodies, headers or credentials in the message:

```python
from draco.core import ProviderError

async def generate(self, request):
    try:
        data = await self._client.post(...)
    except TimeoutError as exc:
        raise ProviderError("my-model-server timed out", retryable=True) from exc
```

## See also

- [Budgets and limits](../concepts/budgets.md)
- [Tracing and redaction](../concepts/observability.md): what `model.*` events record
- [Security model](../concepts/security.md): how `Secret` keeps keys out of logs
