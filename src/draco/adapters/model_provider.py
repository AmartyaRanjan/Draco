"""Provider-neutral model call contract. Concrete providers (vLLM, OpenAI-compatible, Anthropic ...) live
in optional integration modules and only need to satisfy `ModelProvider`."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from draco.core.context import ExecutionContext
from draco.harness.instrument import call_model


@dataclass(frozen=True)
class ModelRequest:
    messages: list[dict[str, Any]]
    model: str | None = None
    params: dict[str, Any] = field(default_factory=dict)  # temperature, max_tokens ... provider-interpreted


@dataclass
class ModelResponse:
    text: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None  # provider payload; never exported to traces

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@runtime_checkable
class ModelProvider(Protocol):
    name: str

    async def generate(self, request: ModelRequest) -> ModelResponse: ...


async def generate(ctx: ExecutionContext, provider: ModelProvider, request: ModelRequest) -> ModelResponse:
    """Run a provider call as a budget-charged, cancellable, evented model call with token accounting.
    Provider exceptions propagate unchanged: have providers raise `ProviderError` with `retryable` set,
    then wrap with `draco.harness.call_with_retry` if the call is safe to repeat."""
    return await call_model(ctx, provider.generate, request, provider=provider.name, model=request.model or "",
                            usage_of=lambda r: r.total_tokens or None)
