"""Anthropic Messages API over httpx (no SDK dependency)."""
from __future__ import annotations

from typing import Any

from draco.adapters.model_provider import ModelRequest, ModelResponse
from draco.adapters.providers._http import post_json
from draco.core.errors import ConfigurationError, ProviderError
from draco.security.secrets import Secret

API_VERSION = "2023-06-01"


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: Secret, *, model: str | None = None, base_url: str = "https://api.anthropic.com",
                 client: Any = None, timeout_s: float = 60.0, default_max_tokens: int = 1024):
        self._key, self.model, self.base_url = api_key, model, base_url.rstrip("/")
        self._client, self._timeout, self.default_max_tokens = client, timeout_s, default_max_tokens

    def _http(self) -> Any:
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        model = request.model or self.model
        if not model:
            raise ConfigurationError("no model configured for this request")
        system = "\n\n".join(str(m["content"]) for m in request.messages if m.get("role") == "system")
        messages = [m for m in request.messages if m.get("role") != "system"]
        body: dict[str, Any] = {"model": model, "max_tokens": self.default_max_tokens, "messages": messages,
                                **request.params}
        if system:
            body["system"] = system
        headers = {"x-api-key": self._key.reveal(), "anthropic-version": API_VERSION, "content-type": "application/json"}
        data = await post_json(self._http(), f"{self.base_url}/v1/messages", headers=headers, body=body, provider=self.name)
        try:
            text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        except (KeyError, TypeError, AttributeError) as exc:
            raise ProviderError("anthropic returned an unexpected payload") from exc
        usage = data.get("usage") or {}
        return ModelResponse(text, str(data.get("model") or model), int(usage.get("input_tokens") or 0),
                             int(usage.get("output_tokens") or 0), raw=data)
