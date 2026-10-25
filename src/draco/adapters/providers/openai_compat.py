"""OpenAI-compatible chat completions (OpenAI, vLLM, llama.cpp server, ...)."""
from __future__ import annotations

from typing import Any

from draco.adapters.model_provider import ModelRequest, ModelResponse
from draco.adapters.providers._http import post_json
from draco.core.errors import ConfigurationError, ProviderError
from draco.security.secrets import Secret


class OpenAICompatProvider:
    def __init__(self, base_url: str, *, api_key: Secret | None = None, model: str | None = None,
                 client: Any = None, timeout_s: float = 60.0, name: str = "openai_compat"):
        """`base_url` includes the version prefix, e.g. https://host/v1. Pass `client` (httpx.AsyncClient)
        to control transport, proxies or testing; otherwise one is created lazily and owned by the provider."""
        self.name, self.base_url, self.model, self._key, self._timeout = name, base_url.rstrip("/"), model, api_key, timeout_s
        self._client = client

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
        headers = {"content-type": "application/json"}
        if self._key is not None:
            headers["authorization"] = f"Bearer {self._key.reveal()}"
        body: dict[str, Any] = {"model": model, "messages": request.messages, **request.params}
        data = await post_json(self._http(), f"{self.base_url}/chat/completions", headers=headers, body=body,
                               provider=self.name)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"{self.name} returned an unexpected payload") from exc
        usage = data.get("usage") or {}
        return ModelResponse(text, str(data.get("model") or model), int(usage.get("prompt_tokens") or 0),
                             int(usage.get("completion_tokens") or 0), raw=data)
