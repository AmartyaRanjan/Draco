from __future__ import annotations

from typing import Any

from draco.core.errors import ProviderError

_RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504, 529}


async def post_json(client: Any, url: str, *, headers: dict[str, str], body: dict[str, Any], provider: str) -> dict[str, Any]:
    """POST and decode JSON, mapping every failure to ProviderError. Response bodies and headers are never put
    in the error (they can echo prompts or credentials); only the status code is."""
    import httpx
    try:
        resp = await client.post(url, headers=headers, json=body)
    except httpx.TimeoutException as exc:
        raise ProviderError(f"{provider} request timed out", retryable=True) from exc
    except httpx.HTTPError as exc:
        raise ProviderError(f"{provider} request failed ({type(exc).__name__})", retryable=True) from exc
    if resp.status_code >= 400:
        raise ProviderError(f"{provider} returned HTTP {resp.status_code}", retryable=resp.status_code in _RETRYABLE,
                            details={"status_code": resp.status_code})
    try:
        data = resp.json()
    except ValueError as exc:
        raise ProviderError(f"{provider} returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise ProviderError(f"{provider} returned an unexpected payload")
    return data
