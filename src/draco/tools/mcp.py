"""Optional MCP integration: expose tools of a connected MCP client as registry capabilities.

Duck-typed against the client surface shared by `mcp.Client` and `mcp.ClientSession`
(`await list_tools()`, `await call_tool(name, arguments)`), so the core never imports the `mcp`
package. Verified against mcp 2.3.0 (see tests/draco/integration/test_mcp_real.py); install it with
`pip install draco[mcp]`.

Remote servers are untrusted. Descriptions are sanitized, outputs are flagged untrusted, nothing is
imported unless explicitly allow-listed, tools are treated as state-mutating (never auto-retried)
and need the `mcp` permission, which the execution policy must grant.
"""
from __future__ import annotations

import re
from typing import Any, cast

from app.agent.contracts import CapabilityError, ErrorCode
from app.agent.registry import Capability, CapabilityContext, CapabilityRegistry
from app.agent.web import looks_injected
from pydantic import BaseModel, ConfigDict, create_model

from draco.core.errors import ContractError

_NAME_OK = re.compile(r"[^A-Za-z0-9_-]")
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_PY_TYPES: dict[str, Any] = {"string": str, "integer": int, "number": float, "boolean": bool,
                             "array": list, "object": dict}
WITHHELD = "Remote MCP tool (description withheld: unverified remote text)."


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
        if v is not None:
            return v
    return default


def sanitize_description(text: str | None, limit: int = 300) -> str:
    if not text:
        return "Remote MCP tool."
    clean = _CTRL.sub(" ", text).strip()
    return WITHHELD if looks_injected(clean) else clean[:limit]


def input_model_from_schema(name: str, schema: dict[str, Any] | None) -> type[BaseModel]:
    """Strict pydantic model for a (simple) JSON schema: unknown fields are rejected, required fields
    enforced, scalar/array/object types checked. Complex constructs fall back to `Any` for that field."""
    props = (schema or {}).get("properties") or {}
    required = set((schema or {}).get("required") or [])
    fields: dict[str, Any] = {}
    for key, spec in props.items():
        py: Any = cast(Any, _PY_TYPES.get(str(spec.get("type")))) if isinstance(spec, dict) else None
        py = py or Any
        fields[key] = (py, ...) if key in required else (py | None, None)
    return create_model(f"{name}_Input", __config__=ConfigDict(extra="forbid"), **fields)


class McpOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: list[str] = []
    structured: Any = None


async def register_mcp_tools(registry: CapabilityRegistry, client: Any, *, server_name: str, allow: set[str],
                             permissions: frozenset[str] | set[str] = frozenset({"mcp"}),
                             timeout_s: float = 30.0, max_text_chars: int = 8000,
                             requires_approval: bool = False) -> list[str]:
    """Register the allow-listed remote tools as `mcp__<server>__<tool>`. Returns registered names.
    `allow` is mandatory: nothing is imported implicitly. Unknown names in `allow` are an error so a
    typo cannot silently widen or narrow exposure."""
    if not allow:
        raise ContractError("register_mcp_tools requires an explicit non-empty allow-list")
    listing = await client.list_tools()
    if _get(listing, "next_cursor", "nextCursor"):
        raise ContractError("MCP server returned a paginated tool list; pagination is not supported yet")
    tools: dict[str, Any] = {str(_get(t, "name")): t for t in _get(listing, "tools", default=[])}
    unknown = allow - tools.keys()
    if unknown:
        raise ContractError(f"MCP server '{server_name}' has no tool(s) {sorted(unknown)}")
    server = _NAME_OK.sub("_", server_name)
    registered: list[str] = []
    for remote in sorted(allow):
        tool = tools[remote]
        local = f"mcp__{server}__{_NAME_OK.sub('_', remote)}"
        schema = _get(tool, "input_schema", "inputSchema", default={})
        in_model = input_model_from_schema(local, schema)

        def make_handler(remote_name: str):
            async def handler(args: BaseModel, ctx: CapabilityContext) -> McpOutput:
                res = await client.call_tool(remote_name, args.model_dump(exclude_none=True))
                if _get(res, "is_error", "isError", default=False):
                    # remote error text is untrusted and may contain internals: report generically
                    raise CapabilityError(ErrorCode.upstream_error, f"Remote tool '{remote_name}' reported an error")
                texts: list[str] = []
                for item in _get(res, "content", default=[]):
                    t = _get(item, "text")
                    texts.append(t[:max_text_chars] if isinstance(t, str) else "[non-text content omitted]")
                return McpOutput(content=texts, structured=_get(res, "structured_content", "structuredContent"))
            return handler

        registry.register(Capability(
            name=local, group=f"mcp:{server}", description=sanitize_description(_get(tool, "description")),
            input_model=in_model, output_model=McpOutput, handler=make_handler(remote),
            permissions=frozenset(permissions), mutates_state=True,  # unknown side effects: no auto-retry
            requires_approval=requires_approval, untrusted_output=True, timeout_s=timeout_s))
        registered.append(local)
    return registered
