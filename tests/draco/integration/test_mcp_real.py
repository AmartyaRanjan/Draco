"""Against the real MCP SDK (optional: `pip install draco[mcp]`). Skipped when `mcp` is absent."""
import pytest

mcp = pytest.importorskip("mcp")

from draco.core import DracoSettings, ExecutionContext, Policy  # noqa: E402
from draco.harness import Harness  # noqa: E402
from draco.tools import CapabilityRegistry, ToolRuntime  # noqa: E402
from draco.tools.mcp import register_mcp_tools  # noqa: E402


async def test_real_mcp_server_round_trip():
    from mcp import Client
    from mcp.server.mcpserver import MCPServer
    srv = MCPServer("demo")

    @srv.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    @srv.tool()
    def boom() -> str:
        """Always fails"""
        raise RuntimeError("secret internal path /etc/x")

    async with Client(srv) as client:
        reg = CapabilityRegistry()
        names = await register_mcp_tools(reg, client, server_name="demo", allow={"add", "boom"})
        h = Harness(DracoSettings())
        ctx = ExecutionContext(services=h.services, policy=Policy(allowed_tools=frozenset(names),
                                                                   permissions=frozenset({"mcp"})))
        rt = ToolRuntime(reg)
        r = await rt.call(ctx, "mcp__demo__add", {"a": 2, "b": 3})
        assert r.status == "success" and r.data["content"] == ["5"] and r.data["structured"] == {"result": 5}
        assert (await rt.call(ctx, "mcp__demo__add", {"a": "x", "b": 3})).error.code.value == "invalid_input"
        e = await rt.call(ctx, "mcp__demo__boom", {})
        assert e.status == "error" and "secret" not in e.error.message
