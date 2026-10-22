"""Example D: register a native tool, validate its schema, enforce permissions, execute, audit."""
import asyncio

from pydantic import BaseModel, Field

from draco import ExecutionContext, Harness, Policy
from draco.tools import ToolRegistry, ToolRuntime, native_tool

registry = ToolRegistry()


class Convert(BaseModel):
    celsius: float = Field(ge=-273.15)


@native_tool(registry, name="to_fahrenheit", description="Convert Celsius to Fahrenheit", input_model=Convert)
async def to_fahrenheit(args, ctx):
    return {"fahrenheit": args.celsius * 9 / 5 + 32}


class Empty(BaseModel):
    pass


@native_tool(registry, name="delete_records", description="Delete records", input_model=Empty,
             permissions={"write"}, mutates_state=True)
async def delete_records(args, ctx):
    return {"deleted": 3}


async def main():
    h = Harness()
    tools = ToolRuntime(registry)
    ctx = ExecutionContext(services=h.services, agent_id="demo", policy=Policy(
        allowed_tools=frozenset({"to_fahrenheit", "delete_records"}), permissions=frozenset({"read"})))
    print("exposed to the model:", [s["function"]["name"] for s in tools.specs(ctx)])
    for name, args in (("to_fahrenheit", {"celsius": 100}), ("to_fahrenheit", {"celsius": -500}),
                       ("delete_records", {}), ("no_such_tool", {})):
        r = await tools.call(ctx, name, args)
        print(f"{name:<15} {args!s:<22} -> {r.status:<7}", r.data or (r.error.code.value, r.error.message))
    print("audit events:", [(e["event"], e["data"].get("tool")) for e in h.tracer.trace(ctx.execution_id)])
    print("budget used:", ctx.budget.snapshot())

asyncio.run(main())
