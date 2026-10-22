"""Example G: trusted identity, tenant isolation, role grants and an audit trail. No starter app involved."""
import asyncio

from pydantic import BaseModel

from draco import Harness, Identity, Policy
from draco.security import Resource, RoleAuthorizer
from draco.tools import ToolRegistry, ToolRuntime, native_tool

registry = ToolRegistry()
NOTES = {"acme": ["acme roadmap"], "globex": ["globex salaries"]}


class ReadNotes(BaseModel):
    tenant: str  # untrusted tool argument: the model chose it


@native_tool(registry, name="read_notes", description="Read a tenant's notes", input_model=ReadNotes)
async def read_notes(args, ctx):
    # tool arguments are untrusted: authorize the resource they name against the trusted identity
    from draco.security import authorize
    await authorize(ctx, "notes.read", Resource("notes", args.tenant, tenant_id=args.tenant))
    return {"notes": NOTES[args.tenant]}


async def main():
    h = Harness(authorizer=RoleAuthorizer({"analyst": {"workload.*", "tool.*", "notes.*"}}, require_identity=True))
    tools = ToolRuntime(registry)
    alice = Identity("alice", "acme", frozenset({"analyst"}))

    async def agent(ctx, tenant):
        r = await tools.call(ctx, "read_notes", {"tenant": tenant})
        return r.data if r.ok else f"{r.error.code.value}: {r.error.message}"
    h.register("agent", agent)
    policy = Policy(allowed_tools=frozenset({"read_notes"}))
    print("own tenant   ->", (await h.run("agent", "acme", identity=alice, policy=policy)).output)
    print("other tenant ->", (await h.run("agent", "globex", identity=alice, policy=policy)).output)
    print("no identity  ->", (await h.run("agent", "acme", policy=policy)).error.message)
    spoof = await h.run("agent", "acme", identity=alice, tenant_id="globex", policy=policy)
    print("spoofed tenant ->", spoof.error.message)

asyncio.run(main())
