"""Example A: run a callable through the harness; inspect the structured result and trace."""
import asyncio
import json

from draco import Harness


async def greeter(ctx, name):
    return {"greeting": f"hello {name}", "agent": ctx.agent_id}


async def main():
    h = Harness()
    h.register("greeter", greeter)
    r = await h.run("greeter", "world", session_id="demo-session", user_id="alice")
    print(json.dumps({k: v for k, v in r.to_dict().items() if k in
                      ("status", "output", "session_id", "agent_id", "usage")}, indent=2))
    print("execution_id:", r.execution_id)
    print("trace events:", [e["event"] for e in h.tracer.trace(r.execution_id)])
    print("persisted status:", (await h.store.get_execution(r.execution_id)).status)

asyncio.run(main())
