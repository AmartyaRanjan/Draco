"""Example E: controlled failure, structured error, then checkpoint-based resumption. Completed steps
are replayed from the durable record, never re-executed; an interrupted non-idempotent step needs
reconciliation instead of a blind retry."""
import asyncio
import os
import tempfile

from draco import DracoSettings, Harness, step
from draco.state import SqliteStore

side_effects = []
provider_up = False


async def checkout(ctx, order):
    async def charge():
        side_effects.append("charge card")
        return {"charge_id": "ch_1"}

    async def email():
        side_effects.append("send email")
        if not provider_up:
            raise ConnectionError("smtp unreachable")
        return "sent"
    charged = await step(ctx, "charge", charge)  # not idempotent: runs once
    return {"charge": charged, "email": await step(ctx, "email", email, idempotent=True)}


async def main():
    global provider_up
    with tempfile.TemporaryDirectory() as d:
        h = Harness(DracoSettings(), store=SqliteStore(os.path.join(d, "state.db")))
        h.register("checkout", checkout)
        r = await h.run("checkout", {"order": 7})
        print("run 1:", r.status.value, "| error:", r.error.type, r.error.code, "retryable=", r.error.retryable)
        print("side effects so far:", side_effects)
        print("durable steps:", [(s.step, s.status) for s in await h.store.list_steps(r.execution_id)])
        provider_up = True
        r2 = await h.resume(r.execution_id)
        print("run 2 (resume):", r2.status.value, r2.output, "| same execution id:", r2.execution_id == r.execution_id)
        print("side effects total:", side_effects, "<- 'charge card' not repeated")
        print("history:", [e["name"] for e in await h.store.list_events(r.execution_id)])
        print("attempts:", (await h.store.get_execution(r.execution_id)).attempts)
        h.store.close()

asyncio.run(main())
