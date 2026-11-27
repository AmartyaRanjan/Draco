# Tutorial: recover from a crash

**You will build** a checkout workflow that charges a card and then sends an email, break it half way, and resume it without charging the customer twice. Then you will see what Draco does when it *cannot* know whether a step finished.

**You need** `pip install draco-harness`. No extras: SQLite is part of Python.

## The problem

A workflow with side effects can die between two of them: the process is killed during a deploy, a network call times out, the database blips. A naive retry from the top runs everything again, and the card is charged twice. Not retrying leaves the order stuck.

Draco handles this with **steps**. A step is a unit of work whose completion is written to durable storage. On resume, a step that already finished returns its recorded result instead of running again.

## 1. Wrap the side effects in steps

`step(ctx, name, fn)` runs `fn` once per execution and records the result. Results must be JSON-serializable, because they are stored:

```python
from draco import step

async def checkout(ctx, order):
    async def charge():
        return {"charge_id": "ch_1"}       # charge the card (not safe to repeat)

    async def email():
        ...                                 # send the receipt (safe to repeat)

    charged = await step(ctx, "charge", charge)
    sent = await step(ctx, "email", email, idempotent=True)
    return {"charge": charged, "email": sent}
```

The `idempotent` flag is a promise **you** make about the step: running it twice has the same effect as running it once. Sending a receipt twice is annoying but harmless; charging a card twice is not. The default is `False`, the cautious answer.

## 2. Give the harness a durable store

Steps are only durable if the store is. The default store is in memory and disappears with the process. A `SqliteStore` writes to a file:

```python
from draco.state import SqliteStore

h = Harness(store=SqliteStore("state.db"))
```

For several processes sharing state, use `PostgresStore` instead. See [persistence](../guides/persistence.md).

## 3. Break it, then resume

This program makes the email step fail the first time, as if the mail server were down. Then it resumes the same execution:

<!-- doctest -->
```python
import asyncio
import os
import tempfile

from draco import Harness, step
from draco.state import SqliteStore

side_effects = []
mail_server_up = False

async def checkout(ctx, order):
    async def charge():
        side_effects.append("charge card")
        return {"charge_id": "ch_1"}

    async def email():
        side_effects.append("send email")
        if not mail_server_up:
            raise ConnectionError("smtp unreachable")
        return "sent"

    charged = await step(ctx, "charge", charge)                  # not idempotent: runs once
    sent = await step(ctx, "email", email, idempotent=True)      # safe to repeat
    return {"charge": charged, "email": sent}

async def main():
    global mail_server_up
    with tempfile.TemporaryDirectory() as d:
        h = Harness(store=SqliteStore(os.path.join(d, "state.db")))
        h.register("checkout", checkout)

        first = await h.run("checkout", {"order": 7})
        print("run 1:", first.status, "|", first.error.type)
        print("side effects:", side_effects)
        print("durable steps:", [(s.step, s.status) for s in await h.store.list_steps(first.execution_id)])

        mail_server_up = True
        second = await h.resume(first.execution_id)
        print("resume:", second.status, second.output)
        print("same execution id:", second.execution_id == first.execution_id)
        print("side effects:", side_effects)
        print("attempts:", (await h.store.get_execution(first.execution_id)).attempts)
        h.store.close()

asyncio.run(main())
```

<!-- output -->
```text
run 1: failed | ConnectionError
side effects: ['charge card', 'send email']
durable steps: [('charge', 'done'), ('email', 'started')]
resume: completed {'charge': {'charge_id': 'ch_1'}, 'email': 'sent'}
same execution id: True
side effects: ['charge card', 'send email', 'send email']
attempts: 2
```

Read the second `side effects` line: `'charge card'` appears **once**. On resume, the `charge` step was already `done`, so Draco returned its recorded result and skipped it. The email step had only `started`, and because it is marked `idempotent=True`, Draco ran it again. The resumed run keeps the **same execution id**, so traces and records stay together, and `attempts` counts the second try.

## 4. When Draco cannot know

Now the harder case. Suppose the *charge* step is the one that breaks, after the card was charged but before Draco could record that it finished. The store says `started`, never `done`. Did the customer get charged? Draco cannot tell, and the step is not marked idempotent, so it does not guess:

<!-- doctest -->
```python
import asyncio
import os
import tempfile

from draco import Harness, step
from draco.state import SqliteStore

effects = []
network_ok = False

async def checkout(ctx, order):
    async def charge():
        effects.append("charge card")
        if not network_ok:
            raise ConnectionError("network dropped after the card was charged")
        return {"charge_id": "ch_1"}

    return await step(ctx, "charge", charge)        # not idempotent

async def main():
    global network_ok
    with tempfile.TemporaryDirectory() as d:
        h = Harness(store=SqliteStore(os.path.join(d, "state.db")))
        h.register("checkout", checkout)

        first = await h.run("checkout", {"order": 7})
        print("run 1:", first.status, effects)

        network_ok = True
        second = await h.resume(first.execution_id)
        print("resume:", second.status, second.error.type, second.error.retryable)
        print(second.error.message)
        print("card charged again?", effects.count("charge card") > 1)
        h.store.close()

asyncio.run(main())
```

<!-- output -->
```text
run 1: failed ['charge card']
resume: failed ReconciliationRequired False
step 'charge' started before an interruption and is not marked idempotent; verify its effect externally before resuming
card charged again? False
```

The resume **fails on purpose** with `ReconciliationRequired`. It is not retryable, and it tells you why: a human or a reconciliation job has to check with the payment provider whether the charge went through. Once you know, you can finish the order yourself or make the charge step idempotent, for example by sending a stable idempotency key to the provider, and mark it `idempotent=True`.

!!! warning "What Draco does and does not promise"
    Draco **avoids re-running steps it knows finished**, and **refuses to blindly re-run a non-idempotent step it cannot account for**. It does **not** promise exactly-once effects. Making a side effect safe to repeat, usually with an idempotency key, is your job. Draco's job is to stop you from repeating one by accident.

## 5. Other ways a run pauses

A workload can also *choose* to stop and wait, for example for a human approval, by returning `Waiting`. The run's status becomes `waiting`, and `h.resume(execution_id, input=...)` continues it later under the same id. The [LangGraph guide](../guides/langgraph.md) shows this with a graph interrupt.

## Next

- [State, steps and resume](../concepts/state.md) explains what is stored and how resume decides what to replay.
- [Persist, encrypt and purge state](../guides/persistence.md) covers Postgres, encryption at rest and retention.
