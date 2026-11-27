# Persist, encrypt and purge state

This guide covers the practical side of [state](../concepts/state.md): choosing and configuring a store, encrypting what it holds, and deleting old data on a schedule.

## Choose a store

Pass one to the harness, or let the harness open one from settings:

```python
from draco import Harness
from draco.state import SqliteStore

h = Harness(store=SqliteStore("state.db"))           # one host
```

```python
from draco.state.postgres import PostgresStore        # pip install "draco-harness[postgres]"

h = Harness(store=PostgresStore("postgresql://draco@db.internal/draco"))   # several processes
```

!!! warning "Keep the password out of the connection string"
    A `postgresql://` URL that contains a password is rejected in settings, because settings are logged. Put the password in `PGPASSWORD` or `~/.pgpass`, or pass a `draco.security.Secret`. `PostgresStore` names its tables `draco_executions`, `draco_events` and `draco_steps` (change the prefix with `table_prefix=`), so it can share a database with your own tables.

With no `store=`, `Harness` reads the settings:

| Setting | Environment variable | Effect |
|---|---|---|
| `state_url` | `DRACO_STATE_URL` | A `postgresql://` URL: use `PostgresStore`. Takes priority. |
| `state_path` | `DRACO_STATE_PATH` | A file: use `SqliteStore`. |
| neither | | In memory. Nothing survives a restart. |

Both SQL stores share one schema. A SQLite file created before the identity fields existed is migrated in place when opened.

## Encrypt at rest

Execution **inputs, outputs and errors, events and step data** can be encrypted with Fernet (AES with an HMAC, authenticated), using `pip install "draco-harness[crypto]"`. Ids, status, times, tenant, user, roles, usage and the policy stay in plaintext, because lookups and retention have to query them.

Keys come from a **secret**, not from code. The secret holds one or more comma-separated Fernet keys, **newest first**: the first key encrypts, and every key can decrypt.

<!-- doctest requires=cryptography -->
```python
import asyncio
import os
import sqlite3
import tempfile

from draco import Harness
from draco.security.crypto import FernetCipher
from draco.security.secrets import EnvSecretProvider
from draco.state import SqliteStore

async def save_card(ctx, card):
    return {"last4": card[-4:]}

async def main():
    # In production the key lives in your secret manager; here it is generated for the demo.
    os.environ["DEMO_STATE_KEY"] = FernetCipher.generate_key()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "state.db")
        cipher = FernetCipher.from_provider(EnvSecretProvider(), "DEMO_STATE_KEY")

        h = Harness(store=SqliteStore(path, cipher=cipher))
        h.register("save_card", save_card)
        r = await h.run("save_card", "4242424242424242")

        raw = sqlite3.connect(path).execute(
            "select input_json, output_json from executions where execution_id=?", (r.execution_id,)).fetchone()
        print("on disk:", raw[0][:7], raw[1][:7])

        saved = await h.store.get_execution(r.execution_id)
        print("read back:", saved.input, saved.output)
        h.store.close()

asyncio.run(main())
```

<!-- output -->
```text
on disk: enc:v1: enc:v1:
read back: 4242424242424242 {'last4': '4242'}
```

On disk the payload is ciphertext marked `enc:v1:`; through the store it reads back as normal. To turn encryption on through settings instead of code, set `DRACO_STATE_ENCRYPTION_KEY` to the **name** of the secret (for example `MY_STATE_KEYS`) and make that secret available to the secret provider (by default, an environment variable of that name). Setting an encryption key without a persistent store is a configuration error.

**Rotating a key.** Prepend a new key to the secret, restart so the cipher picks it up, call `await store.rotate_encryption()` to re-encrypt everything under the new key (it also encrypts any plaintext left from before you enabled encryption), and only then remove the old key. Removing it first makes old records unreadable.

!!! warning "Why you want this"
    Draco redacts *events*, but the execution record keeps your run's input and output exactly as given, because resume needs them. If a workload receives a card number, a token or a customer message, that value is in the store. Encryption protects it on disk and in backups; it does not protect it from code that has the key.

## Purge old data

Finished runs and old events pile up. Retention deletes them on a schedule, and it **never** touches `waiting` or `running` executions or their descendants, because they can still be resumed. Purging a completed run also forgets its idempotency key, so a retry with that key after the window runs again.

<!-- doctest -->
```python
import asyncio
import time

from draco import DracoSettings, Harness, Waiting
from draco.state import apply_retention

async def finish(ctx, x):
    return x

async def wait_for_human(ctx, x):
    return Waiting("needs approval", x)

async def main():
    h = Harness()                                      # the default in-memory store supports retention too
    h.register("finish", finish)
    h.register("wait", wait_for_human)
    done = await h.run("finish", 1)
    waiting = await h.run("wait", 2)

    settings = DracoSettings(execution_retention_days=1)
    two_days_from_now = time.time() + 2 * 86400        # pretend time has passed
    result = await apply_retention(h.store, settings, now=two_days_from_now)
    print(result)

    print("completed run kept:", await h.store.get_execution(done.execution_id) is not None)
    print("waiting run kept:", await h.store.get_execution(waiting.execution_id) is not None)

asyncio.run(main())
```

<!-- output -->
```text
PurgeResult(executions=1, events=3, steps=0)
completed run kept: False
waiting run kept: True
```

The windows come from settings: `execution_retention_days` (`DRACO_EXECUTION_RETENTION_DAYS`) removes finished executions with their events and steps, and `event_retention_days` (`DRACO_EVENT_RETENTION_DAYS`) removes only events older than that while keeping the execution records. `apply_retention` does nothing when no window is set.

Run it periodically: from cron, a systemd timer, a Kubernetes `CronJob`, or a background task in your service. The Draco repository includes a ready-made command, [`scripts/draco_retention.py`](https://github.com/AmartyaRanjan/Draco/blob/main/scripts/draco_retention.py), that reads the same `DRACO_*` variables and prints what it deleted as JSON. It is safe to run while harnesses are writing. For SQLite, pass `--vacuum` to hand the freed space back to the filesystem.

## Failures of the store

If saving a record fails, Draco logs it and **keeps the result it computed**, so a store outage does not fail a run that otherwise succeeded. Create the harness with `strict_persistence=True` if you would rather fail: a run that completed but could not be recorded is then reported as `failed`. Choose strict when losing the record means losing the ability to resume or to prove what happened.

## See also

- [State, steps and resume](../concepts/state.md)
- [Configuration reference](../reference/configuration.md): every `DRACO_*` setting
