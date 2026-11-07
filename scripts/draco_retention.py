#!/usr/bin/env python
"""Apply Draco's retention policy to a persisted state store (SQLite or Postgres).

Settings come from DRACO_* environment variables (DracoSettings.from_env) and can be overridden here:

  python scripts/draco_retention.py --state-path app/storage/draco.db --execution-days 30 --event-days 7
  DRACO_STATE_URL=postgresql://draco@db/draco DRACO_EVENT_RETENTION_DAYS=14 python scripts/draco_retention.py

Executions in `waiting` or `running` state (and their descendants) are never deleted. Prints the purge
counts as JSON. Exit codes: 0 done, 2 configuration problem (no store or no retention window).
Schedule it (cron/systemd timer/Kubernetes CronJob); it is safe to run while harnesses are writing.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run from a checkout without installing

from pydantic import ValidationError  # noqa: E402

from draco.core import ConfigurationError, DracoError, DracoSettings  # noqa: E402
from draco.harness import open_store  # noqa: E402
from draco.state import apply_retention  # noqa: E402


def parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--state-path", help="SQLite state file (DRACO_STATE_PATH)")
    p.add_argument("--state-url", help="postgresql:// URL (DRACO_STATE_URL); password via PGPASSWORD")
    p.add_argument("--execution-days", type=float, help="purge finished executions older than N days (DRACO_EXECUTION_RETENTION_DAYS)")
    p.add_argument("--event-days", type=float, help="purge events older than N days (DRACO_EVENT_RETENTION_DAYS)")
    p.add_argument("--vacuum", action="store_true", help="SQLite only: VACUUM afterwards to return space to the filesystem")
    return p.parse_args(argv)


async def run(args: argparse.Namespace, env: Mapping[str, str]) -> dict[str, object]:
    base = DracoSettings.from_env(env)
    overrides = {k: v for k, v in {"state_path": args.state_path, "state_url": args.state_url,
                                   "execution_retention_days": args.execution_days,
                                   "event_retention_days": args.event_days}.items() if v is not None}
    try:
        settings = DracoSettings.model_validate({**base.model_dump(), **overrides})
    except ValidationError as exc:
        raise ConfigurationError("invalid option: " + "; ".join(
            f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())) from exc
    if not (settings.state_path or settings.state_url):
        raise ConfigurationError("no persistent store: set --state-path/--state-url or DRACO_STATE_PATH/DRACO_STATE_URL")
    if settings.execution_retention_days is None and settings.event_retention_days is None:
        raise ConfigurationError("no retention window: set --execution-days/--event-days or "
                                 "DRACO_EXECUTION_RETENTION_DAYS/DRACO_EVENT_RETENTION_DAYS")
    # purge never decrypts payloads, so an encrypted store does not need its key here
    store = open_store(settings.model_copy(update={"state_encryption_key": None}))
    try:
        res = await apply_retention(store, settings)
        if args.vacuum and hasattr(store, "vacuum"):
            await store.vacuum()
    finally:
        store.close()
    return {"executions": res.executions, "events": res.events, "steps": res.steps,
            "execution_retention_days": settings.execution_retention_days,
            "event_retention_days": settings.event_retention_days,
            "backend": "postgres" if settings.state_url else "sqlite"}


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    args = parse(argv)
    try:
        out = asyncio.run(run(args, os.environ if env is None else env))
    except ConfigurationError as exc:
        print(f"draco_retention: {exc.message}", file=sys.stderr)
        return 2
    except DracoError as exc:
        print(f"draco_retention: {type(exc).__name__}: {exc.message}", file=sys.stderr)
        return 1
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
