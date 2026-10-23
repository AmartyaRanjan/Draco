"""Bounded execution for unavoidable blocking work on async request paths."""

import asyncio
import functools
import os
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar


T = TypeVar("T")
_WORKERS = max(1, int(os.getenv("ASYNC_BLOCKING_WORKERS", "8")))
_EXECUTOR = ThreadPoolExecutor(max_workers=_WORKERS, thread_name_prefix="draco-blocking")
_SEMAPHORES = weakref.WeakKeyDictionary()
_SEMAPHORES_LOCK = threading.Lock()


def _slots() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    with _SEMAPHORES_LOCK:
        semaphore = _SEMAPHORES.get(loop)
        if semaphore is None:
            semaphore = asyncio.Semaphore(_WORKERS)
            _SEMAPHORES[loop] = semaphore
        return semaphore


async def run_blocking(func, *args, timeout: float | None = None, **kwargs) -> T:
    """Run blocking work with bounded admission and a cooperative timeout.

    Cancellation cannot interrupt a running Python/native worker thread. The
    slot is therefore released only when the underlying future actually
    finishes, preventing timed-out requests from spawning unbounded work.
    """
    slots = _slots()
    await slots.acquire()
    loop = asyncio.get_running_loop()
    try:
        future = loop.run_in_executor(_EXECUTOR, functools.partial(func, *args, **kwargs))
    except BaseException:
        slots.release()
        raise

    def release(_future):
        slots.release()

    future.add_done_callback(release)
    try:
        awaited = asyncio.shield(future)
        if timeout is None:
            return await awaited
        return await asyncio.wait_for(awaited, timeout)
    except asyncio.TimeoutError:
        # The callback owns slot release and will run when the worker finishes.
        raise
