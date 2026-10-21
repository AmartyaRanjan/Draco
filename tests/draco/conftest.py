"""Run `async def test_*` functions without a pytest plugin."""
import asyncio
import inspect

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    fn = pyfuncitem.obj
    if inspect.iscoroutinefunction(fn):
        args = {a: pyfuncitem.funcargs[a] for a in pyfuncitem._fixtureinfo.argnames}
        asyncio.run(fn(**args))
        return True
    return None
