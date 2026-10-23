import asyncio
import threading
import time
import unittest

from app.core.blocking import run_blocking


class BoundedBlockingTests(unittest.TestCase):
    def test_timeout_does_not_release_slot_until_worker_finishes(self):
        finished = threading.Event()

        def work():
            time.sleep(0.05)
            finished.set()
            return "done"

        async def run():
            with self.assertRaises(asyncio.TimeoutError):
                await run_blocking(work, timeout=0.001)
            self.assertFalse(finished.is_set())
            await asyncio.sleep(0.08)
            self.assertTrue(finished.is_set())

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
