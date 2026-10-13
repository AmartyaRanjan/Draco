"""Closed-loop load test against a running API.

  python scripts/load_test.py --base http://127.0.0.1:18080 --key $KEY --path /query --concurrency 8 --requests 64 [--generate]

Reports status-code mix, latency percentiles (successful requests only) and throughput. Run it twice for a full picture:
with production rate limits (expect 429s with Retry-After once the bucket drains) and with limits raised
(RATE_LIMIT_HEAVY=100000,100000) to see what the model/GPU can actually serve and when the LLM gate sheds load (503).
"""
import argparse
import asyncio
import collections
import json
import statistics
import time

import httpx

QUESTIONS = ["How many AI household agents were simulated in the healthcare policy simulator?",
             "What CGPA does Amartya Ranjan have?", "What is the physics timestep in the robotic arm report?",
             "Which n and alpha gave the best RMS error in the Lab 8 n-step TD experiment?",
             "What does RAPP stand for in the domain randomization roadmap?", "How many tilings does tile coding use in Lab 8?",
             "What is the torque limit of the benchmark arm?", "Which hackathon was Amartya Ranjan a finalist in?"]


async def worker(client, a, q, results):
    while True:
        try:
            i = q.get_nowait()
        except asyncio.QueueEmpty:
            return
        body = {"query": QUESTIONS[i % len(QUESTIONS)], "generate": a.generate}
        if a.path == "/chat":
            body = {"message": body["query"], "mode": "rag"}
        t0 = time.perf_counter()
        try:
            r = await client.post(a.path, json=body, headers={"x-api-key": a.key})
            results.append((r.status_code, time.perf_counter() - t0, r.headers.get("retry-after")))
        except Exception as exc:
            results.append((type(exc).__name__, time.perf_counter() - t0, None))


async def main(a):
    q, results = asyncio.Queue(), []
    for i in range(a.requests):
        q.put_nowait(i)
    async with httpx.AsyncClient(base_url=a.base, timeout=a.timeout) as client:
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(client, a, q, results) for _ in range(a.concurrency)))
        wall = time.perf_counter() - t0
    ok = sorted(t for s, t, _ in results if s == 200)
    pct = lambda p: round(ok[min(len(ok) - 1, int(p / 100 * len(ok)))], 2) if ok else None
    out = {"path": a.path, "generate": a.generate, "concurrency": a.concurrency, "requests": a.requests,
           "status": dict(collections.Counter(str(s) for s, _, _ in results)), "wall_s": round(wall, 1),
           "throughput_rps_ok": round(len(ok) / wall, 2), "latency_s_ok": {"p50": pct(50), "p95": pct(95), "max": round(ok[-1], 2) if ok else None},
           "retry_after_seen": sorted({r for s, _, r in results if r})}
    print(json.dumps(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18080")
    ap.add_argument("--key", required=True)
    ap.add_argument("--path", default="/query")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--generate", action="store_true")
    asyncio.run(main(ap.parse_args()))
