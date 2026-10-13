"""One-shot snapshot of vLLM Prometheus metrics (loopback) + GPU memory.
PYTHONPATH=. python scripts/vllm_monitor.py [--url http://127.0.0.1:18001]"""
import argparse
import json
import re
import subprocess
import urllib.request

KEYS = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
        "vllm:num_preemptions_total", "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
        "vllm:request_success_total", "vllm:prompt_tokens_total", "vllm:generation_tokens_total")


def snapshot(url):
    text = urllib.request.urlopen(f"{url}/metrics", timeout=5).read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for k in KEYS:
            if line.startswith(k):
                out[k] = out.get(k, 0.0) + float(line.rsplit(" ", 1)[1])
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    used, total = (int(x) for x in gpu.split(","))
    return {"vllm": out, "gpu_mib": {"used": used, "total": total}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18001")
    print(json.dumps(snapshot(ap.parse_args().url), indent=1))
