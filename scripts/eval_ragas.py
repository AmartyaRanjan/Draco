"""Run the RAGAS library on eval_results/judge_<tag>.jsonl with the local vLLM model as the judge.

  <eval-venv>/bin/python scripts/eval_ragas.py --tag as_written

Needs `pip install ragas langchain-openai "langchain-community<0.4"` in a SEPARATE environment (ragas pulls its own langchain
pins). The judge is the same Qwen3-4B that generated the answers, so scores are self-judged and noisy: treat them as a
smoke test of the library wiring and a lower-confidence second opinion, not as the headline numbers (see evals.md).
"""
import argparse
import json
import math
import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, FactualCorrectness, LLMContextPrecisionWithReference, LLMContextRecall
from ragas.run_config import RunConfig

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
env = {k: v for k, v in (l.strip().split("=", 1) for l in open(".env") if "=" in l and not l.startswith("#"))}
rows = [json.loads(l) for l in open(f"eval_results/judge_{a.tag}.jsonl")]
rows = [r for r in rows if r["answerable"] and r["answer"].strip() and not r["abstained"]]
if a.limit:
    rows = rows[:a.limit]
llm = LangchainLLMWrapper(ChatOpenAI(base_url=env["INFERENCE_BASE_URL"], api_key=env["INFERENCE_API_KEY"], model=env["INFERENCE_MODEL"],
                                     temperature=0, max_tokens=1500, timeout=240,
                                     extra_body={"chat_template_kwargs": {"enable_thinking": False}}))
ds = EvaluationDataset.from_list([{"user_input": r["question"], "response": r["answer"], "retrieved_contexts": r["contexts"],
                                   "reference": r["reference_answer"]} for r in rows])
metrics = [Faithfulness(), LLMContextPrecisionWithReference(), LLMContextRecall(), FactualCorrectness()]
res = evaluate(ds, metrics=metrics, llm=llm, run_config=RunConfig(timeout=240, max_workers=2, max_retries=2), show_progress=False)
df = res.to_pandas()
out = {"n": len(df), "judge": env["INFERENCE_MODEL"], "metrics": {}}
for c in df.columns:
    if df[c].dtype.kind in "fi" and c not in ("n",):
        vals = [v for v in df[c].tolist() if v is not None and not (isinstance(v, float) and math.isnan(v))]
        out["metrics"][c] = {"mean": round(sum(vals) / len(vals), 4) if vals else None, "n_scored": len(vals), "n_nan": len(df) - len(vals)}
Path(f"eval_results/ragas_{a.tag}.json").write_text(json.dumps({"summary": out, "rows": json.loads(df.to_json(orient="records", force_ascii=False))}, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1))
