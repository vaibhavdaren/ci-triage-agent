"""Evaluation harness — the part that makes this an engineering project.

Metrics:
  retrieval_hit@6 : did any retrieved chunk come from the true failing test?
  triage_accuracy : did the agent name the correct failing test?
  groundedness    : did every cited chunk_id actually get retrieved?

Acts as a regression gate: exits 1 if triage_accuracy drops below
THRESHOLD, so prompt/retrieval changes can't silently regress.
Grow eval_set.jsonl with every real failure you diagnose — 30-50
cases from actual Fedora CI history is the target.

Usage:  python evals/run_evals.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ci_triage.agent import ask               # noqa: E402
from ci_triage.chunk import chunk_runs        # noqa: E402
from ci_triage.ingest import parse_directory  # noqa: E402
from ci_triage.retrieve import HybridIndex    # noqa: E402

THRESHOLD = 0.66
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    runs = parse_directory(ROOT / "sample_logs")
    index = HybridIndex()
    index.build(chunk_runs(runs))

    cases = [json.loads(l) for l in (ROOT / "evals/eval_set.jsonl").read_text().splitlines() if l.strip()]
    hits = correct = grounded = 0

    for case in cases:
        result = ask(index, case["question"])
        retrieved_tests = {cid.split("|")[1] for cid in result["_retrieved"] if "|" in cid}
        hit = case["expected_test"] in retrieved_tests
        ok = case["expected_test"] in result.get("failing_test", "")
        cited = set(result.get("evidence", []))
        ground = cited.issubset(set(result["_retrieved"]))
        hits += hit; correct += ok; grounded += ground
        print(f"[{'OK ' if ok else 'MISS'}] {case['question'][:60]!r} -> "
              f"{result.get('failing_test')} (hit@6={hit}, grounded={ground})")

    n = len(cases)
    acc, hit_rate, ground_rate = correct / n, hits / n, grounded / n
    print(f"\nretrieval_hit@6={hit_rate:.2f}  triage_accuracy={acc:.2f}  "
          f"groundedness={ground_rate:.2f}  (n={n}, gate={THRESHOLD})")
    if acc < THRESHOLD:
        print("REGRESSION GATE FAILED")
        return 1
    print("gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
