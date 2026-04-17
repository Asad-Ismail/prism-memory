from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontier_memory.benchmarks import dump_suite_result, run_synthetic_suite
from frontier_memory.config import load_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-candidate", type=str, default="candidates/champion_latest.yaml")
    parser.add_argument("--model", type=str, default="gpt-5.2")
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--dataset-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    candidate = load_candidate(args.base_candidate)
    candidate.set(("candidate_id",), f"{candidate.candidate_id}_{args.model}_{args.reasoning_effort}")
    candidate.set(("llm", "enabled"), True)
    candidate.set(("llm", "provider"), "openai")
    candidate.set(("llm", "model"), args.model)
    candidate.set(("llm", "reasoning_effort"), args.reasoning_effort)

    payload = run_synthetic_suite(
        candidate,
        dataset_size=args.dataset_size,
        seed=args.seed,
    )
    if args.output:
        dump_suite_result(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
