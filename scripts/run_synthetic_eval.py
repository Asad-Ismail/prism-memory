from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontier_memory.benchmarks import dump_suite_result, run_synthetic_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local synthetic frontier-memory suite.")
    parser.add_argument("--candidate", default="candidates/bootstrap_v0.yaml")
    parser.add_argument("--dataset-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmarks", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    benchmark_names = [name.strip() for name in args.benchmarks.split(",") if name.strip()] or None
    result = run_synthetic_suite(
        args.candidate,
        dataset_size=args.dataset_size,
        seed=args.seed,
        benchmark_names=benchmark_names,
    )

    if args.output:
        dump_suite_result(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
