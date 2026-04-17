from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontier_memory.search import run_search_iteration


def main() -> None:
    parser = argparse.ArgumentParser(description="Mutate a candidate, evaluate children, and append logs.")
    parser.add_argument("--base-candidate", default="candidates/bootstrap_v0.yaml")
    parser.add_argument("--num-children", type=int, default=4)
    parser.add_argument("--dataset-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--generated-dir", default="candidates/generated")
    parser.add_argument("--champion-path", default="candidates/champion_latest.yaml")
    args = parser.parse_args()

    summary = run_search_iteration(
        args.base_candidate,
        num_children=args.num_children,
        dataset_size=args.dataset_size,
        seed=args.seed,
        logs_dir=args.logs_dir,
        generated_dir=args.generated_dir,
        champion_path=args.champion_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
