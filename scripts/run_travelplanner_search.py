from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontier_memory.travelplanner import TravelPlannerPolicy
from frontier_memory.travelplanner_search import load_policy, run_travelplanner_search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-file", type=str, default="")
    parser.add_argument("--num-children", type=int, default=8)
    parser.add_argument("--dev-limit", type=int, default=30)
    parser.add_argument("--full-limit", type=int, default=0)
    parser.add_argument("--full-eval-top-n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logs-dir", type=str, default="logs")
    parser.add_argument("--generated-dir", type=str, default="candidates/generated")
    parser.add_argument("--champion-path", type=str, default="candidates/travelplanner_champion.yaml")
    args = parser.parse_args()

    if args.policy_file:
        policy = load_policy(args.policy_file)
    else:
        policy = TravelPlannerPolicy(policy_id="travelplanner_bootstrap_v0")

    result = run_travelplanner_search(
        policy,
        num_children=args.num_children,
        dev_limit=args.dev_limit,
        full_limit=args.full_limit or None,
        full_eval_top_n=args.full_eval_top_n,
        seed=args.seed,
        logs_dir=args.logs_dir,
        generated_dir=args.generated_dir,
        champion_path=args.champion_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
