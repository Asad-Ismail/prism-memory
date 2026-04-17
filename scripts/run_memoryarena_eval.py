from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontier_memory.memoryarena_benchmark import MemoryArenaTravelPolicy, evaluate_group_travel_planner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-rows", type=int, default=20)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--disable-archive", action="store_true")
    args = parser.parse_args()

    policy = MemoryArenaTravelPolicy(
        policy_id=f"memoryarena_group_travel_r{args.limit_rows}",
        limit_rows=args.limit_rows,
        use_archive=not args.disable_archive,
    )
    result = evaluate_group_travel_planner(policy, write_json_path=args.output or None)
    print(
        json.dumps(
            {
                "policy_id": result.policy.policy_id,
                "num_rows": result.num_rows,
                "num_travelers": result.num_travelers,
                "exact_match_rate": result.exact_match_rate,
                "mutable_slot_accuracy": result.mutable_slot_accuracy,
                "summary": result.policy.summary(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
