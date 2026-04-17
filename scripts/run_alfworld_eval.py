from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontier_memory.alfworld_benchmark import AlfworldPolicy, evaluate_retrieval_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["valid_seen", "valid_unseen"], default="valid_seen")
    parser.add_argument("--train-games", type=int, default=64)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    policy = AlfworldPolicy(
        policy_id=f"alfworld_retrieval_t{args.train_games}_e{args.eval_games}",
        train_games=args.train_games,
        eval_games=args.eval_games,
        max_steps=args.max_steps,
    )
    result = evaluate_retrieval_policy(
        policy,
        split=args.split,
        write_json_path=args.output or None,
    )
    print(
        json.dumps(
            {
                "policy_id": result.policy.policy_id,
                "split": result.split,
                "num_games": result.num_games,
                "success_rate": result.success_rate,
                "mean_steps": result.mean_steps,
                "summary": result.policy.summary(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
