from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontier_memory.memoryarena_suite import evaluate_memoryarena_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    result = evaluate_memoryarena_suite(write_json_path=args.output or None)
    print(
        json.dumps(
            {
                "overall_task_exact_match_rate": result.overall_task_exact_match_rate,
                "config_scores": [
                    {
                        "config_name": score.config_name,
                        "num_rows": score.num_rows,
                        "num_tasks": score.num_tasks,
                        "exact_match_rate": score.exact_match_rate,
                    }
                    for score in result.config_scores
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
