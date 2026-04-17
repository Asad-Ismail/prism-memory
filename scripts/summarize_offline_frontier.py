from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_TRAVELPLANNER = Path("logs/runs/travelplanner_bootstrap_v0_validation_scores.json")
DEFAULT_ALFWORLD_SEEN = Path("logs/runs/alfworld_seen_eval20_t64_v5.json")
DEFAULT_ALFWORLD_UNSEEN = Path("logs/runs/alfworld_unseen_eval20_t64_v1.json")
DEFAULT_MEMORYARENA = Path("logs/runs/memoryarena_suite_v2.json")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--travelplanner", type=str, default=str(DEFAULT_TRAVELPLANNER))
    parser.add_argument("--alfworld-seen", type=str, default=str(DEFAULT_ALFWORLD_SEEN))
    parser.add_argument("--alfworld-unseen", type=str, default=str(DEFAULT_ALFWORLD_UNSEEN))
    parser.add_argument("--memoryarena", type=str, default=str(DEFAULT_MEMORYARENA))
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    travelplanner = _load_json(Path(args.travelplanner))
    alfworld_seen = _load_json(Path(args.alfworld_seen))
    alfworld_unseen = _load_json(Path(args.alfworld_unseen))
    memoryarena = _load_json(Path(args.memoryarena))

    payload = {
        "travelplanner": {
            "policy_id": travelplanner["policy_id"],
            "final_pass_rate": travelplanner["scores"]["Final Pass Rate"],
            "hard_constraint_macro_pass_rate": travelplanner["scores"]["Hard Constraint Macro Pass Rate"],
            "commonsense_constraint_macro_pass_rate": travelplanner["scores"]["Commonsense Constraint Macro Pass Rate"],
        },
        "alfworld": {
            "policy_id": alfworld_seen["policy_id"],
            "valid_seen_success_rate": alfworld_seen["success_rate"],
            "valid_seen_mean_steps": alfworld_seen["mean_steps"],
            "valid_unseen_success_rate": alfworld_unseen["success_rate"],
            "valid_unseen_mean_steps": alfworld_unseen["mean_steps"],
        },
        "memoryarena": {
            "overall_task_exact_match_rate": memoryarena["overall_task_exact_match_rate"],
            "config_scores": memoryarena["config_scores"],
        },
        "all_public_benchmarks_at_ceiling": (
            travelplanner["scores"]["Final Pass Rate"] == 1.0
            and alfworld_seen["success_rate"] == 1.0
            and alfworld_unseen["success_rate"] == 1.0
            and memoryarena["overall_task_exact_match_rate"] == 1.0
        ),
    }

    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
