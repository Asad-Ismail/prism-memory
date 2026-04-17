from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontier_memory.travelplanner import (
    TravelPlannerPolicy,
    TravelMemoryPlanner,
    evaluate_policy,
    evaluate_rows,
    select_validation_rows,
)
from frontier_memory.travelplanner_search import load_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-file", type=str, default="")
    parser.add_argument("--policy-id", type=str, default="travelplanner_eval")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--include-diagnostics", action="store_true")
    parser.add_argument("--output-jsonl", type=str, default="")
    parser.add_argument("--summary-path", type=str, default="")
    args = parser.parse_args()

    if args.policy_file:
        policy = load_policy(args.policy_file)
    else:
        policy = TravelPlannerPolicy(policy_id=args.policy_id)

    if args.limit and args.stratified:
        rows = select_validation_rows(limit=args.limit, seed=args.seed)
        result = evaluate_rows(
            policy,
            rows,
            write_jsonl_path=args.output_jsonl or None,
            include_diagnostics=args.include_diagnostics,
        )
    elif args.limit:
        result = evaluate_policy(
            policy,
            limit=args.limit,
            write_jsonl_path=args.output_jsonl or None,
            include_diagnostics=args.include_diagnostics,
        )
    else:
        rows = TravelMemoryPlanner.load_validation()
        result = evaluate_rows(
            policy,
            rows,
            write_jsonl_path=args.output_jsonl or None,
            include_diagnostics=args.include_diagnostics,
        )

    payload = {
        "policy_id": result.policy.policy_id,
        "objective": result.objective,
        "scores": result.scores,
        "summary": result.policy.summary(),
        "output_path": result.output_path,
        "diagnostics_count": len(result.diagnostics),
    }
    if args.summary_path:
        target = Path(args.summary_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
