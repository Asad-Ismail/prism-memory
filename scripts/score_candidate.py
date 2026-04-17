#!/usr/bin/env python3
"""Score a candidate run from normalized benchmark outputs.

Input JSON format:
{
  "candidate_id": "bootstrap_v0",
  "candidate": {
    "alfworld_proxy": 0.71,
    "travelplanner_proxy": 0.62
  },
  "incumbent": {
    "alfworld_proxy": 0.68
  }
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TIER_WEIGHTS = {
    "tier0": {
        "alfworld_proxy": 0.35,
        "travelplanner_proxy": 0.35,
        "compiler_unit": 0.30,
    },
    "tier1": {
        "alfworld_seen_shard": 0.25,
        "alfworld_unseen_shard": 0.40,
        "travelplanner_dev_shard": 0.35,
    },
    "tier2": {
        "mem2actbench": 0.22,
        "memoryarena": 0.22,
        "memoryagentbench": 0.18,
        "proced_mem": 0.18,
        "shiftbench": 0.10,
        "efficiency_guardrail": 0.10,
    },
}

GLOBAL_WEIGHTS = {
    "tier0": 0.20,
    "tier1": 0.35,
    "tier2": 0.45,
}

HARD_GUARDRAILS = {
    "alfworld_unseen_shard": 0.02,
    "travelplanner_dev_shard": 0.02,
    "shiftbench": 0.03,
    "proced_mem": 0.03,
}


def weighted_average(scores: dict[str, float], weights: dict[str, float]) -> float | None:
    present = {k: v for k, v in weights.items() if k in scores}
    if not present:
        return None
    denom = sum(present.values())
    if denom == 0:
        return None
    total = sum(scores[k] * w for k, w in present.items())
    return total / denom


def compute_summary(candidate: dict[str, float], incumbent: dict[str, float]) -> dict:
    tier_scores = {}
    for tier, weights in TIER_WEIGHTS.items():
        tier_scores[tier] = weighted_average(candidate, weights)

    present_global_weights = {
        tier: weight
        for tier, weight in GLOBAL_WEIGHTS.items()
        if tier_scores.get(tier) is not None
    }
    global_score = None
    if present_global_weights:
        denom = sum(present_global_weights.values())
        total = sum(tier_scores[tier] * weight for tier, weight in present_global_weights.items())
        global_score = total / denom

    guardrails = {}
    for metric, allowed_drop in HARD_GUARDRAILS.items():
        if metric in candidate and metric in incumbent:
            guardrails[metric] = candidate[metric] >= incumbent[metric] - allowed_drop

    return {
        "tier_scores": tier_scores,
        "global_score": global_score,
        "guardrails": guardrails,
        "guardrails_pass": all(guardrails.values()) if guardrails else True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("--out", help="Optional output path for summary JSON")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text())
    candidate_scores = payload.get("candidate", {})
    incumbent_scores = payload.get("incumbent", {})

    summary = {
        "candidate_id": payload.get("candidate_id"),
        **compute_summary(candidate_scores, incumbent_scores),
    }

    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    else:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
