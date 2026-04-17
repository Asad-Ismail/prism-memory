from __future__ import annotations

import json
import random
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .travelplanner import (
    TravelMemoryPlanner,
    TravelPlannerPolicy,
    TravelPlannerOfficialEvaluator,
    evaluate_rows,
    select_validation_rows,
)
from .utils import clamp


def dump_policy(policy: TravelPlannerPolicy, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(asdict(policy), sort_keys=False))


def load_policy(path: str | Path) -> TravelPlannerPolicy:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    return TravelPlannerPolicy(**payload)


class TravelPlannerPolicyMutator:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def mutate(self, parent: TravelPlannerPolicy, *, child_index: int) -> tuple[TravelPlannerPolicy, List[str]]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        child = replace(parent, policy_id=f"{parent.policy_id}_iter_{timestamp}_{child_index:02d}")
        operations = [
            self._mutate_transport_mode,
            self._mutate_city_pool_size,
            self._mutate_city_weights,
            self._mutate_cuisine_weight,
            self._mutate_restaurant_preferences,
            self._mutate_flight_bias,
            self._mutate_travel_dinner_policy,
        ]
        self.rng.shuffle(operations)
        selected = operations[: self.rng.randint(2, 4)]
        notes: List[str] = []
        for operation in selected:
            notes.append(operation(child))
        return child, notes

    def _mutate_transport_mode(self, policy: TravelPlannerPolicy) -> str:
        choices = ["auto", "air_taxi", "self_driving_only"]
        updated = self.rng.choice([choice for choice in choices if choice != policy.transport_mode])
        policy.transport_mode = updated
        return f"transport_mode={updated}"

    def _mutate_city_pool_size(self, policy: TravelPlannerPolicy) -> str:
        delta = self.rng.choice([-3, -2, 2, 3])
        updated = max(4, min(18, policy.city_pool_size + delta))
        policy.city_pool_size = updated
        return f"city_pool_size={updated}"

    def _mutate_city_weights(self, policy: TravelPlannerPolicy) -> str:
        target = self.rng.choice(
            [
                "city_cost_weight",
                "city_transport_weight",
                "city_restaurant_weight",
                "city_attraction_weight",
            ]
        )
        delta = self.rng.choice([-0.3, -0.2, 0.2, 0.3])
        updated = round(clamp(getattr(policy, target) + delta, 0.0, 3.0), 2)
        setattr(policy, target, updated)
        return f"{target}={updated}"

    def _mutate_cuisine_weight(self, policy: TravelPlannerPolicy) -> str:
        target = self.rng.choice(["city_cuisine_weight", "required_cuisine_bonus"])
        if target == "city_cuisine_weight":
            delta = self.rng.choice([-0.4, -0.2, 0.2, 0.4])
            updated = round(clamp(policy.city_cuisine_weight + delta, 0.0, 3.0), 2)
            policy.city_cuisine_weight = updated
        else:
            delta = self.rng.choice([-20.0, -10.0, 10.0, 20.0])
            updated = round(clamp(policy.required_cuisine_bonus + delta, 0.0, 120.0), 2)
            policy.required_cuisine_bonus = updated
        return f"{target}={updated}"

    def _mutate_restaurant_preferences(self, policy: TravelPlannerPolicy) -> str:
        updated = round(clamp(policy.restaurant_rating_weight + self.rng.choice([-1.0, -0.5, 0.5, 1.0]), 0.0, 5.0), 2)
        policy.restaurant_rating_weight = updated
        return f"restaurant_rating_weight={updated}"

    def _mutate_flight_bias(self, policy: TravelPlannerPolicy) -> str:
        updated = round(clamp(policy.prefer_flight_multiplier + self.rng.choice([-0.15, -0.1, 0.1, 0.15]), 0.7, 1.3), 2)
        policy.prefer_flight_multiplier = updated
        return f"prefer_flight_multiplier={updated}"

    def _mutate_travel_dinner_policy(self, policy: TravelPlannerPolicy) -> str:
        choices = ["adaptive", "always", "never"]
        updated = self.rng.choice([choice for choice in choices if choice != policy.travel_dinner_policy])
        policy.travel_dinner_policy = updated
        return f"travel_dinner_policy={updated}"


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _append_journal(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + "\n".join(lines) + "\n")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def run_travelplanner_search(
    base_policy: TravelPlannerPolicy,
    *,
    num_children: int = 8,
    dev_limit: int = 30,
    full_limit: int | None = None,
    full_eval_top_n: int = 3,
    seed: int = 42,
    logs_dir: str | Path = "logs",
    generated_dir: str | Path = "candidates/generated",
    champion_path: str | Path = "candidates/travelplanner_champion.yaml",
) -> Dict[str, Any]:
    log_root = Path(logs_dir)
    results_dir = log_root / "runs"
    candidate_dir = Path(generated_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    mutator = TravelPlannerPolicyMutator(seed)
    planner = TravelMemoryPlanner()
    evaluator = TravelPlannerOfficialEvaluator()

    dev_rows = select_validation_rows(limit=dev_limit, seed=seed)
    if full_limit is None:
        full_rows = TravelMemoryPlanner.load_validation()
    elif full_limit >= 180:
        full_rows = TravelMemoryPlanner.load_validation(limit=full_limit)
    else:
        full_rows = select_validation_rows(limit=full_limit, seed=seed + 1)

    candidates: List[tuple[TravelPlannerPolicy, List[str], str | None]] = [(base_policy, ["baseline"], None)]
    for child_index in range(1, num_children + 1):
        child, notes = mutator.mutate(base_policy, child_index=child_index)
        child_path = candidate_dir / f"{child.policy_id}.yaml"
        dump_policy(child, child_path)
        candidates.append((child, notes, str(child_path)))

    dev_results = []
    for policy, mutations, candidate_path in candidates:
        evaluation = evaluate_rows(policy, dev_rows, planner=planner, evaluator=evaluator)
        dev_results.append(
            {
                "policy": policy,
                "mutations": mutations,
                "candidate_path": candidate_path,
                "evaluation": evaluation,
            }
        )
        _append_jsonl(
            log_root / "experiments.jsonl",
            {
                "date": timestamp,
                "event": "travelplanner_dev_eval",
                "policy_id": policy.policy_id,
                "mutations": mutations,
                "objective": evaluation.objective,
                "scores": evaluation.scores,
                "summary": policy.summary(),
            },
        )

    dev_results.sort(key=lambda item: item["evaluation"].objective, reverse=True)
    finalists = dev_results[: max(1, min(full_eval_top_n, len(dev_results)))]

    full_results = []
    for item in finalists:
        policy = item["policy"]
        output_jsonl = results_dir / f"{policy.policy_id}_travelplanner_validation.jsonl"
        evaluation = evaluate_rows(
            policy,
            full_rows,
            write_jsonl_path=output_jsonl,
            planner=planner,
            evaluator=evaluator,
        )
        result_payload = {
            "policy_id": policy.policy_id,
            "objective": evaluation.objective,
            "scores": evaluation.scores,
            "summary": policy.summary(),
            "output_path": str(output_jsonl),
        }
        _write_json(results_dir / f"{policy.policy_id}_travelplanner_validation_scores.json", result_payload)
        _append_jsonl(
            log_root / "experiments.jsonl",
            {
                "date": timestamp,
                "event": "travelplanner_full_eval",
                "policy_id": policy.policy_id,
                "mutations": item["mutations"],
                "objective": evaluation.objective,
                "scores": evaluation.scores,
                "summary": policy.summary(),
            },
        )
        full_results.append(
            {
                "policy": policy,
                "mutations": item["mutations"],
                "evaluation": evaluation,
                "output_path": str(output_jsonl),
            }
        )

    full_results.sort(key=lambda item: item["evaluation"].objective, reverse=True)
    champion = full_results[0]
    champion_policy = champion["policy"]
    champion_jsonl = results_dir / f"{champion_policy.policy_id}_travelplanner_validation.jsonl"
    champion_detailed = evaluate_rows(
        champion_policy,
        full_rows,
        write_jsonl_path=champion_jsonl,
        include_diagnostics=True,
        planner=planner,
        evaluator=evaluator,
    )
    dump_policy(champion_policy, champion_path)
    _write_json(
        results_dir / f"{champion_policy.policy_id}_travelplanner_validation_diagnostics.json",
        {
            "policy_id": champion_policy.policy_id,
            "objective": champion_detailed.objective,
            "scores": champion_detailed.scores,
            "summary": champion_policy.summary(),
            "diagnostics": champion_detailed.diagnostics,
            "output_path": str(champion_jsonl),
        },
    )
    _append_jsonl(
        log_root / "experiments.jsonl",
        {
            "date": timestamp,
            "event": "travelplanner_champion",
            "policy_id": champion_policy.policy_id,
            "mutations": champion["mutations"],
            "objective": champion_detailed.objective,
            "scores": champion_detailed.scores,
            "summary": champion_policy.summary(),
            "output_path": str(champion_jsonl),
        },
    )
    _append_journal(
        log_root / "journal.md",
        [
            f"## {timestamp[:10]} - TravelPlanner Search",
            "",
            f"Base policy: `{base_policy.policy_id}`",
            f"Dev shard size: `{dev_limit}`",
            f"Full validation size: `{len(full_rows)}`",
            f"Children explored: `{num_children}`",
            "",
            f"Champion: `{champion_policy.policy_id}` with objective `{round(champion_detailed.objective, 4)}`",
            f"Champion final pass rate: `{round(champion_detailed.scores['Final Pass Rate'], 4)}`",
            f"Champion hard macro pass rate: `{round(champion_detailed.scores['Hard Constraint Macro Pass Rate'], 4)}`",
            "",
            "Dev leaderboard:",
            *[
                f"- `{item['policy'].policy_id}`: objective `{round(item['evaluation'].objective, 4)}` on dev shard"
                for item in dev_results[:5]
            ],
        ],
    )

    return {
        "timestamp": timestamp,
        "dev_limit": dev_limit,
        "full_limit": len(full_rows),
        "champion_policy_id": champion_policy.policy_id,
        "champion_objective": champion_detailed.objective,
        "champion_scores": champion_detailed.scores,
        "champion_path": str(champion_path),
        "evaluated_policy_ids": [item["policy"].policy_id for item in dev_results],
    }
