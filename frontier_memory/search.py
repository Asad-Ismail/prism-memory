from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .benchmarks import dump_suite_result, run_synthetic_suite
from .config import CandidateConfig, load_candidate
from .utils import clamp


class CandidateMutator:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def mutate(self, parent: CandidateConfig, *, child_index: int) -> tuple[CandidateConfig, List[str]]:
        child = parent.clone()
        mutations: List[str] = []
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        child.raw["parent_ids"] = [parent.candidate_id]
        child.raw["status"] = "trial"
        child.raw["candidate_id"] = f"{parent.candidate_id}_iter_{timestamp}_{child_index:02d}"

        mutation_choices = [
            self._mutate_merge_similarity,
            self._mutate_top_k,
            self._mutate_retrieval_order,
            self._mutate_passes,
            self._mutate_controller,
            self._mutate_reliability,
        ]
        self.rng.shuffle(mutation_choices)
        selected = mutation_choices[: self.rng.randint(2, 4)]
        for operation in selected:
            note = operation(child)
            if note:
                mutations.append(note)
        child.raw["pivot_reason"] = "auto search iteration"
        return child, mutations

    def _mutate_merge_similarity(self, candidate: CandidateConfig) -> str:
        value = float(
            candidate.get(
                "consolidation",
                "thresholds",
                "merge_similarity",
                default=0.90,
            )
        )
        delta = self.rng.choice([-0.04, -0.02, 0.02, 0.04])
        updated = round(clamp(value + delta, 0.70, 0.99), 2)
        candidate.set(["consolidation", "thresholds", "merge_similarity"], updated)
        return f"merge_similarity={updated}"

    def _mutate_top_k(self, candidate: CandidateConfig) -> str:
        store = self.rng.choice(["procedural", "semantic", "episodic"])
        value = int(candidate.get("retrieval", "top_k", store, default=3))
        delta = self.rng.choice([-2, -1, 1, 2])
        updated = max(1, min(12, value + delta))
        candidate.set(["retrieval", "top_k", store], updated)
        return f"top_k.{store}={updated}"

    def _mutate_retrieval_order(self, candidate: CandidateConfig) -> str:
        order = list(candidate.get("retrieval", "order", default=["procedural", "semantic", "episodic"]))
        self.rng.shuffle(order)
        candidate.set(["retrieval", "order"], order)
        return f"retrieval.order={'/'.join(order)}"

    def _mutate_passes(self, candidate: CandidateConfig) -> str:
        passes = list(candidate.get("consolidation", "passes", default=[]))
        optional = [
            "extract_negative_constraints",
            "mark_critical_vs_noise",
            "merge_duplicates",
            "split_overbroad_routines",
            "compose_playbooks",
            "deprecate_stale",
            "compact_summaries",
        ]
        target = self.rng.choice(optional)
        if target in passes:
            passes.remove(target)
            action = "drop"
        else:
            passes.append(target)
            action = "add"
        candidate.set(["consolidation", "passes"], passes)
        return f"{action}_pass:{target}"

    def _mutate_controller(self, candidate: CandidateConfig) -> str:
        architecture = candidate.get("controller", "architecture", default="single_agent")
        if architecture == "single_agent":
            updated = "orchestrator_worker"
            use_policy = "favor_decomposition"
        else:
            updated = "single_agent"
            use_policy = "inject_minimal_supporting_context"
        candidate.set(["controller", "architecture"], updated)
        candidate.set(["controller", "use_policy"], use_policy)
        return f"controller={updated}"

    def _mutate_reliability(self, candidate: CandidateConfig) -> str:
        alpha = float(candidate.get("memory", "procedural", "reliability", "prior_alpha", default=1.0))
        beta = float(candidate.get("memory", "procedural", "reliability", "prior_beta", default=1.0))
        alpha = round(clamp(alpha + self.rng.choice([-0.5, 0.5, 1.0]), 0.5, 4.0), 2)
        beta = round(clamp(beta + self.rng.choice([-0.5, 0.5, 1.0]), 0.5, 4.0), 2)
        candidate.set(["memory", "procedural", "reliability", "prior_alpha"], alpha)
        candidate.set(["memory", "procedural", "reliability", "prior_beta"], beta)
        return f"reliability=beta({alpha},{beta})"


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _append_journal(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + "\n".join(lines) + "\n")


def run_search_iteration(
    base_candidate_path: str | Path,
    *,
    num_children: int = 4,
    dataset_size: int = 4,
    seed: int = 42,
    logs_dir: str | Path = "logs",
    generated_dir: str | Path = "candidates/generated",
    champion_path: str | Path = "candidates/champion_latest.yaml",
) -> Dict[str, Any]:
    base_candidate = load_candidate(base_candidate_path)
    log_root = Path(logs_dir)
    results_dir = log_root / "runs"
    candidate_output_dir = Path(generated_dir)
    mutator = CandidateMutator(seed)

    evaluations: List[Dict[str, Any]] = []

    base_result = run_synthetic_suite(base_candidate, dataset_size=dataset_size, seed=seed)
    evaluations.append(
        {
            "candidate": base_candidate,
            "result": base_result,
            "mutations": ["baseline"],
        }
    )

    for child_index in range(1, num_children + 1):
        child, mutations = mutator.mutate(base_candidate, child_index=child_index)
        child_path = candidate_output_dir / f"{child.candidate_id}.yaml"
        child.dump(child_path)
        result = run_synthetic_suite(child, dataset_size=dataset_size, seed=seed)
        evaluations.append(
            {
                "candidate": child,
                "candidate_path": str(child_path),
                "result": result,
                "mutations": mutations,
            }
        )

    evaluations.sort(key=lambda item: item["result"]["global_score"], reverse=True)
    best = evaluations[0]
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    best["candidate"].dump(champion_path)

    for item in evaluations:
        result = item["result"]
        run_name = f"{result['candidate_id']}.json"
        dump_suite_result(results_dir / run_name, result)
        _append_jsonl(
            log_root / "experiments.jsonl",
            {
                "date": timestamp,
                "event": "search_iteration",
                "candidate_id": result["candidate_id"],
                "parent_ids": item["candidate"].parent_ids,
                "mutations": item["mutations"],
                "global_score": result["global_score"],
                "benchmarks": {
                    key: value["overall"]
                    for key, value in result["benchmarks"].items()
                },
                "summary": item["candidate"].summary(),
                "decision": "champion" if result["candidate_id"] == best["result"]["candidate_id"] else "observed",
            },
        )

    _append_journal(
        log_root / "journal.md",
        [
            f"## {timestamp[:10]} - Search Iteration",
            "",
            f"Base candidate: `{base_candidate.candidate_id}`",
            f"Dataset size per benchmark: `{dataset_size}`",
            f"Children explored: `{num_children}`",
            "",
            f"Champion: `{best['result']['candidate_id']}` with global score `{best['result']['global_score']}`",
            "",
            "Mutations tested:",
            *[f"- `{item['result']['candidate_id']}`: {', '.join(item['mutations'])}" for item in evaluations],
        ],
    )

    return {
        "timestamp": timestamp,
        "best_candidate_id": best["result"]["candidate_id"],
        "best_global_score": best["result"]["global_score"],
        "champion_path": str(champion_path),
        "evaluations": [
            {
                "candidate_id": item["result"]["candidate_id"],
                "global_score": item["result"]["global_score"],
                "mutations": item["mutations"],
            }
            for item in evaluations
        ],
    }
