from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontier_memory.benchmarks import dump_suite_result, run_synthetic_suite
from frontier_memory.config import CandidateConfig, load_candidate


def _parse_variants(raw_values: List[str]) -> list[tuple[str, str]]:
    variants = []
    for raw in raw_values:
        if ":" not in raw:
            raise ValueError(f"Variant must be MODEL:EFFORT, got {raw}")
        model, effort = raw.split(":", 1)
        variants.append((model.strip(), effort.strip()))
    return variants


def _candidate_with_llm(base: CandidateConfig, model: str, effort: str) -> CandidateConfig:
    candidate = base.clone()
    candidate.set(("candidate_id",), f"{base.candidate_id}_{model}_{effort}")
    candidate.set(("llm", "enabled"), True)
    candidate.set(("llm", "provider"), "openai")
    candidate.set(("llm", "model"), model)
    candidate.set(("llm", "reasoning_effort"), effort)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-candidate", type=str, default="candidates/champion_latest.yaml")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="MODEL:EFFORT, for example gpt-5.2:medium",
    )
    parser.add_argument("--dataset-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--champion-path", type=str, default="candidates/openai_memory_champion.yaml")
    args = parser.parse_args()

    variants = _parse_variants(args.variant or ["gpt-5.2:medium", "gpt-5.2:high", "gpt-5-mini:medium"])
    base = load_candidate(args.base_candidate)

    runs = []
    best_result = None
    best_candidate = None
    for model, effort in variants:
        candidate = _candidate_with_llm(base, model, effort)
        started = time.time()
        result = run_synthetic_suite(candidate, dataset_size=args.dataset_size, seed=args.seed)
        elapsed = round(time.time() - started, 2)
        runs.append(
            {
                "candidate_id": candidate.candidate_id,
                "model": model,
                "reasoning_effort": effort,
                "elapsed_seconds": elapsed,
                "global_score": result["global_score"],
                "benchmarks": result["benchmarks"],
            }
        )
        if best_result is None or result["global_score"] > best_result["global_score"]:
            best_result = result
            best_candidate = candidate

    assert best_result is not None and best_candidate is not None
    Path(args.champion_path).parent.mkdir(parents=True, exist_ok=True)
    best_candidate.dump(args.champion_path)

    payload = {
        "base_candidate": base.candidate_id,
        "dataset_size": args.dataset_size,
        "seed": args.seed,
        "runs": sorted(runs, key=lambda item: item["global_score"], reverse=True),
        "best_candidate_id": best_candidate.candidate_id,
        "best_model": best_candidate.get("llm", "model"),
        "best_reasoning_effort": best_candidate.get("llm", "reasoning_effort"),
        "best_global_score": best_result["global_score"],
        "champion_path": args.champion_path,
    }
    if args.output:
        dump_suite_result(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
