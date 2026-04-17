from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List

from .config import CandidateConfig, load_candidate
from .synthetic_benchmarks import (
    AttributionStressBenchmark,
    ContraFactBenchmark,
    EntityChainBenchmark,
    LowFrequencyBenchmark,
    SurpriseRecallBenchmark,
    TemporalDriftBenchmark,
)
from .system import HybridMemorySystem, simple_answer_score
from .types import BenchmarkResult


BenchmarkFactory = Callable[[int], object]


BENCHMARKS: Dict[str, BenchmarkFactory] = {
    "temporal_drift": lambda seed: TemporalDriftBenchmark(seed=seed, total_turns=80),
    "contra_fact": lambda seed: ContraFactBenchmark(seed=seed, total_turns=100),
    "attribution_stress": lambda seed: AttributionStressBenchmark(seed=seed, n_speakers=4),
    "entity_chain": lambda seed: EntityChainBenchmark(seed=seed, depths=[2, 3, 4]),
    "low_frequency": lambda seed: LowFrequencyBenchmark(seed=seed, high_freq_mentions=10),
    "surprise_recall": lambda seed: SurpriseRecallBenchmark(seed=seed, n_mundane=8, n_surprising=3),
}


BENCHMARK_WEIGHTS = {
    "temporal_drift": 0.22,
    "contra_fact": 0.20,
    "attribution_stress": 0.16,
    "entity_chain": 0.18,
    "low_frequency": 0.12,
    "surprise_recall": 0.12,
}


def evaluate_benchmark(
    candidate: CandidateConfig,
    benchmark_name: str,
    *,
    dataset_size: int,
    seed: int,
) -> BenchmarkResult:
    factory = BENCHMARKS[benchmark_name]
    benchmark = factory(seed)
    memory = HybridMemorySystem(candidate)
    per_type_scores: Dict[str, List[float]] = {}
    example_count = 0

    for _ in range(dataset_size):
        example = benchmark.generate_example()
        example_count += 1
        memory.reset()
        for turn in example.conversation:
            memory.ingest(turn)
        for qa in example.qa_pairs:
            prediction = memory.answer(qa.question)
            score = simple_answer_score(qa.question, prediction, qa.answer)
            per_type_scores.setdefault(qa.question_type, []).append(score)
            per_type_scores.setdefault("overall", []).append(score)

    metrics = {
        key: sum(values) / len(values)
        for key, values in per_type_scores.items()
        if values
    }
    return BenchmarkResult(
        benchmark=benchmark_name,
        overall=metrics.get("overall", 0.0),
        per_type=metrics,
        count=example_count,
    )


def run_synthetic_suite(
    candidate_or_path: CandidateConfig | str | Path,
    *,
    dataset_size: int = 8,
    seed: int = 42,
    benchmark_names: List[str] | None = None,
) -> Dict[str, object]:
    candidate = (
        candidate_or_path
        if isinstance(candidate_or_path, CandidateConfig)
        else load_candidate(candidate_or_path)
    )
    names = benchmark_names or list(BENCHMARKS.keys())
    results = {}
    objective = 0.0
    weight_sum = 0.0

    for offset, benchmark_name in enumerate(names):
        result = evaluate_benchmark(
            candidate,
            benchmark_name,
            dataset_size=dataset_size,
            seed=seed + offset,
        )
        results[benchmark_name] = {
            "overall": round(result.overall, 4),
            "count": result.count,
            "per_type": {k: round(v, 4) for k, v in result.per_type.items()},
        }
        weight = BENCHMARK_WEIGHTS.get(benchmark_name, 1.0)
        objective += result.overall * weight
        weight_sum += weight

    global_score = objective / weight_sum if weight_sum else 0.0
    return {
        "candidate_id": candidate.candidate_id,
        "global_score": round(global_score, 4),
        "benchmarks": results,
    }


def dump_suite_result(path: str | Path, payload: Dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
