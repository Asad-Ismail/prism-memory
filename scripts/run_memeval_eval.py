#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from frontier_memory.memeval_adapter import (
    evaluate_candidate_on_benchmark,
    evaluate_registry_system_on_benchmark,
)
from frontier_memory.config import load_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frontier memory runtime on MemEval benchmarks.")
    parser.add_argument(
        "--benchmark",
        default="locomo",
        choices=("locomo", "longmemeval"),
        help="MemEval benchmark to run.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Benchmark split, e.g. oracle/s/m for LongMemEval.",
    )
    parser.add_argument(
        "--candidate",
        default="candidates/openai_memory_champion.yaml",
        help="Frontier-memory candidate YAML.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of conversations or questions to evaluate.",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip judge evaluation and report benchmark-native F1 only.",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Optional custom MemEval-format JSON file.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional output prefix. Writes <prefix>_summary.json and <prefix>_rows.jsonl.",
    )
    parser.add_argument(
        "--compare-systems",
        default="",
        help="Optional comma-separated MemEval baseline systems to run on the same sample, e.g. propmem.",
    )
    parser.add_argument(
        "--baseline-llm-model",
        default="gpt-5.2",
        help="LLM model for MemEval baseline systems such as PropMem.",
    )
    parser.add_argument(
        "--disable-memeval-tuning",
        action="store_true",
        help="Use the candidate exactly as written instead of applying conversational benchmark overrides.",
    )
    return parser.parse_args()


def dump_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    candidate = load_candidate(args.candidate)

    result = evaluate_candidate_on_benchmark(
        benchmark=args.benchmark,
        candidate=candidate,
        split=args.split,
        num_samples=args.num_samples,
        data_file=args.data_file,
        run_judge=not args.skip_judge,
        memeval_tuned=not args.disable_memeval_tuning,
    )

    compare_systems = [item.strip() for item in args.compare_systems.split(",") if item.strip()]
    comparisons: dict[str, dict] = {}
    for system_name in compare_systems:
        baseline = evaluate_registry_system_on_benchmark(
            benchmark=args.benchmark,
            system_name=system_name,
            llm_model=args.baseline_llm_model,
            split=args.split,
            num_samples=args.num_samples,
            data_file=args.data_file,
            run_judge=not args.skip_judge,
        )
        comparisons[system_name] = baseline.summary

    summary = {
        "benchmark": result.benchmark,
        "benchmark_name": result.benchmark_name,
        "split": result.split,
        "candidate_id": result.candidate_id,
        "candidate_path": str(Path(args.candidate).resolve()),
        "llm_model": result.llm_model,
        "summary": result.summary,
        "comparisons": comparisons,
    }

    if args.output_prefix:
        prefix = Path(args.output_prefix)
    else:
        split_tag = args.split or "default"
        prefix = Path("logs/runs") / f"memeval_{args.benchmark}_{split_tag}_{result.candidate_id}"

    summary_path = prefix.parent / f"{prefix.name}_summary.json"
    rows_path = prefix.parent / f"{prefix.name}_rows.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    dump_rows(rows_path, result.rows)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")
    print(f"Wrote {rows_path}")


if __name__ == "__main__":
    main()
