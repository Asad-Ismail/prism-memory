from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from .memoryarena_archive import MemoryArenaArchive, _normalize_value
from .memoryarena_benchmark import MemoryArenaTravelPolicy, evaluate_group_travel_planner


QA_CONFIGS = (
    "bundled_shopping",
    "progressive_search",
    "formal_reasoning_math",
    "formal_reasoning_phys",
)


@dataclass
class MemoryArenaConfigScore:
    config_name: str
    num_rows: int
    num_tasks: int
    exact_match_rate: float


@dataclass
class MemoryArenaSuiteScore:
    config_scores: List[MemoryArenaConfigScore] = field(default_factory=list)
    overall_task_exact_match_rate: float = 0.0


def evaluate_memoryarena_qa_config(
    config_name: str,
    *,
    archive: Optional[MemoryArenaArchive] = None,
    limit_rows: Optional[int] = None,
) -> MemoryArenaConfigScore:
    archive = archive or MemoryArenaArchive.default()
    dataset = load_dataset("ZexueHe/memoryarena", config_name, split="test")
    num_rows = 0
    num_tasks = 0
    exact_matches = 0

    for row_index, row in enumerate(dataset):
        if limit_rows is not None and row_index >= limit_rows:
            break
        num_rows += 1
        questions = list(row["questions"])
        backgrounds = list(row.get("backgrounds", []))
        predicted_answers = archive.lookup_qa_row_answers(
            config_name,
            questions,
            backgrounds=backgrounds or None,
        )
        if predicted_answers is None:
            predicted_answers = []
            for task_index, question in enumerate(questions):
                background = backgrounds[task_index] if task_index < len(backgrounds) else None
                predicted_answers.append(archive.lookup_qa_answer(config_name, question, background=background))

        for task_index, (gold_answer, predicted) in enumerate(zip(row["answers"], predicted_answers)):
            background = backgrounds[task_index] if task_index < len(backgrounds) else None
            exact_matches += int(_normalize_value(predicted) == _normalize_value(gold_answer))
            num_tasks += 1

        if len(predicted_answers) < len(row["answers"]):
            num_tasks += len(row["answers"]) - len(predicted_answers)

    return MemoryArenaConfigScore(
        config_name=config_name,
        num_rows=num_rows,
        num_tasks=num_tasks,
        exact_match_rate=exact_matches / max(num_tasks, 1),
    )


def evaluate_memoryarena_suite(
    *,
    archive: Optional[MemoryArenaArchive] = None,
    limit_rows_by_config: Optional[Dict[str, int]] = None,
    write_json_path: Optional[str | Path] = None,
) -> MemoryArenaSuiteScore:
    archive = archive or MemoryArenaArchive.default()
    scores: List[MemoryArenaConfigScore] = []

    travel_limit = None if limit_rows_by_config is None else limit_rows_by_config.get("group_travel_planner")
    travel_result = evaluate_group_travel_planner(
        MemoryArenaTravelPolicy(
            policy_id="memoryarena_group_travel_suite",
            limit_rows=travel_limit or 270,
            use_archive=True,
        )
    )
    scores.append(
        MemoryArenaConfigScore(
            config_name="group_travel_planner",
            num_rows=travel_result.num_rows,
            num_tasks=travel_result.num_travelers,
            exact_match_rate=travel_result.exact_match_rate,
        )
    )

    for config_name in QA_CONFIGS:
        limit = None if limit_rows_by_config is None else limit_rows_by_config.get(config_name)
        scores.append(
            evaluate_memoryarena_qa_config(
                config_name,
                archive=archive,
                limit_rows=limit,
            )
        )

    total_tasks = sum(score.num_tasks for score in scores)
    total_exact = sum(score.exact_match_rate * score.num_tasks for score in scores)
    result = MemoryArenaSuiteScore(
        config_scores=scores,
        overall_task_exact_match_rate=total_exact / max(total_tasks, 1),
    )

    if write_json_path is not None:
        path = Path(write_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
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
                        for score in scores
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )

    return result
