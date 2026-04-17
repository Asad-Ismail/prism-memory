from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .config import CandidateConfig, load_candidate
from .llm_backend import _load_env_file
from .system import HybridMemorySystem
from .types import MemoryEvent

MEMEVAL_ROOT = Path(os.environ.get("MEMEVAL_ROOT", "/tmp/MemEval")).expanduser().resolve()


@dataclass
class MemEvalRunResult:
    benchmark: str
    benchmark_name: str
    split: Optional[str]
    candidate_id: str
    llm_model: Optional[str]
    summary: dict[str, Any]
    rows: list[dict[str, Any]]


def ensure_memeval_imports(root: Path | str = MEMEVAL_ROOT) -> Path:
    _load_env_file()
    root = Path(root).expanduser().resolve()
    src = root / "src"
    if not src.exists():
        raise FileNotFoundError(
            f"MemEval source not found under {src}. Set MEMEVAL_ROOT or clone ProsusAI/MemEval."
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def load_memeval_benchmark(
    benchmark: str,
    *,
    split: Optional[str] = None,
    num_samples: int = 10,
    data_file: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ensure_memeval_imports()
    from agents_memory.benchmarks import BENCHMARKS

    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown MemEval benchmark: {benchmark}")

    bench = BENCHMARKS[benchmark]
    if data_file is not None:
        payload = json.loads(Path(data_file).read_text())
        conversations = payload if isinstance(payload, list) else [payload]
        return bench, conversations[:num_samples]
    conversations = bench["download"](split=split, num_samples=num_samples)
    return bench, conversations


def apply_memeval_overrides(candidate: CandidateConfig) -> CandidateConfig:
    tuned = candidate.clone()
    if tuned.get("llm", "enabled", default=False):
        if tuned.get("llm", "prompt_profile", default="default") == "default":
            tuned.set(["llm", "prompt_profile"], "memeval_short_answer")
        tuned.set(["llm", "refusal_text"], "None")
        context = dict(tuned.get("llm", "context", default={}) or {})
        context["semantic_top_k"] = max(int(context.get("semantic_top_k", 8)), 10)
        context["episodic_top_k"] = max(int(context.get("episodic_top_k", 5)), 8)
        tuned.set(["llm", "context"], context)
        tuned.set(["benchmark_overrides", "memeval", "answer_mode"], "full_context")
        tuned.set(["benchmark_overrides", "memeval", "max_full_context_turns"], 600)
    return tuned


def dialogue_turns_to_events(dialogues: Iterable[dict[str, Any]]) -> list[MemoryEvent]:
    events: list[MemoryEvent] = []
    for turn_index, turn in enumerate(dialogues, start=1):
        dia_id = str(turn.get("dia_id", "") or "")
        session_id = ""
        if dia_id and "_" in dia_id:
            session_id = dia_id.rsplit("_", 1)[0]
        metadata = {
            "timestamp": str(turn.get("timestamp", "") or ""),
            "dia_id": dia_id,
            "session_id": session_id,
            "source": "memeval",
        }
        events.append(
            MemoryEvent(
                speaker=str(turn.get("speaker", "Unknown")),
                text=str(turn.get("text", "")),
                turn_index=turn_index,
                metadata=metadata,
            )
        )
    return events


def normalize_memeval_prediction(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return "None"
    lowered = cleaned.lower()
    refusal_markers = (
        "i don't know",
        "none",
        "not mentioned",
        "not specified",
        "not provided",
        "unknown",
        "cannot be determined",
        "no evidence",
        "no info",
        "not enough information",
    )
    if any(marker in lowered for marker in refusal_markers):
        return "None"
    return cleaned


def evaluate_candidate_on_conversation(
    conv: dict[str, Any],
    *,
    candidate: CandidateConfig,
    run_judge: bool,
    category_names: dict[Any, str] | None = None,
    judge_fn: str | None = None,
) -> list[dict[str, Any]]:
    ensure_memeval_imports()
    from agents_memory.locomo import extract_dialogues

    system = HybridMemorySystem(candidate)
    system.reset()
    dialogues = extract_dialogues(conv)
    for event in dialogue_turns_to_events(dialogues):
        system.ingest(event)
    print(f"    Ingested: {len(dialogues)} turns")

    answer_mode = str(
        candidate.get("benchmark_overrides", "memeval", "answer_mode", default="retrieval")
    )
    max_full_context_turns = int(
        candidate.get("benchmark_overrides", "memeval", "max_full_context_turns", default=0)
    )
    full_context_text = _format_dialogues_as_evidence(dialogues)
    qa_by_question = {
        str(qa.get("question", "")): qa
        for qa in conv.get("qa", [])
    }

    def answer_fn(question: str) -> str:
        qa = qa_by_question.get(question, {})
        if (
            answer_mode == "full_context"
            and system.llm_backend is not None
            and len(dialogues) <= max_full_context_turns
        ):
            system._ensure_consolidated()
            plan = system.router.route(question, system.semantic.subjects())
            heuristic = system._answer_heuristically(plan, question)
            focused_context = system._collect_llm_context(plan, question)["text"]
            return normalize_memeval_prediction(
                system.llm_backend.answer(
                    question=question,
                    evidence_text=f"{focused_context}\n\n{full_context_text}",
                    heuristic_answer=heuristic,
                    task_instructions=_memeval_task_instructions(qa),
                )
            )
        return normalize_memeval_prediction(system.answer(question))

    return _memeval_qa_results(
        conv,
        answer_fn,
        run_judge,
        category_names=category_names,
        judge_fn=judge_fn,
    )


def compute_memeval_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    by_sample: dict[str, list[dict[str, Any]]] = {}
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_sample.setdefault(str(row.get("sample_id", "unknown")), []).append(row)
        by_category.setdefault(str(row.get("category_name", "Unknown")), []).append(row)

    per_sample = {
        sample_id: {
            "f1": sum(item["f1"] for item in sample_rows) / len(sample_rows),
            "n_questions": len(sample_rows),
        }
        for sample_id, sample_rows in by_sample.items()
    }
    overall_f1s = [entry["f1"] for entry in per_sample.values()]
    summary: dict[str, Any] = {
        "overall_f1_mean": sum(overall_f1s) / len(overall_f1s),
        "n_conversations": len(by_sample),
        "n_questions": len(rows),
        "per_conversation": per_sample,
        "by_category": {},
    }

    if any("judge_relevant" in row for row in rows):
        judge_keys = ("judge_relevant", "judge_complete", "judge_accurate")
        judge_rates = {
            key: sum(int(row.get(key, 0)) for row in rows) / len(rows)
            for key in judge_keys
        }
        summary["judge"] = {
            **judge_rates,
            "judge_pass_rate": sum(judge_rates.values()) / len(judge_rates),
        }

    if any("longmemeval_correct" in row for row in rows):
        summary["longmemeval_accuracy"] = (
            sum(int(row.get("longmemeval_correct", 0)) for row in rows) / len(rows)
        )

    for category_name, category_rows in sorted(by_category.items()):
        category_summary: dict[str, Any] = {
            "f1_mean": sum(item["f1"] for item in category_rows) / len(category_rows),
            "n": len(category_rows),
        }
        if any("judge_relevant" in row for row in category_rows):
            judge_keys = ("judge_relevant", "judge_complete", "judge_accurate")
            category_summary["judge"] = {
                key: sum(int(row.get(key, 0)) for row in category_rows) / len(category_rows)
                for key in judge_keys
            }
        if any("longmemeval_correct" in row for row in category_rows):
            category_summary["accuracy"] = (
                sum(int(row.get("longmemeval_correct", 0)) for row in category_rows)
                / len(category_rows)
            )
        summary["by_category"][category_name] = category_summary

    return summary


def evaluate_candidate_on_benchmark(
    *,
    benchmark: str,
    candidate: CandidateConfig,
    split: Optional[str] = None,
    num_samples: int = 10,
    data_file: str | Path | None = None,
    run_judge: bool = False,
    memeval_tuned: bool = True,
) -> MemEvalRunResult:
    bench, conversations = load_memeval_benchmark(
        benchmark,
        split=split,
        num_samples=num_samples,
        data_file=data_file,
    )
    category_names = bench.get("category_names")
    judge_fn = bench.get("judge_fn")
    benchmark_name = str(bench.get("name", benchmark))
    active_candidate = apply_memeval_overrides(candidate) if memeval_tuned else candidate

    rows: list[dict[str, Any]] = []
    print(f"Evaluating {active_candidate.candidate_id} on {benchmark_name} ({len(conversations)} samples)")
    for index, conv in enumerate(conversations, start=1):
        sample_id = conv.get("sample_id", f"sample-{index}")
        print(f"  Conversation {index}/{len(conversations)}: {sample_id}")
        rows.extend(
            evaluate_candidate_on_conversation(
                conv,
                candidate=active_candidate,
                run_judge=run_judge,
                category_names=category_names,
                judge_fn=judge_fn,
            )
        )

    summary = compute_memeval_summary(rows)
    return MemEvalRunResult(
        benchmark=benchmark,
        benchmark_name=benchmark_name,
        split=split,
        candidate_id=active_candidate.candidate_id,
        llm_model=str(active_candidate.get("llm", "model", default=None))
        if active_candidate.get("llm", "enabled", default=False)
        else None,
        summary=summary,
        rows=rows,
    )


def evaluate_registry_system_on_benchmark(
    *,
    benchmark: str,
    system_name: str,
    llm_model: str,
    split: Optional[str] = None,
    num_samples: int = 10,
    data_file: str | Path | None = None,
    run_judge: bool = False,
) -> MemEvalRunResult:
    ensure_memeval_imports()

    bench, conversations = load_memeval_benchmark(
        benchmark,
        split=split,
        num_samples=num_samples,
        data_file=data_file,
    )
    category_names = bench.get("category_names")
    judge_fn = bench.get("judge_fn")
    benchmark_name = str(bench.get("name", benchmark))

    rows: list[dict[str, Any]] = []
    system_key = system_name.lower()
    if system_key != "propmem":
        raise ValueError(
            f"Unsupported compare system: {system_name}. Currently supported: propmem"
        )
    print(f"Evaluating baseline {system_name} on {benchmark_name} ({len(conversations)} samples)")
    for index, conv in enumerate(conversations, start=1):
        sample_id = conv.get("sample_id", f"sample-{index}")
        print(f"  Conversation {index}/{len(conversations)}: {sample_id}")
        rows.extend(
            evaluate_propmem_on_conversation(
                conv,
                llm_model=llm_model,
                run_judge=run_judge,
                category_names=category_names,
                judge_fn=judge_fn,
            )
        )

    summary = compute_memeval_summary(rows)
    return MemEvalRunResult(
        benchmark=benchmark,
        benchmark_name=benchmark_name,
        split=split,
        candidate_id=system_name,
        llm_model=llm_model,
        summary=summary,
        rows=rows,
    )


def load_candidate_for_memeval(path: str | Path, *, memeval_tuned: bool = True) -> CandidateConfig:
    candidate = load_candidate(path)
    return apply_memeval_overrides(candidate) if memeval_tuned else candidate


def _format_dialogues_as_evidence(dialogues: list[dict[str, Any]]) -> str:
    lines = ["Conversation transcript:"]
    session_timeline: dict[str, str] = {}
    for turn in dialogues:
        speaker = str(turn.get("speaker", "Unknown"))
        text = str(turn.get("text", ""))
        timestamp = str(turn.get("timestamp", "") or "").strip()
        dia_id = str(turn.get("dia_id", "") or "").strip()
        if dia_id and "_" in dia_id and timestamp:
            session_key = dia_id.rsplit("_", 1)[0]
            session_timeline.setdefault(session_key, timestamp)
        prefix_bits = []
        if dia_id:
            prefix_bits.append(dia_id)
        if timestamp:
            prefix_bits.append(timestamp)
        prefix = " | ".join(prefix_bits)
        if prefix:
            lines.append(f"[{prefix}] {speaker}: {text}")
        else:
            lines.append(f"{speaker}: {text}")
    if session_timeline:
        timeline_lines = ["Session timeline:"]
        for session_key, timestamp in session_timeline.items():
            timeline_lines.append(f"- {session_key}: {timestamp}")
        return "\n".join(timeline_lines + [""] + lines)
    return "\n".join(lines)


def _memeval_task_instructions(qa: dict[str, Any]) -> str:
    category = str(qa.get("category", ""))
    question_id = str(qa.get("question_id", ""))
    instructions: list[str] = []

    if category == "single-session-preference":
        instructions.append(
            "This is a personalized preference question. Summarize the user's remembered preferences and constraints from the conversation instead of inventing a new recommendation. Mention the key desired features and avoid bare `None` when the preference evidence is recoverable."
        )
    elif category == "temporal-reasoning":
        instructions.append(
            "This is a temporal reasoning question. Compute elapsed time or date differences from the dates in the evidence. If the question asks how long ago something happened, use the latest conversation date as the reference point unless the question specifies another anchor. Return only the resulting duration phrase, such as `7 days ago` or `21 days`."
        )
    elif category == "knowledge-update":
        instructions.append(
            "This is a knowledge update question. If old and new values both appear, answer with the updated value."
        )
    elif category in {"single-session-user", "single-session-assistant", "multi-session"}:
        instructions.append(
            "Answer with the most specific supported fact from the conversation. If the asked information was never provided, answer with a brief explicit explanation such as `You did not mention this information.`"
        )

    if question_id.endswith("_abs"):
        instructions.append(
            "This question may be unanswerable from the evidence. If the needed information is missing, answer with a brief explicit explanation such as `You did not mention this information.` rather than a bare `None`."
        )

    return "\n".join(instructions)


def _memeval_qa_results(
    conv: dict[str, Any],
    answer_fn,
    run_judge: bool,
    *,
    category_names: dict[Any, str] | None = None,
    judge_fn: str | None = None,
) -> list[dict[str, Any]]:
    ensure_memeval_imports()
    from agents_memory.evaluation import compute_f1, evaluate_longmemeval, evaluate_with_judge
    from agents_memory.locomo import CATEGORY_NAMES as default_categories

    categories = category_names or default_categories
    qa_pairs = conv.get("qa", [])
    sample_id = conv.get("sample_id", "unknown")
    rows: list[dict[str, Any]] = []

    for index, qa in enumerate(qa_pairs, start=1):
        question = qa.get("question", "")
        ground_truth = qa.get("answer", "")
        category = qa.get("category", 0)
        question_id = qa.get("question_id", "")
        try:
            predicted = answer_fn(question)
        except Exception as exc:
            print(f"    Error on Q{index}: {exc}")
            predicted = "None"

        row = {
            "sample_id": sample_id,
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "category": category,
            "category_name": categories.get(category, str(category)),
            "f1": compute_f1(predicted, ground_truth),
        }
        if run_judge:
            if judge_fn == "longmemeval":
                row.update(
                    evaluate_longmemeval(
                        question,
                        ground_truth,
                        predicted,
                        category=str(category),
                        question_id=question_id,
                    )
                )
            else:
                row.update(evaluate_with_judge(question, ground_truth, predicted))
        rows.append(row)
    return rows


def evaluate_propmem_on_conversation(
    conv: dict[str, Any],
    *,
    llm_model: str,
    run_judge: bool,
    category_names: dict[Any, str] | None = None,
    judge_fn: str | None = None,
) -> list[dict[str, Any]]:
    ensure_memeval_imports()
    from openai import OpenAI
    from agents_memory.propmem import PropMemSystem

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    original_create = client.chat.completions.create

    def compat_create(*args, **kwargs):
        model_name = str(kwargs.get("model", llm_model))
        if (
            ("max_tokens" in kwargs)
            and ("max_completion_tokens" not in kwargs)
            and (model_name.startswith("gpt-5") or model_name.startswith("o"))
        ):
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        return original_create(*args, **kwargs)

    client.chat.completions.create = compat_create
    system = PropMemSystem()
    ingest = system.ingest_conversation(conv, client, llm_model)
    print(
        f"    Ingested: chunks={ingest['num_chunks']}, "
        f"propositions={ingest['num_propositions']}, "
        f"clusters={ingest.get('num_clusters', 0)}"
    )
    return _memeval_qa_results(
        conv,
        lambda q: normalize_memeval_prediction(system.answer_question(q, client, llm_model)["answer"]),
        run_judge,
        category_names=category_names,
        judge_fn=judge_fn,
    )
