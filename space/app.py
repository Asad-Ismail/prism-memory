from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
RELEASE_MODEL_NAME = "PRISM-Memory 7B Adapter"
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def _resolve_root() -> Path:
    for candidate in (APP_DIR, APP_DIR.parent):
        if (candidate / "results" / "release_summary.json").exists():
            return candidate
        if (candidate / "docs" / "release" / "extraction-skill.md").exists():
            return candidate
        if (candidate / "MEMORY_EXTRACTION_SKILL.md").exists():
            return candidate
    return APP_DIR.parent


ROOT = _resolve_root()
RESULTS_DIR = ROOT / "results"
SUMMARY_CANDIDATES = [RESULTS_DIR / "release_summary.json"]
SCENARIO_CANDIDATES = [RESULTS_DIR / "benchmark_cases.json"]
EXAMPLE_CANDIDATES = [RESULTS_DIR / "extraction_examples.json"]
SKILL_CANDIDATES = [
    ROOT / "docs" / "release" / "extraction-skill.md",
    ROOT / "MEMORY_EXTRACTION_SKILL.md",
]
USE_CASE_CANDIDATES = [
    ROOT / "docs" / "release" / "memory-scenarios.md",
]
DATASET_CANDIDATES = [
    ROOT / "docs" / "release" / "datasets.md",
    ROOT / "DATASETS.md",
]
LOCOMO_CATEGORY_NAMES = {
    "1": "factual",
    "2": "temporal",
    "3": "inferential",
    "4": "multi-hop",
    "5": "adversarial",
}
LME_CATEGORY_ORDER = [
    "knowledge-update",
    "multi-session",
    "single-session-assistant",
    "single-session-preference",
    "single-session-user",
    "temporal-reasoning",
]


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _load_json_from_candidates(candidates: list[Path], default):
    for path in candidates:
        if path.exists():
            return _load_json(path, default)
    return default


def _clean_markdown(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("[Back to Repo]"):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines).strip()


def _load_markdown(candidates: list[Path], fallback: str) -> str:
    for path in candidates:
        if path.exists():
            return _clean_markdown(path.read_text())
    return fallback


def _load_summary() -> dict:
    return _load_json_from_candidates(SUMMARY_CANDIDATES, {"results": [], "failures": []})


def _load_scenarios() -> dict:
    return _load_json_from_candidates(SCENARIO_CANDIDATES, {"scenarios": []})


def _load_examples() -> dict:
    return _load_json_from_candidates(EXAMPLE_CANDIDATES, {"examples": []})


def _load_skill() -> str:
    return _load_markdown(SKILL_CANDIDATES, "Skill document not found.")


def _load_use_cases() -> str:
    return _load_markdown(USE_CASE_CANDIDATES, "End-to-end scenario summary not found.")


def _load_datasets() -> str:
    return _load_markdown(DATASET_CANDIDATES, "Dataset summary not found.")


def _best_result() -> dict | None:
    results = _load_summary().get("results", [])
    return results[0] if results else None


def _model_name(item: dict) -> str:
    return item.get("model_name", RELEASE_MODEL_NAME)


def _base_model(item: dict) -> str:
    return item.get("base_model", BASE_MODEL_NAME)


def release_markdown() -> str:
    item = _best_result()
    if not item:
        return "## No confirmed release result yet"
    locomo = item["locomo"]["mean"]
    lme = item["lme"]["mean"]
    return "\n".join(
        [
            "# PRISM-Memory",
            "",
            "**Turn conversations into durable, searchable memory.**",
            "",
            f"Released model: `{_model_name(item)}`",
            f"Base model: `{_base_model(item)}`",
            "",
            "| Benchmark | PRISM-Memory | GPT-4.1-based PropMem reference |",
            "|---|---:|---:|",
            f"| LongMemEval | `{lme:.3f}` | `0.465` |",
            f"| LoCoMo | `{locomo:.3f}` | `0.536` |",
            "",
            "This Space shows one public release: confirmed metrics, benchmark cases, end-to-end scenarios, side-by-side extraction examples, the synthetic-data summary, and the canonical extraction skill.",
        ]
    )


def summary_df() -> pd.DataFrame:
    item = _best_result()
    if not item:
        return pd.DataFrame(columns=["model", "base_model", "locomo_mean", "lme_mean", "cache_hits", "cache_misses", "eval_minutes"])
    return pd.DataFrame(
        [
            {
                "model": _model_name(item),
                "base_model": _base_model(item),
                "locomo_mean": round(item["locomo"]["mean"], 3),
                "lme_mean": round(item["lme"]["mean"], 3),
                "cache_hits": item["qa_cache"]["hits"],
                "cache_misses": item["qa_cache"]["misses"],
                "eval_minutes": item["elapsed_min"],
            }
        ]
    )


def category_df() -> pd.DataFrame:
    item = _best_result()
    if not item:
        return pd.DataFrame(columns=["benchmark", "category", "score"])
    rows = []
    for category in sorted(item["locomo"]["categories"], key=int):
        score = item["locomo"]["categories"][category]
        rows.append(
            {
                "benchmark": "LoCoMo",
                "category": LOCOMO_CATEGORY_NAMES.get(category, category),
                "score": round(score, 3),
            }
        )
    for category in LME_CATEGORY_ORDER:
        if category not in item["lme"]["categories"]:
            continue
        score = item["lme"]["categories"][category]
        rows.append({"benchmark": "LongMemEval", "category": category, "score": round(score, 3)})
    return pd.DataFrame(rows)


def _scenario_label(item: dict) -> str:
    return f"{item['title']}: {item['question']}"


def scenario_choices() -> list[str]:
    scenarios = _load_scenarios().get("scenarios", [])
    return [_scenario_label(s) for s in scenarios]


def _pick_release_system(systems: list[dict]) -> dict:
    for system in systems:
        if system.get("name") == "release_model":
            return system
        if system.get("display_name") == RELEASE_MODEL_NAME:
            return system
    return systems[0]


def render_scenario(choice: str):
    scenarios = _load_scenarios().get("scenarios", [])
    if not scenarios:
        return "No scenario data yet.", pd.DataFrame(columns=["prediction", "top_retrieval"])

    item = next(
        (scenario for scenario in scenarios if _scenario_label(scenario) == choice or scenario["id"] == choice),
        scenarios[0],
    )
    system = _pick_release_system(item.get("systems", []))
    header = [
        f"### {item['title']}",
        "",
        f"**Question:** {item['question']}",
        f"**Gold answer:** {item['gold_answer']}",
        f"**What this case shows:** {item.get('note', 'Selected benchmark case.')}",
        f"**Case type:** {item.get('kind', 'n/a')}",
    ]
    table = pd.DataFrame(
        [
            {
                "prediction": system.get("prediction", ""),
                "top_retrieval": "\n".join(system.get("top_retrieval", [])[:3]),
            }
        ]
    )
    return "\n".join(header), table


def _example_label(item: dict) -> str:
    return item["title"]


def example_choices() -> list[str]:
    examples = _load_examples().get("examples", [])
    return [_example_label(example) for example in examples]


def render_example(choice: str) -> str:
    examples = _load_examples().get("examples", [])
    if not examples:
        return "No extraction examples available yet."

    item = next(
        (example for example in examples if _example_label(example) == choice or example["id"] == choice),
        examples[0],
    )
    body = [
        f"### {item['title']}",
        "",
        f"**Session date:** `{item['session_date']}`",
        f"**Overlap score:** `{item['overlap_score']:.3f}`",
        f"**What this example shows:** {item['note']}",
        "",
        "**Turn**",
        "",
        f"> {item['user_message']}",
        "",
        "**GPT-4.1 reference**",
    ]
    body.extend([f"- {entry}" for entry in item.get("gpt41_reference", [])])
    body.extend(["", "**PRISM-Memory**"])
    body.extend([f"- {entry}" for entry in item.get("prism_memory", [])])
    return "\n".join(body)


with gr.Blocks(title="PRISM-Memory Demo") as demo:
    gr.Markdown(release_markdown())

    with gr.Tab("Metrics"):
        gr.Markdown("## Released Model")
        metrics = gr.Dataframe(value=summary_df(), interactive=False, wrap=True)
        gr.Markdown("## Category Breakdown")
        categories = gr.Dataframe(value=category_df(), interactive=False, wrap=True)
        refresh = gr.Button("Refresh Data")
        refresh.click(fn=lambda: (summary_df(), category_df()), outputs=[metrics, categories])

    with gr.Tab("Benchmark Cases"):
        choices = scenario_choices() or ["pending"]
        picker = gr.Dropdown(choices=choices, value=choices[0], label="Benchmark Case")
        scenario_md = gr.Markdown()
        scenario_table = gr.Dataframe(interactive=False, wrap=True)
        picker.change(render_scenario, inputs=picker, outputs=[scenario_md, scenario_table])
        demo.load(fn=lambda: render_scenario(choices[0]), outputs=[scenario_md, scenario_table])

    with gr.Tab("Use Cases"):
        gr.Markdown(_load_use_cases())

    with gr.Tab("Extraction Examples"):
        choices = example_choices() or ["pending"]
        picker = gr.Dropdown(choices=choices, value=choices[0], label="Held-Out Example")
        example_md = gr.Markdown()
        picker.change(render_example, inputs=picker, outputs=example_md)
        demo.load(fn=lambda: render_example(choices[0]), outputs=example_md)

    with gr.Tab("Data"):
        gr.Markdown(_load_datasets())

    with gr.Tab("Skill"):
        gr.Markdown(_load_skill())


if __name__ == "__main__":
    demo.launch()
