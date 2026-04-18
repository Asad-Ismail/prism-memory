from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
RELEASE_MODEL_NAME = "PRISM-Memory 7B Adapter"
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REPO_ID = os.environ.get("PRISM_MODEL_REPO", "AsadIsmail/prism-memory")
SYSTEM_PROMPT = (
    "You are a memory extraction assistant. Given a conversation turn, "
    "extract 0-5 atomic, standalone facts. Each fact must be a complete "
    "sentence about a specific person, event, preference, or property. "
    "Include dates/times when mentioned. Skip greetings, filler, and questions. "
    'Output ONLY a JSON array of strings, e.g. ["fact1", "fact2"] or [].'
)
TURN_PATTERN = re.compile(
    r"^\s*(?:\[(?P<bracket_date>[^\]]+)\]\s*)?"
    r"(?:(?P<plain_date>\d{4}-\d{2}-\d{2})\s+)?"
    r"(?P<speaker>[^:]{1,40}):\s*(?P<text>.+?)\s*$"
)
FILLER_PREFIX = re.compile(
    r"^(yeah|yep|ok|okay|well|so|honestly|basically|actually|good point|i think|we should probably)\b[:, -]*",
    re.IGNORECASE,
)
FIRST_PERSON_PATTERNS = [
    (re.compile(r"\bI'm\b", re.IGNORECASE), "{speaker} is"),
    (re.compile(r"\bI am\b", re.IGNORECASE), "{speaker} is"),
    (re.compile(r"\bI have\b", re.IGNORECASE), "{speaker} has"),
    (re.compile(r"\bI want\b", re.IGNORECASE), "{speaker} wants"),
    (re.compile(r"\bI need\b", re.IGNORECASE), "{speaker} needs"),
    (re.compile(r"\bI started\b", re.IGNORECASE), "{speaker} started"),
    (re.compile(r"\bI bought\b", re.IGNORECASE), "{speaker} bought"),
    (re.compile(r"\bI signed up\b", re.IGNORECASE), "{speaker} signed up"),
    (re.compile(r"\bmy\b", re.IGNORECASE), "{speaker}'s"),
    (re.compile(r"\bme\b", re.IGNORECASE), "{speaker}"),
    (re.compile(r"\bI\b", re.IGNORECASE), "{speaker}"),
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
EXAMPLE_CANDIDATES = [RESULTS_DIR / "extraction_examples.json"]
TRY_IT_CANDIDATES = [RESULTS_DIR / "try_it_sessions.json"]
SKILL_CANDIDATES = [
    ROOT / "docs" / "release" / "extraction-skill.md",
    ROOT / "MEMORY_EXTRACTION_SKILL.md",
]
DATASET_CANDIDATES = [
    ROOT / "docs" / "release" / "datasets.md",
    ROOT / "DATASETS.md",
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


def _load_examples() -> dict:
    return _load_json_from_candidates(EXAMPLE_CANDIDATES, {"examples": []})


def _load_try_it_examples() -> dict:
    return _load_json_from_candidates(TRY_IT_CANDIDATES, {"examples": []})


def _load_skill() -> str:
    return _load_markdown(SKILL_CANDIDATES, "Skill document not found.")


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
            "This Space shows the public release in a product-shaped way: one model, an interactive try-it flow, held-out extraction examples, the synthetic-data summary, and the canonical extraction skill.",
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
        rows.append(
            {
                "benchmark": "LoCoMo",
                "category": LOCOMO_CATEGORY_NAMES.get(category, category),
                "score": round(item["locomo"]["categories"][category], 3),
            }
        )
    for category in LME_CATEGORY_ORDER:
        if category in item["lme"]["categories"]:
            rows.append(
                {
                    "benchmark": "LongMemEval",
                    "category": category,
                    "score": round(item["lme"]["categories"][category], 3),
                }
            )
    return pd.DataFrame(rows)


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


def _session_label(item: dict) -> str:
    return item["title"]


def try_it_choices() -> list[str]:
    sessions = _load_try_it_examples().get("examples", [])
    return [_session_label(item) for item in sessions]


def _get_session(choice: str | None) -> dict | None:
    sessions = _load_try_it_examples().get("examples", [])
    if not sessions:
        return None
    if not choice:
        return sessions[0]
    return next(
        (item for item in sessions if _session_label(item) == choice or item["id"] == choice),
        sessions[0],
    )


def load_try_it_session(choice: str):
    item = _get_session(choice)
    if not item:
        return "", "No bundled example sessions available."
    intro = "\n".join(
        [
            f"### {item['title']}",
            "",
            f"**What this session shows:** {item['note']}",
            "",
            "**Later question**",
            "",
            item["later_question"],
        ]
    )
    return item["transcript"], intro


def load_and_run_session(choice: str):
    transcript, intro = load_try_it_session(choice)
    status, table, memory_md, qa_md = run_try_it(choice, transcript)
    return transcript, intro, status, table, memory_md, qa_md


def _parse_transcript(transcript: str) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for raw_line in transcript.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = TURN_PATTERN.match(line)
        if match:
            turns.append(
                {
                    "date": (match.group("bracket_date") or match.group("plain_date") or "").strip(),
                    "speaker": match.group("speaker").strip(),
                    "text": match.group("text").strip(),
                }
            )
            continue
        if turns:
            turns[-1]["text"] = f"{turns[-1]['text']} {line}".strip()
    return turns


def _normalize_first_person(text: str, speaker: str) -> str:
    value = text
    for pattern, replacement in FIRST_PERSON_PATTERNS:
        value = pattern.sub(replacement.format(speaker=speaker), value)
    return value


def _clean_clause(text: str) -> str:
    text = FILLER_PREFIX.sub("", text).strip()
    text = text.strip(" -\t")
    return re.sub(r"\s+", " ", text)


def _preview_extract_turn(turn: dict[str, str], _: str) -> list[str]:
    pieces = re.split(r"[.;]\s+|\n+", turn["text"])
    facts: list[str] = []
    for piece in pieces:
        clause = _clean_clause(piece)
        if not clause or clause.endswith("?") or len(clause) < 8:
            continue
        clause = _normalize_first_person(clause, turn["speaker"])
        if clause.lower().startswith(("hi ", "hello ", "thanks ", "thank you ")):
            continue
        clause = clause[0].upper() + clause[1:]
        if not clause.endswith((".", "!", "?")):
            clause += "."
        facts.append(clause)
        if len(facts) == 5:
            break
    return facts


def _parse_props(raw: str) -> list[str]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    match = re.search(r"\[([^\]]*)\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads("[" + match.group(1) + "]")
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass
    return _preview_extract_turn({"speaker": "Speaker", "text": raw}, "")


@lru_cache(maxsize=1)
def _load_live_stack():
    if os.environ.get("PRISM_ENABLE_LIVE_MODEL", "").lower() not in {"1", "true", "yes"}:
        raise RuntimeError("Live model loading is disabled for this Space runtime.")

    try:
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - dependency gate
        raise RuntimeError("Live model dependencies are not installed in this runtime.") from exc

    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID, trust_remote_code=True)
    tokenizer.padding_side = "left"
    device_map = "auto"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoPeftModelForCausalLM.from_pretrained(
        MODEL_REPO_ID,
        trust_remote_code=True,
        device_map=device_map,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
    )
    model.eval()
    return torch, tokenizer, model


def _live_extract_turn(turn: dict[str, str], context: str) -> list[str]:
    torch_mod, tokenizer, model = _load_live_stack()
    user_prompt = "\n".join(
        [
            f"Date: {turn['date']}" if turn.get("date") else "",
            f"Recent context: ...{context[-300:]}" if context else "",
            f"Speaker ({turn['speaker']}): {turn['text']}",
        ]
    ).strip()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=768,
    ).to(model.device)
    with torch_mod.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    raw = tokenizer.decode(output[0][encoded.input_ids.shape[1] :], skip_special_tokens=True).strip()
    return _parse_props(raw)[:5]


def _dedupe_items(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        key = value.casefold().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(value.strip())
    return output


def _memory_markdown(items: list[str]) -> str:
    if not items:
        return "No memory records written yet."
    return "\n".join(["### Accumulated Memory", ""] + [f"- {item}" for item in items])


def _question_markdown(item: dict | None) -> str:
    if not item:
        return ""
    return "\n".join(
        [
            "### Later Question",
            "",
            f"**Question:** {item['later_question']}",
            f"**Answer from memory:** {item['answer_from_memory']}",
        ]
    )


def run_try_it(choice: str, transcript: str):
    session = _get_session(choice)
    normalized_transcript = transcript.strip()
    if not normalized_transcript:
        return "Paste a transcript or load one of the bundled sessions.", pd.DataFrame(), "No memory records yet.", ""

    if session and normalized_transcript == session["transcript"].strip():
        rows = []
        final_memory: list[str] = []
        for turn in session["turns"]:
            extracted = turn.get("prism_memory", [])
            rows.append(
                {
                    "turn": turn["turn_index"],
                    "date": turn["date"],
                    "speaker": turn["speaker"],
                    "memory_records": "\n".join(extracted),
                }
            )
            final_memory.extend(
                [f"[{turn['date']}] {record}" if turn.get("date") else record for record in extracted]
            )
        status = "\n".join(
            [
                "### Try It",
                "",
                "**Mode:** released model output (bundled example)",
                "",
                "These per-turn memory records were precomputed with the released PRISM-Memory adapter. The model is trained to write memory turn by turn, then let retrieval use the accumulated store later.",
            ]
        )
        return status, pd.DataFrame(rows), _memory_markdown(_dedupe_items(final_memory)), _question_markdown(session)

    turns = _parse_transcript(normalized_transcript)
    if not turns:
        return (
            "Could not parse the transcript. Use one turn per line, for example `[2025-03-01] Dana: We have 20 concurrent jobs max.`",
            pd.DataFrame(),
            "No memory records yet.",
            "",
        )

    extractor = _preview_extract_turn
    mode = "preview extractor"
    note = (
        "This runtime is using a lightweight turn-by-turn preview that follows the same extraction contract. "
        "The bundled example sessions above use actual released-model outputs."
    )
    try:
        _load_live_stack()
        extractor = _live_extract_turn
        mode = "released model (live)"
        note = "This runtime successfully loaded the released adapter and is extracting memory turn by turn."
    except Exception:
        pass

    rows = []
    final_memory: list[str] = []
    context_lines: list[str] = []
    for index, turn in enumerate(turns, start=1):
        context = "\n".join(context_lines[-6:])
        extracted = extractor(turn, context)[:5]
        rows.append(
            {
                "turn": index,
                "date": turn.get("date", ""),
                "speaker": turn["speaker"],
                "memory_records": "\n".join(extracted),
            }
        )
        final_memory.extend(
            [f"[{turn['date']}] {record}" if turn.get("date") else record for record in extracted]
        )
        context_lines.append(f"[{turn.get('date', '')}] {turn['speaker']}: {turn['text']}")

    status = "\n".join(
        [
            "### Try It",
            "",
            f"**Mode:** {mode}",
            "",
            note,
            "",
            "Expected transcript format: one turn per line as `[YYYY-MM-DD] Speaker: message` or `Speaker: message`.",
        ]
    )
    return status, pd.DataFrame(rows), _memory_markdown(_dedupe_items(final_memory)), ""


INITIAL_TRY_IT_CHOICES = try_it_choices()
INITIAL_TRY_IT_CHOICE = INITIAL_TRY_IT_CHOICES[0] if INITIAL_TRY_IT_CHOICES else ""
INITIAL_TRY_IT_TRANSCRIPT, INITIAL_TRY_IT_INTRO = load_try_it_session(INITIAL_TRY_IT_CHOICE) if INITIAL_TRY_IT_CHOICE else ("", "No bundled example sessions available.")
INITIAL_TRY_IT_STATUS, INITIAL_TRY_IT_DF, INITIAL_TRY_IT_MEMORY, INITIAL_TRY_IT_QA = (
    run_try_it(INITIAL_TRY_IT_CHOICE, INITIAL_TRY_IT_TRANSCRIPT)
    if INITIAL_TRY_IT_CHOICE
    else ("No bundled example sessions available.", pd.DataFrame(), "No memory records yet.", "")
)
INITIAL_EXAMPLE_CHOICES = example_choices()
INITIAL_EXAMPLE_CHOICE = INITIAL_EXAMPLE_CHOICES[0] if INITIAL_EXAMPLE_CHOICES else "pending"
INITIAL_EXAMPLE_MD = render_example(INITIAL_EXAMPLE_CHOICE) if INITIAL_EXAMPLE_CHOICES else "No extraction examples available yet."


with gr.Blocks(title="PRISM-Memory Demo") as demo:
    gr.Markdown(release_markdown())

    with gr.Tab("Metrics"):
        gr.Markdown("## Released Model")
        metrics = gr.Dataframe(value=summary_df(), interactive=False, wrap=True)
        gr.Markdown("## Category Breakdown")
        categories = gr.Dataframe(value=category_df(), interactive=False, wrap=True)
        refresh = gr.Button("Refresh Data")
        refresh.click(fn=lambda: (summary_df(), category_df()), outputs=[metrics, categories])

    with gr.Tab("Try It"):
        gr.Markdown(
            "\n".join(
                [
                    "Use one of the bundled sessions or paste your own transcript.",
                    "",
                    "PRISM-Memory is trained to write memory **turn by turn**, not to summarize a whole session in one shot.",
                ]
            )
        )
        choices = INITIAL_TRY_IT_CHOICES or ["No bundled sessions"]
        session_picker = gr.Dropdown(choices=choices, value=choices[0], label="Example Session")
        session_intro = gr.Markdown(value=INITIAL_TRY_IT_INTRO)
        transcript_box = gr.Textbox(
            label="Transcript",
            lines=10,
            value=INITIAL_TRY_IT_TRANSCRIPT,
            placeholder="[2025-03-01] Dana: We have 20 concurrent jobs max on GitHub Actions right now.",
        )
        run_button = gr.Button("Extract Memory")
        try_it_status = gr.Markdown(value=INITIAL_TRY_IT_STATUS)
        per_turn_df = gr.Dataframe(value=INITIAL_TRY_IT_DF, interactive=False, wrap=True, label="Per-Turn Memory")
        final_memory_md = gr.Markdown(value=INITIAL_TRY_IT_MEMORY)
        later_question_md = gr.Markdown(value=INITIAL_TRY_IT_QA)

        session_picker.change(
            load_and_run_session,
            inputs=session_picker,
            outputs=[transcript_box, session_intro, try_it_status, per_turn_df, final_memory_md, later_question_md],
        )
        run_button.click(
            run_try_it,
            inputs=[session_picker, transcript_box],
            outputs=[try_it_status, per_turn_df, final_memory_md, later_question_md],
        )

    with gr.Tab("Extraction Examples"):
        choices = INITIAL_EXAMPLE_CHOICES or ["pending"]
        picker = gr.Dropdown(choices=choices, value=choices[0], label="Held-Out Example")
        example_md = gr.Markdown(value=INITIAL_EXAMPLE_MD)
        picker.change(render_example, inputs=picker, outputs=example_md)

    with gr.Tab("Data"):
        gr.Markdown(_load_datasets())

    with gr.Tab("Skill"):
        gr.Markdown(_load_skill())


if __name__ == "__main__":
    demo.launch()
