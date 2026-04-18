#!/usr/bin/env python3
"""Generate selected PRISM-vs-GPT-4.1 extraction examples for the release docs."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BETTER_MEMORY_ROOT = Path(
    os.environ.get("BETTER_MEMORY_ROOT", "/home/ec2-user/SageMaker/better_memory")
)
DEFAULT_MODEL_PATH = next(
    (
        DEFAULT_BETTER_MEMORY_ROOT / candidate
        for candidate in ("prism_memory_release", "exp15_sft_qwen7b_4ep")
        if (DEFAULT_BETTER_MEMORY_ROOT / candidate).exists()
    ),
    DEFAULT_BETTER_MEMORY_ROOT / "exp15_sft_qwen7b_4ep",
)
DEFAULT_DATA_PATH = DEFAULT_BETTER_MEMORY_ROOT / "data" / "output" / "eval_sft.jsonl"
DEFAULT_SHORTLIST = Path(__file__).with_name("extraction_example_shortlist.json")
DEFAULT_OUTPUT_JSON = REPO_ROOT / "results" / "extraction_examples.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "docs" / "release" / "extraction-examples.md"
MODEL_NAME = "PRISM-Memory 7B Adapter"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = (
    "You are a memory extraction assistant. Given a conversation turn, "
    "extract 0-5 atomic, standalone facts. Each fact must be a complete "
    "sentence about a specific person, event, preference, or property. "
    "Include dates/times when mentioned. Skip greetings, filler, and questions. "
    'Output ONLY a JSON array of strings, e.g. ["fact1", "fact2"] or [].'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--better-memory-root",
        type=Path,
        default=DEFAULT_BETTER_MEMORY_ROOT,
        help="Path to the original better_memory workspace.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the release checkpoint directory.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to eval_sft.jsonl from the original better_memory workspace.",
    )
    parser.add_argument(
        "--shortlist",
        type=Path,
        default=DEFAULT_SHORTLIST,
        help="JSON shortlist describing which held-out examples to surface.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Where to write the structured example artifact.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help="Where to write the release markdown page.",
    )
    return parser.parse_args()


def _load_generation_stack():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as exc:  # pragma: no cover - dependency gate
        raise RuntimeError(
            "This script requires torch, transformers, and bitsandbytes. "
            "Use the existing pytorch_p310 environment or an equivalent runtime."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _make_user_prompt(speaker: str, text: str, date: str = "", context: str = "") -> str:
    parts: list[str] = []
    if date:
        parts.append(f"Date: {date}")
    if context and context not in ("[Start of conversation]", ""):
        parts.append(f"Recent context: ...{context[-200:]}")
    parts.append(f"Speaker ({speaker}): {text[:400]}")
    return "\n".join(parts)


def _format_as_chat(speaker: str, text: str, date: str = "", context: str = "") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _make_user_prompt(speaker, text, date, context)},
    ]


def _parse_props(raw: str) -> list[str]:
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            return [str(item).strip() for item in arr if str(item).strip()]
    except Exception:
        pass

    match = re.search(r"\[([^\]]*)\]", raw, re.DOTALL)
    if match:
        try:
            arr = json.loads("[" + match.group(1) + "]")
            if isinstance(arr, list):
                return [str(item).strip() for item in arr if str(item).strip()]
        except Exception:
            pass

    props: list[str] = []
    for line in raw.splitlines():
        line = re.sub(r'^[-•*\d.)\s"\']+', "", line).strip().rstrip('",\'')
        if line and len(line) > 10 and not line.endswith("?"):
            props.append(line)
    return props


def _normalize_tokens(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return [token for token in cleaned.split() if token]


def _token_f1(left: str, right: str) -> float:
    left_tokens = _normalize_tokens(left)
    right_tokens = _normalize_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0

    left_counts: dict[str, int] = {}
    right_counts: dict[str, int] = {}
    for token in left_tokens:
        left_counts[token] = left_counts.get(token, 0) + 1
    for token in right_tokens:
        right_counts[token] = right_counts.get(token, 0) + 1

    overlap = 0
    for token, count in left_counts.items():
        overlap += min(count, right_counts.get(token, 0))
    if overlap == 0:
        return 0.0

    precision = overlap / len(right_tokens)
    recall = overlap / len(left_tokens)
    return 2 * precision * recall / (precision + recall)


def _set_overlap(reference: list[str], predicted: list[str]) -> float:
    if not reference and not predicted:
        return 1.0
    if not reference or not predicted:
        return 0.0

    ref_score = sum(max(_token_f1(ref, pred) for pred in predicted) for ref in reference) / len(reference)
    pred_score = sum(max(_token_f1(pred, ref) for ref in reference) for pred in predicted) / len(predicted)
    return 2 * ref_score * pred_score / (ref_score + pred_score) if (ref_score + pred_score) else 0.0


def _load_shortlist(shortlist_path: Path) -> list[dict[str, Any]]:
    with shortlist_path.open() as handle:
        shortlist = json.load(handle)
    if not isinstance(shortlist, list):
        raise ValueError(f"Expected a list in {shortlist_path}")
    return shortlist


def _load_eval_examples(data_path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with data_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for item in record.get("sft_examples", []):
                target = [
                    tool_call.get("content")
                    for tool_call in item.get("output", {}).get("tool_calls", [])
                    if tool_call.get("content")
                ]
                if not target:
                    continue
                examples.append(
                    {
                        "session_date": item.get("session_date", ""),
                        "speaker": item.get("speaker", "User"),
                        "user_message": item.get("input", {}).get("user_message", ""),
                        "recent_conversation": item.get("input", {}).get("recent_conversation", ""),
                        "reference": target,
                    }
                )
    return examples


def _match_example(examples: list[dict[str, Any]], substring: str) -> dict[str, Any]:
    lowered = substring.lower()
    for example in examples:
        if lowered in example["user_message"].lower():
            return example
    raise ValueError(f"Could not find held-out example containing substring: {substring!r}")


def _load_model(model_path: Path):
    torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = _load_generation_stack()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return torch, tokenizer, model


def _generate_prediction(torch_mod, tokenizer, model, example: dict[str, Any]) -> list[str]:
    messages = _format_as_chat(
        example["speaker"],
        example["user_message"],
        example["session_date"],
        example["recent_conversation"],
    )
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
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
    return _parse_props(raw)


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "[Back to Repo](../../README.md) · [Release Docs](README.md) · [Result Artifacts](../../results/README.md)",
        "",
        "# PRISM-Memory Extraction Examples",
        "",
        "Selected held-out examples from the synthetic evaluation split.",
        "The `GPT-4.1 reference` rows come from the supervised target memory labels.",
        f"The `{MODEL_NAME}` rows were regenerated with greedy decoding using the same extraction prompt family used during evaluation.",
        "",
        "These examples are illustrations, not the benchmark itself. Use",
        "[release-results.md](release-results.md) for the aggregate numbers.",
        "",
    ]

    for item in payload["examples"]:
        lines.extend(
            [
                f"## {item['title']}",
                "",
                f"- Overlap score: `{item['overlap_score']:.3f}`",
                f"- Session date: `{item['session_date']}`",
                f"- Note: {item['note']}",
                "",
                "**Turn**",
                "",
                f"> {item['user_message']}",
                "",
                "**GPT-4.1 reference**",
                "",
            ]
        )
        lines.extend([f"- {entry}" for entry in item["gpt41_reference"]])
        lines.extend(["", "**PRISM-Memory**", ""])
        lines.extend([f"- {entry}" for entry in item["prism_memory"]])
        lines.append("")

    lines.extend(
        [
            "## Regeneration",
            "",
            "```bash",
            "conda run -n pytorch_p310 python scripts/release/generate_extraction_examples.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    shortlist = _load_shortlist(args.shortlist)
    eval_examples = _load_eval_examples(args.data_path)
    torch_mod, tokenizer, model = _load_model(args.model_path)

    output_examples: list[dict[str, Any]] = []
    for item in shortlist:
        matched = _match_example(eval_examples, item["match_substring"])
        predicted = _generate_prediction(torch_mod, tokenizer, model, matched)
        output_examples.append(
            {
                "id": item["id"],
                "title": item["title"],
                "note": item["note"],
                "session_date": matched["session_date"],
                "user_message": matched["user_message"],
                "gpt41_reference": matched["reference"],
                "prism_memory": predicted,
                "overlap_score": _set_overlap(matched["reference"], predicted),
            }
        )

    payload = {
        "dataset_name": "Held-out synthetic evaluation split",
        "model_name": MODEL_NAME,
        "base_model": BASE_MODEL,
        "output_examples": len(output_examples),
        "examples": output_examples,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")

    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(_render_markdown(payload) + "\n")

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_markdown}")


if __name__ == "__main__":
    main()
