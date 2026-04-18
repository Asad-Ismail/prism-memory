#!/usr/bin/env python3
"""Generate bundled example sessions for the interactive PRISM-Memory Space tab."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BETTER_MEMORY_ROOT = Path(
    os.environ.get("BETTER_MEMORY_ROOT", REPO_ROOT.parent / "better_memory")
)
DEFAULT_MODEL_PATH = Path(os.environ.get("PRISM_CHECKPOINT_DIR", ""))
DEFAULT_SPECS = Path(__file__).with_name("try_it_session_specs.json")
DEFAULT_OUTPUT = REPO_ROOT / "results" / "try_it_sessions.json"
MODEL_NAME = "PRISM-Memory 7B Adapter"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = (
    "You are a memory extraction assistant. Given a conversation turn, "
    "extract 0-5 atomic, standalone facts. Each fact must be a complete "
    "sentence about a specific person, event, preference, or property. "
    "Include dates/times when mentioned. Skip greetings, filler, and questions. "
    'Output ONLY a JSON array of strings, e.g. ["fact1", "fact2"] or [].'
)

REQUIRED_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
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
        default=DEFAULT_MODEL_PATH if str(DEFAULT_MODEL_PATH) else None,
        help="Path to the release checkpoint directory.",
    )
    parser.add_argument(
        "--specs",
        type=Path,
        default=DEFAULT_SPECS,
        help="Path to the session spec JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the bundled example sessions JSON.",
    )
    return parser.parse_args()


def _is_checkpoint_dir(path: Path | None) -> bool:
    if path is None or not path.is_dir():
        return False
    return all((path / relpath).exists() for relpath in REQUIRED_FILES)


def _discover_model_path(better_memory_root: Path) -> Path:
    preferred = [
        better_memory_root / "prism_memory_release",
        better_memory_root / "release_model",
    ]
    for candidate in preferred:
        if _is_checkpoint_dir(candidate):
            return candidate

    found = sorted(better_memory_root.glob("**/adapter_model.safetensors"))
    for adapter_path in found:
        candidate = adapter_path.parent
        if _is_checkpoint_dir(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find a PRISM checkpoint directory. Pass --model-path or set PRISM_CHECKPOINT_DIR."
    )


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
    if context:
        parts.append(f"Recent context: ...{context[-300:]}")
    parts.append(f"Speaker ({speaker}): {text[:500]}")
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
        if line and len(line) > 6 and not line.endswith("?"):
            props.append(line)
    return props[:5]


def _load_model(model_path: Path):
    torch_mod, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = _load_generation_stack()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_mod.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return torch_mod, tokenizer, model


def _generate_prediction(torch_mod, tokenizer, model, turn: dict[str, str], context: str) -> list[str]:
    messages = _format_as_chat(turn["speaker"], turn["text"], turn["date"], context)
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
    return _parse_props(raw)


def _dedupe_memories(memories: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for memory in memories:
        key = memory.casefold().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(memory.strip())
    return output


def _format_transcript(turns: list[dict[str, str]]) -> str:
    return "\n".join(f"[{turn['date']}] {turn['speaker']}: {turn['text']}" for turn in turns)


def _load_specs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")
    return payload


def main() -> None:
    args = parse_args()
    model_path = args.model_path
    if not _is_checkpoint_dir(model_path):
        model_path = _discover_model_path(args.better_memory_root)

    specs = _load_specs(args.specs)
    torch_mod, tokenizer, model = _load_model(model_path)

    rendered_examples: list[dict[str, Any]] = []
    for spec in specs:
        context_lines: list[str] = []
        running_memory: list[str] = []
        rendered_turns: list[dict[str, Any]] = []
        for index, turn in enumerate(spec["turns"], start=1):
            context = "\n".join(context_lines[-6:])
            predicted = _generate_prediction(torch_mod, tokenizer, model, turn, context)
            running_memory.extend(predicted)
            rendered_turns.append(
                {
                    "turn_index": index,
                    "date": turn["date"],
                    "speaker": turn["speaker"],
                    "text": turn["text"],
                    "prism_memory": predicted,
                    "memory_store_after_turn": _dedupe_memories(running_memory),
                }
            )
            context_lines.append(f"[{turn['date']}] {turn['speaker']}: {turn['text']}")

        rendered_examples.append(
            {
                "id": spec["id"],
                "title": spec["title"],
                "note": spec["note"],
                "transcript": _format_transcript(spec["turns"]),
                "later_question": spec["later_question"],
                "answer_from_memory": spec["answer_from_memory"],
                "source_mode": "released_model_precomputed",
                "turns": rendered_turns,
                "final_memory": _dedupe_memories(running_memory),
            }
        )

    payload = {
        "model_name": MODEL_NAME,
        "base_model": BASE_MODEL,
        "examples": rendered_examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
