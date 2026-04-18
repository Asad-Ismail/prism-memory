#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BETTER_MEMORY_ROOT="${BETTER_MEMORY_ROOT:-$ROOT_DIR/../better_memory}"

required_files=(
  adapter_config.json
  adapter_model.safetensors
  chat_template.jinja
  tokenizer.json
  tokenizer_config.json
  training_args.bin
)

is_checkpoint_dir() {
  local candidate="$1"
  [[ -n "$candidate" && -d "$candidate" ]] || return 1
  for relpath in "${required_files[@]}"; do
    [[ -f "$candidate/$relpath" ]] || return 1
  done
}

find_checkpoint_dir() {
  local candidate

  if [[ -n "${PRISM_CHECKPOINT_DIR:-}" ]] && is_checkpoint_dir "$PRISM_CHECKPOINT_DIR"; then
    printf '%s\n' "$PRISM_CHECKPOINT_DIR"
    return 0
  fi

  for candidate in \
    "$BETTER_MEMORY_ROOT/prism_memory_release" \
    "$BETTER_MEMORY_ROOT/release_model"
  do
    if is_checkpoint_dir "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  while IFS= read -r candidate; do
    candidate="${candidate%/adapter_model.safetensors}"
    if is_checkpoint_dir "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(find "$BETTER_MEMORY_ROOT" -maxdepth 3 -type f -name adapter_model.safetensors | sort)

  return 1
}

CHECKPOINT_DIR="$(find_checkpoint_dir || true)"
if [[ -z "$CHECKPOINT_DIR" ]]; then
  echo "Could not locate a PRISM checkpoint directory. Set PRISM_CHECKPOINT_DIR to the adapter folder." >&2
  exit 1
fi

BUNDLE_DIR="$ROOT_DIR/dist/model_bundle"
MODEL_REPO="${1:-}"
SPACE_REPO="${PRISM_SPACE_REPO:-AsadIsmail/prism-memory}"

rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR/docs/release" "$BUNDLE_DIR/results"

cp "$CHECKPOINT_DIR/adapter_config.json" "$BUNDLE_DIR/adapter_config.json"
cp "$CHECKPOINT_DIR/adapter_model.safetensors" "$BUNDLE_DIR/adapter_model.safetensors"
cp "$CHECKPOINT_DIR/chat_template.jinja" "$BUNDLE_DIR/chat_template.jinja"
cp "$CHECKPOINT_DIR/tokenizer.json" "$BUNDLE_DIR/tokenizer.json"
cp "$CHECKPOINT_DIR/tokenizer_config.json" "$BUNDLE_DIR/tokenizer_config.json"
cp "$CHECKPOINT_DIR/training_args.bin" "$BUNDLE_DIR/training_args.bin"
cp "$ROOT_DIR/LICENSE" "$BUNDLE_DIR/LICENSE"
cp "$ROOT_DIR/results/release_summary.json" "$BUNDLE_DIR/results/release_summary.json"
cp "$ROOT_DIR/results/extraction_examples.json" "$BUNDLE_DIR/results/extraction_examples.json"
cp "$ROOT_DIR/results/benchmark_cases.json" "$BUNDLE_DIR/results/benchmark_cases.json"

python - "$ROOT_DIR" "$BUNDLE_DIR" "$SPACE_REPO" <<'PY'
import json
import sys
from pathlib import Path


def strip_repo_nav(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("[Back to Repo]"):
        lines = lines[1:]
        if lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines).strip() + "\n"


def quote_block(text: str) -> str:
    return "\n".join(f"> {line}" for line in text.splitlines()) or ">"


root_dir = Path(sys.argv[1])
bundle_dir = Path(sys.argv[2])
space_repo = sys.argv[3]

doc_paths = [
    "docs/release/datasets.md",
    "docs/release/extraction-examples.md",
    "docs/release/extraction-skill.md",
    "docs/release/memory-scenarios.md",
    "docs/release/release-results.md",
    "docs/release/technical-blog.md",
]

for relpath in doc_paths:
    src = root_dir / relpath
    dst = bundle_dir / relpath
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(strip_repo_nav(src.read_text()), encoding="utf-8")

summary = json.loads((root_dir / "results/release_summary.json").read_text())["results"][0]
examples = json.loads((root_dir / "results/extraction_examples.json").read_text())["examples"]

locomo = summary["locomo"]["mean"]
lme = summary["lme"]["mean"]
qa_hits = summary["qa_cache"]["hits"]
qa_misses = summary["qa_cache"]["misses"]

example_sections = []
for example in examples[:2]:
    gpt_bullets = "\n".join(f"- {item}" for item in example["gpt41_reference"])
    prism_bullets = "\n".join(f"- {item}" for item in example["prism_memory"])
    example_sections.append(
        "\n".join(
            [
                f"### {example['title']}",
                f"- Session date: `{example['session_date']}`",
                f"- Overlap score: `{example['overlap_score']:.3f}`",
                f"- Note: {example['note']}",
                "",
                "**Turn**",
                "",
                quote_block(example["user_message"]),
                "",
                "**GPT-4.1 reference**",
                "",
                gpt_bullets,
                "",
                "**PRISM-Memory**",
                "",
                prism_bullets,
            ]
        )
    )

readme = f"""---
base_model: Qwen/Qwen2.5-7B-Instruct
base_model_relation: adapter
license: apache-2.0
library_name: peft
pipeline_tag: text-generation
tags:
- conversational-memory
- information-extraction
- long-context
- peft
- lora
- qwen2.5
---

# PRISM-Memory

PRISM-Memory is a LoRA adapter that trains `Qwen/Qwen2.5-7B-Instruct` to write
proposition-level memory from dialogue. It is a memory-writing component, not a
general chat model.

## Released model

- Model name: `PRISM-Memory 7B Adapter`
- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Adapter type: `LoRA`

## What this release shows

- A 7B open model can replace GPT-4.1 for the extraction step in this memory pipeline.
- On the confirmed release surface, PRISM-Memory scores `{lme:.4f}` on LongMemEval and `{locomo:.4f}` on LoCoMo.
- The GPT-4.1-based PropMem reference scores `0.4650` on LongMemEval and `0.5360` on LoCoMo.

This comparison holds the QA layer constant. It compares extractor against
extractor, not a full end-to-end GPT-4.1 system.

## Why this is useful

- It keeps hard limits and preferences available for later workflow generation.
- It keeps current state separate from future plans.
- It supports dated recall and clean refusal on unsupported questions.

See [docs/release/memory-scenarios.md](docs/release/memory-scenarios.md) for
compact end-to-end examples.

## Load the adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen2.5-7B-Instruct"
adapter_id = "AsadIsmail/prism-memory"

tokenizer = AutoTokenizer.from_pretrained(adapter_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_id,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, adapter_id)
```

This repo contains adapter weights only. You still need the base model.

## Training data

PRISM-Memory was trained on **synthetic** multi-session memory conversations
with **GPT-4.1-derived** memory-writing labels. The public release does not use
real user chat logs.

| Item | Count | Notes |
|---|---:|---|
| synthetic training conversations | `2,329` | multi-session conversations with inserts, updates, and deletes |
| synthetic held-out conversations | `584` | evaluation split used for held-out examples |
| supervised extraction examples | `100,427` | memory-writing labels derived from the synthetic corpus |
| released training subset | `20,000` | supervised examples used for the public adapter |

### Example training item

**Synthetic scenario**

- Domain: cloud infrastructure performance optimization
- Persona: senior cloud systems engineer at a fintech startup

**Synthetic user turn**

> Here’s the initial architecture outline: deploy microservices on AWS Fargate, use PostgreSQL 13 as the primary database, plan Kubernetes orchestration, use Redis for caching, and keep API latency under 50ms.

**Target memory records**

- Deploy microservices on AWS Fargate
- Orchestrate containers on a Kubernetes cluster (planned)
- Primary database: PostgreSQL 13
- Use Redis as an in-memory caching layer
- Latency target: API responses under 50ms

The release makes the dataset design, counts, and example records public. It
does not bundle the full raw corpus files.

## Confirmed results

| Benchmark | PRISM-Memory | GPT-4.1-based PropMem reference |
|---|---:|---:|
| LongMemEval | `{lme:.4f}` | `0.4650` |
| LoCoMo | `{locomo:.4f}` | `0.5360` |

The reproduced evaluation hit the cached QA surface exactly: `{qa_hits}` hits,
`{qa_misses}` misses.

## Extraction examples

{"\n\n".join(example_sections)}

More held-out examples live in
[docs/release/extraction-examples.md](docs/release/extraction-examples.md).

## Bundled docs and artifacts

- [docs/release/datasets.md](docs/release/datasets.md)
- [docs/release/extraction-examples.md](docs/release/extraction-examples.md)
- [docs/release/extraction-skill.md](docs/release/extraction-skill.md)
- [docs/release/memory-scenarios.md](docs/release/memory-scenarios.md)
- [docs/release/release-results.md](docs/release/release-results.md)
- [docs/release/technical-blog.md](docs/release/technical-blog.md)
- [results/release_summary.json](results/release_summary.json)
- [results/extraction_examples.json](results/extraction_examples.json)
- [results/benchmark_cases.json](results/benchmark_cases.json)

## Demo

The companion Space is live at
`https://huggingface.co/spaces/{space_repo}`.

## Limitations

- This is a memory-writing component, not a general chat model.
- It is a LoRA adapter, not a standalone full checkpoint.
- The evaluation pipeline still uses a separate QA model to score retrieved memory.
- Temporal and inferential categories still trail stronger larger-model baselines.
"""

(bundle_dir / "README.md").write_text(readme, encoding="utf-8")
PY

echo "Prepared model bundle at: $BUNDLE_DIR"

if [[ -z "$MODEL_REPO" ]]; then
  exit 0
fi

python - "$MODEL_REPO" "$BUNDLE_DIR" <<'PY'
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, upload_folder
except ImportError as exc:
    raise SystemExit(
        "huggingface_hub is required to upload the model. "
        "Install it with: python -m pip install huggingface_hub"
    ) from exc

repo_id = sys.argv[1]
bundle_dir = Path(sys.argv[2])

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=str(bundle_dir),
    commit_message="Publish PRISM-Memory adapter bundle",
    delete_patterns=[
        "docs/release/*.md",
        "results/*.json",
    ],
)
print(f"Uploaded bundle to https://huggingface.co/{repo_id}")
PY
