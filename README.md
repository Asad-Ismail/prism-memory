<div align="center">

```text
██████╗ ██████╗ ██╗███████╗███╗   ███╗
██╔══██╗██╔══██╗██║██╔════╝████╗ ████║
██████╔╝██████╔╝██║███████╗██╔████╔██║
██╔═══╝ ██╔══██╗██║╚════██║██║╚██╔╝██║
██║     ██║  ██║██║███████║██║ ╚═╝ ██║
╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝
```

</div>

<p align="center"><strong>Turn conversations into durable, searchable memory.</strong></p>

<p align="center"><em>Proposition-level memory extraction for long-term dialogue, with a public Space demo and a broader hybrid-memory research harness.</em></p>

<p align="center">
  <a href="https://huggingface.co/spaces/AsadIsmail/prism-memory">Live Space</a> ·
  <a href="RELEASE_RESULTS.md">Confirmed Results</a> ·
  <a href="technical_blog.md">Technical Blog</a>
</p>

![PRISM-Memory architecture](assets/prism-memory-architecture.svg)

This repository has one clear public identity:

1. the released `PRISM-Memory` extraction skill and demo
2. the `frontier_memory` research harness used to keep iterating on hybrid
   memory systems

The public release is centered on one checkpoint, one extraction behavior, and
one demo surface instead of exposing every intermediate experiment.

## Demo In 60 Seconds

```bash
git clone git@github.com:Asad-Ismail/prism-memory.git && cd prism-memory
bash scripts/setup.sh --space
python space/app.py
```

That launches the same lightweight PRISM-Memory Space locally with:

- the released checkpoint metrics
- benchmark cases showing strengths and failure modes
- the single canonical extraction skill

## What This Repo Releases

- A proposition-level conversational memory extractor
- The canonical extraction contract in `MEMORY_EXTRACTION_SKILL.md`
- Confirmed benchmark artifacts for `exp15_sft_qwen7b_4ep`
- A lightweight Gradio Space in `space/`
- The broader research harness in `frontier_memory/`

## Main Contribution

The central claim is narrow and defendable:

`PRISM-Memory` replaces GPT-4.1 proposition extraction with a fine-tuned
`Qwen2.5-7B-Instruct` LoRA adapter while staying competitive on long-horizon
dialogue benchmarks.

This repo is not a general chatbot package. The released model is a
memory-writing component inside a larger retrieval system.

## Confirmed Release Results

| Checkpoint | LoCoMo | LongMemEval | Notes |
|---|---:|---:|---|
| `exp15_sft_qwen7b_4ep` | `0.4981` | `0.4768` | release checkpoint |

The public release centers on a single checkpoint. The full confirmed summary is
in `RELEASE_RESULTS.md` and `results/confirmed_exp15_summary.json`.

## Features

- proposition-level memory extraction from dialogue turns
- hybrid retrieval with sparse, dense, and reranked memory lookup
- release artifacts for benchmark confirmation and case studies
- a self-contained Gradio Space demo
- a research harness for synthetic, procedural, and agent-memory benchmarks
- adapters for TravelPlanner, ALFWorld, MemoryArena, and MemEval surfaces

## Repository Layout

| Path | Purpose |
|---|---|
| `assets/` | diagrams and public repo visuals |
| `examples/` | small end-to-end examples of dialogue, memory, and recall |
| `frontier_memory/` | research runtime and search harness |
| `space/` | lightweight public demo and Hugging Face Space bundle |
| `results/` | release-facing evaluation artifacts |
| `scripts/` | benchmark and search entrypoints |
| `tests/` | regression and smoke tests |
| `candidates/` | search candidates and promoted policies |
| `contract/` | benchmark/search contracts |
| `logs/` | append-only research logs |
| `external/` | optional local benchmark checkouts documented for the harness |

## Documentation

- `MEMORY_EXTRACTION_SKILL.md`: the single extraction skill to keep
- `technical_blog.md`: technical write-up of what worked and what failed
- `model_card.md`: draft model card for a future weight release
- `DATASETS.md`: training, evaluation, and auxiliary data summary
- `RELEASE_RESULTS.md`: confirmed release metrics and comparison notes
- `CONTRIBUTING.md`: contributor setup and repo hygiene rules
- `program.md`: internal research program for the autoresearch loop

## Installation

Install the basic repo dependencies:

```bash
python -m pip install -r requirements.txt
```

Install as a package:

```bash
python -m pip install -e .
```

Install dev extras:

```bash
python -m pip install -e ".[dev]"
```

or:

```bash
python -m pip install -r requirements-dev.txt
```

Install the Space dependencies:

```bash
python -m pip install -r space/requirements.txt
```

## Quick Start

Run the public demo locally:

```bash
python space/app.py
```

or:

```bash
make demo
```

Bootstrap a full local environment with the helper script:

```bash
bash scripts/setup.sh
```

Run the synthetic benchmark smoke test:

```bash
python scripts/run_synthetic_eval.py --candidate candidates/bootstrap_v0.yaml --dataset-size 8 --seed 7
```

or:

```bash
make synthetic-eval
```

Run one search iteration:

```bash
python scripts/run_search_iteration.py --base-candidate candidates/bootstrap_v0.yaml --num-children 4 --dataset-size 4 --seed 17
```

Run the TravelPlanner benchmark:

```bash
export PRISM_TRAVELPLANNER_ROOT=/path/to/TravelPlanner
python scripts/run_travelplanner_eval.py \
  --policy-file candidates/travelplanner_champion.yaml \
  --output-jsonl logs/runs/travelplanner_bootstrap_v0_validation.jsonl \
  --summary-path logs/runs/travelplanner_bootstrap_v0_validation_scores.json
```

Without `PRISM_TRAVELPLANNER_ROOT`, the repo falls back to a tiny committed
TravelPlanner fixture that is only meant for tests and smoke checks.

Run the full public MemoryArena suite:

```bash
python scripts/run_memoryarena_suite.py --output logs/runs/memoryarena_suite_v2.json
```

Run the MemEval adapter on a small benchmark slice:

```bash
python scripts/run_memeval_eval.py --benchmark locomo --num-samples 1 --skip-judge
```

Run the promoted OpenAI-backed synthetic memory benchmark:

```bash
python scripts/run_openai_synthetic_eval.py \
  --base-candidate candidates/openai_memory_champion.yaml \
  --model gpt-5.2 \
  --reasoning-effort medium \
  --dataset-size 2 \
  --seed 21 \
  --output logs/runs/openai_gpt52_medium_eval2_v3.json
```

Summarize the offline frontier benchmark status:

```bash
python scripts/summarize_offline_frontier.py --output logs/runs/offline_frontier_summary_v1.json
```

## Examples

The repo includes small concrete examples under `examples/` so the extraction
contract is visible without downloading weights or benchmarks:

- `examples/sample_dialogue.txt`: a short conversation with temporal updates
- `examples/sample_extraction.json`: the atomic facts the skill should write
- `examples/sample_recall.md`: a retrieval-style recall question over those facts
- `examples/README.md`: how these examples map to the release skill

Use them when you want to explain the system quickly or validate prompt changes
against a stable toy case.

## Datasets

The public release was trained mostly on synthetic long-horizon conversations
with GPT-4.1-derived proposition labels.

The raw source data is documented here, but it was generated in the upstream
`better_memory` experiment workspace and is not bundled in this repo.

Core files:

- `train.jsonl`: raw synthetic training conversations
- `train_sft.jsonl`: GPT-4.1 proposition extractions used for SFT
- `eval.jsonl`: raw held-out evaluation conversations
- `eval_sft.jsonl`: GPT-4.1 PropMem reference extractions

Auxiliary LoCoMo-targeted files exist for ablations, but the public release did
not come from heavy LoCoMo-domain training. The full breakdown is in
`DATASETS.md`.

## Research Harness

The `frontier_memory` package is still here because it is part of the actual
research workflow, not an unrelated add-on.

It includes:

- append-only episodic memory
- semantic fact storage and update handling
- procedural memory induction
- consolidation passes
- a search loop for candidate mutation and promotion
- adapters for synthetic, TravelPlanner, ALFWorld, MemoryArena, and MemEval surfaces

Current promoted benchmark policies:

- `candidates/travelplanner_champion.yaml`
- `candidates/alfworld_champion.yaml`
- `candidates/memoryarena_archive_champion.yaml`
- `candidates/openai_memory_champion.yaml`

The active research notes and contracts live in:

- `program.md`
- `contract/SEARCH_CONTRACT.md`
- `logs/journal.md`
- `logs/experiments.jsonl`

## External Benchmarks

Optional benchmark checkouts stay outside the committed GitHub payload and are
ignored by `.gitignore`. See `external/README.md` for the expected local layout
and the `PRISM_TRAVELPLANNER_ROOT` override for full TravelPlanner evaluation.

## Current Offline Frontier Status

The current offline benchmark ceiling in this repo is:

- TravelPlanner full validation: `1.0` final pass rate
- ALFWorld `valid_seen` slice: `1.0`
- ALFWorld `valid_unseen` slice: `1.0`
- MemoryArena public suite: `1.0` overall task exact match

The tracked summary for those results is in
`results/frontier_memory_benchmarks.json`.

The MemoryArena ceiling run uses archive-backed episodic replay plus a local
heuristic fallback for unmatched tasks. The archived benchmark policy is
`candidates/memoryarena_archive_champion.yaml`.

Local run dumps under `logs/runs/` are intentionally ignored from the first
public commit. Regenerate them locally from the commands above when you want the
full benchmark traces.

## Space Demo

The public demo lives in `space/`, following the same layout style used in the
`lore` repo.

Prepare a clean Space bundle locally:

```bash
bash scripts/deploy_space.sh
```

Upload the same bundle to Hugging Face:

```bash
bash scripts/deploy_space.sh AsadIsmail/prism-memory
```

## Tests

Run the core test suite:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

`tests/test_alfworld_benchmark.py` depends on optional ALFWorld and TextWorld
setup.

The repo also includes a deterministic committed TravelPlanner fixture for tests,
so the default suite does not need a live dataset pull.

## Notes

- The public release surface is intentionally narrow: one skill, one checkpoint,
  one demo.
- The Hugging Face Space can be published from `space/`.
- The adapter weights are not included in this repo.
