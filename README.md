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

<p align="center"><em>Proposition-level conversational memory with one public extraction skill, one benchmarked release checkpoint, a live Space demo, and a broader hybrid-memory research harness.</em></p>

<p align="center">
  <a href="https://huggingface.co/spaces/AsadIsmail/prism-memory">Live Space</a> ·
  <a href="docs/release/README.md">Release Docs</a> ·
  <a href="docs/research/README.md">Research Docs</a>
</p>

![PRISM-Memory architecture](assets/prism-memory-architecture.svg)

## What PRISM-Memory Is

`PRISM-Memory` is not a chatbot wrapper and not a generic memory benchmark dump.
The public release is intentionally narrow:

1. one proposition extraction skill
2. one public checkpoint: `exp15_sft_qwen7b_4ep`
3. one public demo surface in `space/`

The broader `frontier_memory` package stays in the repo because it is the
actual runtime and research harness used to test memory behavior across
synthetic, TravelPlanner, ALFWorld, MemoryArena, and MemEval surfaces.

## Confirmed Release Metrics

| Checkpoint | LoCoMo | LongMemEval | Notes |
|---|---:|---:|---|
| `exp15_sft_qwen7b_4ep` | `0.4981` | `0.4768` | public release checkpoint |

Full release numbers, breakdowns, and artifact notes live in
[docs/release/release-results.md](docs/release/release-results.md).

## Demo In 60 Seconds

```bash
git clone git@github.com:Asad-Ismail/prism-memory.git && cd prism-memory
bash scripts/setup.sh --space
python space/app.py
```

That launches the same public demo locally with:

- the confirmed release metrics
- curated benchmark cases showing strengths and failure modes
- the single canonical extraction skill

## Repository Guide

Use these README files as the real entry points to the repo:

| Where to start | What it covers |
|---|---|
| [docs/README.md](docs/README.md) | top-level documentation index |
| [docs/release/README.md](docs/release/README.md) | the public release story: skill, datasets, results, model card, blog |
| [docs/research/README.md](docs/research/README.md) | the internal research program, contracts, and logging surface |
| [space/README.md](space/README.md) | the Gradio demo and Hugging Face Space bundle |
| [scripts/README.md](scripts/README.md) | setup, release helpers, and benchmark entrypoints |
| [frontier_memory/README.md](frontier_memory/README.md) | code-level map of the runtime and benchmark adapters |
| [candidates/README.md](candidates/README.md) | promoted policies and benchmark champions |
| [results/README.md](results/README.md) | tracked public artifacts and internal comparison payloads |
| [examples/README.md](examples/README.md) | the smallest end-to-end extraction and recall examples |
| [external/README.md](external/README.md) | optional local benchmark checkouts and env hooks |
| [tests/README.md](tests/README.md) | regression coverage and fixture layout |

## Project Layout

| Path | Purpose |
|---|---|
| `assets/` | repo visuals |
| `docs/` | release docs and research docs |
| `examples/` | toy dialogue, extraction, and recall examples |
| `frontier_memory/` | runtime package, search logic, and benchmark adapters |
| `space/` | first-class Gradio demo and Space bundle |
| `results/` | release-facing JSON artifacts kept small and readable |
| `scripts/` | setup, release helpers, and benchmark runners |
| `candidates/` | YAML policies and promoted champions |
| `contract/` | scoring and search contracts |
| `logs/` | append-only research notes and experiment history |
| `external/` | optional local benchmark checkouts, not part of the public payload |
| `tests/` | regression tests and committed fixtures |

## Common Tasks

Install the repo:

```bash
python -m pip install -e .
```

Run the demo:

```bash
make demo
```

Run the core test suite:

```bash
make test
```

Build a clean Space bundle:

```bash
make space-bundle
```

Run a benchmark entrypoint:

```bash
python scripts/run_synthetic_eval.py --candidate candidates/bootstrap_v0.yaml --dataset-size 8 --seed 7
python scripts/run_travelplanner_eval.py --policy-file candidates/travelplanner_champion.yaml
python scripts/run_memeval_eval.py --benchmark locomo --num-samples 1 --skip-judge
```

The full command guide is in [scripts/README.md](scripts/README.md).

## Release Surface

The public release documents are split cleanly under `docs/release/`:

| Doc | Why it exists |
|---|---|
| [docs/release/extraction-skill.md](docs/release/extraction-skill.md) | the single extraction behavior to keep |
| [docs/release/datasets.md](docs/release/datasets.md) | what data trained and evaluated the release |
| [docs/release/release-results.md](docs/release/release-results.md) | confirmed metrics and internal comparison notes |
| [docs/release/technical-blog.md](docs/release/technical-blog.md) | lessons learned from the repo history |
| [docs/release/model-card.md](docs/release/model-card.md) | draft HF model card for future weight release |

## Current Benchmark Status

Tracked public benchmark status in this repo:

- PRISM release checkpoint: LoCoMo `0.4981`, LongMemEval `0.4768`
- TravelPlanner validation champion: `1.0`
- ALFWorld seen champion: `1.0`
- ALFWorld unseen champion: `1.0`
- MemoryArena public suite: `1.0`

Those tracked artifacts are summarized in:

- [results/README.md](results/README.md)
- [results/frontier_memory_benchmarks.json](results/frontier_memory_benchmarks.json)

## Optional Local Benchmark Checkouts

The public GitHub repo does not vendor full benchmark assets. For full local
evaluation, point the runtime at local checkouts documented in
[external/README.md](external/README.md).

Most important env hooks:

- `PRISM_TRAVELPLANNER_ROOT`
- `PRISM_TRAVELPLANNER_VALIDATION_PATH`
- `BETTER_MEMORY_ROOT`
- `MEMEVAL_ROOT`
- `PRISM_LOCOMO_PATH`

## Notes

- The adapter weights are not bundled in this repo.
- The current public demo is the Hugging Face Space plus the local `space/` app.
- The repo keeps research provenance, but the public narrative is intentionally
  smaller than the full experiment history.
