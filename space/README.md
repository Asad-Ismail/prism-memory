---
title: PRISM-Memory
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---

# PRISM-Memory Space

**Hook:** Turn conversations into durable, searchable memory.

This Space is the public demo for the released `PRISM-Memory 7B Adapter`, a
LoRA adapter on top of `Qwen/Qwen2.5-7B-Instruct`.

It matches the root repo story:

- one released model
- one extraction behavior
- one set of confirmed benchmark results
- one compact explanation of the synthetic training data and held-out examples

Within the GitHub repo, this directory is a first-class public surface. The
helper script `scripts/deploy_space.sh` bundles this folder with the release
artifacts needed for Hugging Face.

## Inputs

The app reads:

- `results/release_summary.json`
- `results/benchmark_cases.json`
- `results/extraction_examples.json`
- `docs/release/extraction-skill.md`
- `docs/release/memory-scenarios.md`
- `docs/release/datasets.md`

When copied into a standalone Hugging Face Space repo, keep those files beside
`app.py` and `requirements.txt`, preserving the `docs/release/` and `results/`
subdirectories.

## What It Shows

1. The confirmed metrics for the released model
2. Selected benchmark cases showing strengths and failure modes
3. Short end-to-end scenarios showing why the stored memory is useful later
4. Side-by-side held-out extraction examples against the GPT-4.1 reference
5. A compact description of the synthetic conversation data and supervision
6. The single canonical memory extraction skill to keep

## Local Run

```bash
python -m pip install -r requirements.txt
python app.py
```

## From The Repo Root

```bash
bash scripts/deploy_space.sh
```

Useful bundled files:

- `docs/release/extraction-skill.md`
- `docs/release/memory-scenarios.md`
- `docs/release/datasets.md`
- `docs/release/extraction-examples.md`
- `docs/release/release-results.md`
- `results/release_summary.json`
- `results/extraction_examples.json`
