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

This Space is the lightweight public demo for the single released
`PRISM-Memory` extraction skill. It shows the best checkpoint only.

Within the GitHub repo, this directory is a first-class public surface. The
helper script `scripts/deploy_space.sh` bundles this folder with the release
artifacts needed for Hugging Face.

## Inputs

The app reads:

- `results/confirmed_exp15_summary.json`
- `results/scenario_comparisons.json`
- `docs/release/extraction-skill.md`

When copied into a standalone Hugging Face Space repo, keep those files beside
`app.py` and `requirements.txt`, preserving the `docs/release/` and `results/`
subdirectories.

## What It Shows

1. The confirmed metrics for the released checkpoint
2. Selected benchmark cases showing strengths and failure modes
3. The single canonical memory extraction skill to keep

## Local Run

```bash
python -m pip install -r requirements.txt
python app.py
```

## From The Repo Root

```bash
bash scripts/deploy_space.sh
```

Related docs:

- [../docs/release/extraction-skill.md](../docs/release/extraction-skill.md)
- [../docs/release/release-results.md](../docs/release/release-results.md)
- [../results/README.md](../results/README.md)
