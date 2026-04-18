---
base_model: Qwen/Qwen2.5-7B-Instruct
license: apache-2.0
library_name: peft
tags:
- conversational-memory
- information-extraction
- long-context
- peft
- lora
- qwen2.5
---

[Back to Repo](../../README.md) · [Release Docs](README.md) · [Release Results](release-results.md)

# Model Card: PRISM-Memory

**Hook:** Turn conversations into durable, searchable memory.

## Summary

This release packages one public model:

- **Model name:** `PRISM-Memory 7B Adapter`
- **Base model:** `Qwen/Qwen2.5-7B-Instruct`
- **Adapter type:** LoRA
- **Task:** proposition-level memory extraction for long-term dialogue systems

The model is a memory-writing component. It is not intended to be used as a
standalone chat assistant.

## Main Contribution

PRISM-Memory converts raw dialogue turns into explicit memory records that a
retrieval system can index directly. The release shows that a 7B open model can
replace the GPT-4.1 extraction step in this memory pipeline while staying
competitive on long-horizon dialogue benchmarks.

## Data Summary

- **Conversation source:** synthetic multi-session conversations
- **Label source:** GPT-4.1-derived memory-writing labels
- **Supervised label bank:** `100,427` examples
- **Released training subset:** `20,000` examples
- **Public status:** the release documents the dataset design, counts, and
  example records, but it does not bundle the full raw corpora

The full data description lives in [datasets.md](datasets.md).

## Confirmed Results

The release model was reproduced locally on the original evaluation surface.

| Benchmark | Score |
|---|---:|
| LoCoMo | `0.4981204463` |
| LongMemEval | `0.4767574431` |

QA cache behavior during confirmation:

- hits: `460`
- misses: `0`

LoCoMo category scores:

- factual: `0.3339551926`
- temporal: `0.4978785870`
- inferential: `0.2605997475`
- multi-hop: `0.5144477744`
- adversarial: `0.8837209302`

## Why This Model Ships

This model had the strongest public release profile:

- best confirmed LongMemEval score among the 7B runs kept for release review
- strongest confirmed adversarial behavior
- best overall balance after the internal follow-on experiments

Several later variants recovered narrower failures, but none produced a better
public release trade-off.

## Intended Use

This model is intended for:

1. long-term conversational memory systems
2. proposition extraction pipelines
3. benchmarked memory research

It is not intended for:

1. general chat
2. open-ended reasoning without a retrieval layer
3. standalone assistant deployment without downstream memory infrastructure

## Limitations

- temporal and inferential reasoning still trail stronger larger-model baselines
- extraction quality still depends on downstream retrieval quality
- this is a LoRA adapter, not a self-contained full model
- the evaluation pipeline still uses a separate QA model to score retrieved memory

## Release Artifacts

- [datasets.md](datasets.md)
- [extraction-skill.md](extraction-skill.md)
- [memory-scenarios.md](memory-scenarios.md)
- [technical-blog.md](technical-blog.md)
- [../../results/release_summary.json](../../results/release_summary.json)
- [../../results/benchmark_cases.json](../../results/benchmark_cases.json)
- [../../space/app.py](../../space/app.py)

## Publishing Note

The public adapter weights live in the Hugging Face model repo
`AsadIsmail/prism-memory`. This file is the repo-side source document for that
model card and for future release updates.
