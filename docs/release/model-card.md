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

This release packages the best confirmed extraction checkpoint from the
`better_memory` project:

- **Checkpoint:** `exp15_sft_qwen7b_4ep`
- **Base model:** `Qwen/Qwen2.5-7B-Instruct`
- **Adapter type:** LoRA
- **Task:** proposition-level memory extraction for long-term dialogue systems

This is the only checkpoint that should be released publicly from the Exp15
line.

## Main Contribution

`PRISM-Memory` converts raw dialogue turns into explicit propositions and then
retrieves those propositions with a hybrid retrieval system. The model is not a
general chat model. It is a memory-writing component.

## Data Summary

- **Conversation source:** synthetic multi-session memory conversations
- **Label source:** GPT-4.1-derived proposition extractions
- **Released training slice:** `20k` examples sampled from `train_sft.jsonl`
- **Public status:** the release documents the data recipe and examples, but the
  full raw training JSONL files are not published in this repo

## Confirmed Results

The checkpoint was reproduced locally on the original evaluation surface.

- **LoCoMo mean:** `0.4981204463`
- **LongMemEval mean:** `0.4767574431`
- **QA cache hits:** `460`
- **QA cache misses:** `0`

LoCoMo category scores:

- factual: `0.3339551926`
- temporal: `0.4978785870`
- inferential: `0.2605997475`
- multi-hop: `0.5144477744`
- adversarial: `0.8837209302`

## Why This Checkpoint

This checkpoint was selected because it had the strongest total release profile:

- best confirmed LongMemEval score
- best confirmed adversarial score
- best balanced overall behavior after internal ablations

Later variants recovered some narrower category failures, but none produced a
better public release trade-off.

## The Skill To Pair With This Model

Public documentation should point to the single canonical extraction skill in
[extraction-skill.md](extraction-skill.md).

That skill keeps the interface simple:

1. extract `0-5` atomic standalone facts per turn
2. include dates when present
3. skip filler and questions
4. output JSON only

## Intended Use

This model is intended for:

1. long-term conversational memory systems
2. proposition extraction pipelines
3. benchmarked memory research

It is not intended to be used as a standalone assistant model.

## Limitations

- temporal and inferential reasoning still trail stronger larger-model baselines
- the extractor depends on downstream retrieval quality
- this is a LoRA adapter, not a self-contained full model
- the evaluation pipeline still uses a separate QA model to score retrieved memory

## Release Artifacts

- [extraction-skill.md](extraction-skill.md)
- [technical-blog.md](technical-blog.md)
- [../../results/confirmed_exp15_summary.json](../../results/confirmed_exp15_summary.json)
- [../../results/scenario_comparisons.json](../../results/scenario_comparisons.json)
- [../../space/app.py](../../space/app.py)

## Publishing Note

The public adapter weights live in the Hugging Face model repo
`AsadIsmail/prism-memory`. This file remains the repo-side source document for
that model card and for future release updates.
