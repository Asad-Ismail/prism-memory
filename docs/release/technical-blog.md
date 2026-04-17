[Back to Repo](../../README.md) · [Release Docs](README.md) · [Extraction Skill](extraction-skill.md)

# PRISM-Memory: Turn Conversations Into Durable, Searchable Memory

## Summary

`PRISM-Memory` is a long-term conversational memory system that converts raw
dialogue into proposition-level memory and retrieves it with an inspectable
hybrid stack.

This package now ships one public extraction skill and one public checkpoint:

- **Checkpoint:** `exp15_sft_qwen7b_4ep`
- **Confirmed LoCoMo mean:** `0.4981204463`
- **Confirmed LongMemEval mean:** `0.4767574431`
- **QA cache misses during confirmation:** `0`

The public hook is simple:

**PRISM-Memory turns conversations into durable, searchable memory.**

## What The Repo Actually Contributed

The core contribution is not another opaque memory model. The repo showed that a
7B open model can replace GPT-4-class extraction with a transparent memory
pipeline that is still competitive on long-horizon dialogue benchmarks.

The released system has three pieces:

1. A learned proposition extractor (`Qwen2.5-7B-Instruct` + LoRA).
2. Post-processing that cleans speaker references and resolves relative time.
3. Hybrid retrieval (`BM25 + dense retrieval + cross-encoder reranking`).

The important part is the interface between them: extracted propositions are not
just text snippets. They are the memory records that the retriever indexes.

## The Single Skill To Keep

After reviewing the repo history, there should be one canonical extraction skill
and one checkpoint publicly exposed:

- **Skill:** proposition-level memory extraction
- **Model:** `exp15_sft_qwen7b_4ep`
- **Prompt contract:** extract `0-5` atomic standalone facts, include dates when
  present, skip filler and questions, output JSON only

That skill is documented directly in
[extraction-skill.md](extraction-skill.md).

## What Worked

### 1. The best model came from the stable 20k base, not from aggressive add-ons

The repo repeatedly showed that `sft4` was the stable optimum for the 7B line.
The same 20k clean base distribution was critical. Changing the base subset,
shrinking it, or overextending it consistently hurt.

Why that matters:

- the model needed the exact data distribution that produced `sft4`
- 4 epochs was enough to reach the useful local optimum
- follow-on runs often traded away robustness for narrower gains

### 2. Proposition memory plus hybrid retrieval is the real winning combination

The strongest system was not latent-only memory and not raw-turn retrieval. The
best path was proposition extraction plus `PRISMv3Rerank`.

That means:

- sparse retrieval captured lexical anchors
- dense retrieval recovered semantically close memories
- reranking cleaned up the final shortlist

This combination is what made the memory store usable.

### 3. Absolute date anchoring and temporal cleanup helped

Temporal improvement came from making the memory records cleaner, not from
teaching the model to imitate LoCoMo’s relative-answer style.

What helped:

- fixed temporal examples with explicit date resolution
- normalizing session dates
- post-processing relative references like `yesterday` or `last weekend`

What did **not** help:

- training the model to emit relative benchmark-style dates

### 4. Turn-local extraction was better than passing long context windows

The repo tested extraction with added session context and it regressed. The
model worked best when extracting from the current turn and letting the memory
system handle cross-turn reasoning later.

That is an important design lesson: keep extraction local, let retrieval do the
composition.

### 5. Adversarial precision was the strongest reason to keep `sft4`

Many later variants found small gains in temporal or inferential categories, but
they usually damaged adversarial behavior. `sft4` held the best confirmed
adversarial score and the best total LongMemEval score, which is why it is the
only checkpoint worth releasing publicly.

## What Did Not Work

### 1. Benchmark-specific format hacks

Relative-date training was a dead end. It optimized for the look of a benchmark
answer rather than for general extraction quality.

### 2. LoCoMo-domain training data

Adding LoCoMo training conversations consistently regressed performance. The
best generalization signal remained the cleaned LME-style base data.

### 3. More original LME data was not better

Scaling from 20k to 50k original LME examples amplified the temporal-anchor
problem. More noisy temporal labels simply taught the wrong lesson more often.

### 4. Small follow-on bases and heavy QA multipliers

Runs built on 5k clean bases or extreme QA multipliers tended to forget useful
behavior. They often improved a narrow category while hurting adversarial
precision, inferential balance, or LongMemEval.

### 5. Assuming the best checkpoint was easy to improve

The repo’s most expensive lesson was that `sft4` was already a local optimum for
the 7B line. Most additional training made the model more specialized and less
balanced.

## Internal Comparisons That Informed The Release

The internal ablation story still matters, even though the public package keeps
only `sft4`.

Confirmed internal facts:

- `inferential_from_temporal_heavy` nearly tied `sft4` on overall LoCoMo
- it recovered some inferential and temporal misses
- it still lost on LongMemEval and adversarial precision

Question-level comparison on held-out LoCoMo:

- `400` questions replayed
- `152` answer-level disagreements
- `56` questions favored `sft4`
- `52` questions favored the runner-up

That is a useful research result, but not a reason to ship two public models.
The right release decision is one clean skill, one clean checkpoint.

## Failure Modes Still Visible In The Release Model

The selected model is good enough to release, but its errors are clear:

- it can miss specific diagnoses while retaining the broader health frame
- it can overcommit to a salient retrieved clue in inferential questions
- it can remember a coarse book description but miss the exact title

Those are not packaging issues. They are the current limits of the extraction +
retrieval stack at this model size.

## What Ships

Public release surface:

1. `PRISM-Memory`
2. the single extraction skill in [extraction-skill.md](extraction-skill.md)
3. the best confirmed checkpoint `exp15_sft_qwen7b_4ep`
4. the best-only Space demo in [../../space/](../../space/)

Internal analysis artifacts can stay for provenance, but they should not be
positioned as parallel public releases.
