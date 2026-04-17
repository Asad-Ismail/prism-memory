[Back to Repo](../../README.md) · [Release Docs](README.md) · [Extraction Skill](extraction-skill.md)

# PRISM-Memory: Turn Conversations Into Durable, Searchable Memory

## Summary

`PRISM-Memory` is a long-term conversational memory system that converts raw
dialogue into proposition-level memory and retrieves it with an inspectable
hybrid stack.

The point is not that a 7B model chats well. The point is that a 7B open model
can write memory records that another system can actually use later.

This package now ships one public extraction skill and one public checkpoint:

- **Checkpoint:** `exp15_sft_qwen7b_4ep`
- **Confirmed LoCoMo mean:** `0.4981204463`
- **Confirmed LongMemEval mean:** `0.4767574431`
- **QA cache misses during confirmation:** `0`

The public hook is simple:

**PRISM-Memory turns conversations into durable, searchable memory.**

## Why This Is Useful In Practice

A memory writer is only interesting if a later system can ask a pointed
question and get back a useful answer without rereading the original chat. The
public release artifacts already show that pattern.

### 1. Keep hard limits and preferences available for later work

The extractor can turn a single conversational turn into stable memory like:

- GitHub Actions concurrency limit: `20` concurrent jobs
- Snyk Slack notifications should be aggregated and concise

That means a later system can answer:

> What is our GitHub Actions concurrency limit, and how should Snyk alerts look?

with:

> `20` concurrent jobs. Alerts should be aggregated and concise.

That is a real product use case. Teams mention constraints and preferences once,
then expect downstream tools and agents to remember them.

### 2. Keep current state separate from the roadmap

The released extractor can also preserve the difference between what is true
now and what is only planned:

- sidecar CPU limits are already set and monitored
- mTLS is planned for phase two
- rollout strategy is canary deployments plus traffic splitting

So a later question like:

> Did we already enable mTLS, and what rollout strategy are we planning?

can be answered without confusing the present state with the future plan.

This is a core memory problem, not a style problem. Chat history tends to blur
these states together.

### 3. Answer dated questions with dated evidence

One confirmed held-out benchmark case asks:

> Which hobby did Sam take up in May 2023?

The retrieved memory contains explicit dated propositions about Sam trying
painting in May 2023, and the released system answers:

> painting

That matters because the useful behavior is not “remember that hobbies were
discussed.” The useful behavior is “recover the dated fact that actually
answers the later question.”

There is a fourth practical behavior that matters too: refusal. On the held-out
adversarial guitar case, the released model returns `None` instead of inventing
a reason for an unsupported premise. That is also part of being useful.

For the compact scenario version of this story, see
[memory-scenarios.md](memory-scenarios.md).

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
