[Back to Repo](../../README.md) · [Release Docs](README.md) · [Extraction Skill](extraction-skill.md)

# PRISM-Memory: Turn Conversations Into Durable, Searchable Memory

## The Problem

Most long-chat systems do not actually have memory. They have transcript
search. That works until someone asks a later question that depends on a hard
constraint, a changed plan, a dated fact, or a contradiction that happened
months ago.

PRISM-Memory focuses on the part of the stack that usually stays hidden: the
step that decides what should become memory at all.

The release model is a 7B adapter that writes short proposition-level memory
records from dialogue. Those records are then indexed by a hybrid retrieval
stack and used later for recall.

## What This Release Shows

The useful result is narrow and practical:

- a 7B open model can replace the GPT-4.1 extraction step in this memory pipeline
- it scores `0.4768` on LongMemEval versus `0.4650` for the GPT-4.1-based PropMem reference
- it scores `0.4981` on LoCoMo versus `0.5360` for that same reference

This is not a claim that a 7B model beats GPT-4.1 everywhere. It is a claim
that a 7B model can take over the memory-writing step and stay competitive on
the held-out evaluation surface.

## Why That Matters

If the memory-writing step is weak, retrieval never gets a clean chance.
Important details stay buried inside noisy chat turns.

PRISM-Memory is useful when later questions depend on things like:

- a hard operational limit: `20` GitHub Actions jobs
- a durable preference: aggregated Slack alerts instead of noisy ones
- a status distinction: mTLS is not live yet, it is planned for phase two
- a dated fact: Sam took up painting in May 2023
- a refusal case: the system should answer `None` instead of inventing a reason for an unsupported guitar story

Those are memory problems, not style problems.

## How The System Works

The released system has three pieces.

1. A learned extractor based on `Qwen/Qwen2.5-7B-Instruct` with LoRA.
2. Post-processing that cleans speaker references and resolves relative time.
3. Hybrid retrieval with BM25, dense retrieval, and reranking.

The extracted propositions are the important interface. They are the memory
records the retriever indexes. That keeps the memory store inspectable instead
of opaque.

## What The Training Data Actually Was

The release data is synthetic.

- `2,329` synthetic training conversations
- `584` held-out synthetic conversations
- `100,427` supervised extraction examples derived from those conversations
- `20,000` supervised examples used for the released adapter

The conversations were designed to stress real memory behaviors:

- new facts introduced in one session and used later
- updated details that should overwrite stale ones
- deleted or invalidated facts that should stop influencing answers
- mixtures of personal details, project facts, preferences, dates, and plans

The labels were GPT-4.1-derived memory-writing targets. No real user chat logs
are part of the public release.

## What Worked

### 1. The clean supervised base mattered more than clever add-ons

The release model came from a stable `20,000`-example synthetic supervision
base. That base was more valuable than trying to patch the model later with
many narrow benchmark-specific additions.

### 2. Hybrid retrieval was part of the result

The release is not just a model story. It is a model-plus-retrieval story.
Sparse retrieval kept lexical anchors, dense retrieval recovered semantically
close memories, and reranking cleaned the shortlist.

### 3. Explicit time anchoring helped

The model improved when the memory records carried explicit dates and the system
resolved relative references like `yesterday` or `last weekend` into normalized
anchors.

### 4. Turn-local extraction was enough

Feeding long recent-context windows into the extractor made it worse. The
stronger pattern was local extraction at write time and cross-turn composition
later through retrieval.

### 5. Adversarial precision mattered

The release model kept the best adversarial behavior among the runs considered
for public release. That mattered because a memory system that answers
unsupported questions confidently is worse than one that refuses.

## What Did Not Work

### 1. Benchmark-style formatting tricks

Trying to train the model toward benchmark-style relative-date outputs hurt more
than it helped. It optimized the look of answers instead of the quality of the
stored memory.

### 2. Narrow LoCoMo-style add-ons

Adding targeted benchmark-domain data often bought a small gain in one slice of
LoCoMo and then lost balance somewhere else.

### 3. More noisy supervision was not automatically better

Scaling up original noisy temporal supervision amplified the wrong lesson. The
model became more specialized and less balanced.

### 4. Overtraining past the local optimum

Several follow-on variants nearly matched the final release on one metric, but
they usually gave back LongMemEval performance, adversarial precision, or both.

## Why Only One Public Model Ships

The repo tried multiple follow-on variants. The nearest internal runner-up
nearly tied the released model on overall LoCoMo and disagreed on `152` of the
`400` held-out LoCoMo questions, which means the comparison was real.

But the public release decision is simpler than the internal ablation story.
One model ships because it had the best overall release profile:

- strongest LongMemEval score
- strongest adversarial behavior
- best total balance across the held-out surface

That is a better public story than shipping several near-tied variants with
internal names nobody else should care about.

## What Ships

The public release surface is intentionally narrow:

1. `PRISM-Memory`
2. one released model
3. one extraction skill
4. one Space demo
5. one set of release docs and benchmark artifacts

The broader `frontier_memory` harness stays in the repo for ongoing research,
but the release story stays focused on the memory-writing component that proved
worth shipping.
