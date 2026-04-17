# PRISM-Memory Extraction Skill

**Hook:** Turn conversations into durable, searchable memory.

This is the single extraction skill to keep from the `better_memory` work.
Public release should point to one checkpoint and one extraction behavior:

- **Model:** `exp15_sft_qwen7b_4ep`
- **Base model:** `Qwen/Qwen2.5-7B-Instruct`
- **Role:** proposition extraction for long-term conversational memory
- **Why this one:** best confirmed total profile, best adversarial behavior, and
  best LongMemEval score

## Skill Definition

The extractor operates turn by turn and emits `0-5` atomic propositions per
turn. Each proposition should be a standalone fact about a person, event,
preference, or property, with dates carried into the fact when available.

Canonical prompt:

```text
You are a memory extraction assistant. Given a conversation turn, extract 0-5 atomic, standalone facts. Each fact must be a complete sentence about a specific person, event, preference, or property. Include dates/times when mentioned. Skip greetings, filler, and questions. Output ONLY a JSON array of strings, e.g. ["fact1", "fact2"] or [].
```

This prompt comes from
`experiment15_learned_extraction.py`.

## Inference Contract

1. Format the turn with speaker and session date.
2. Extract `0-5` propositions as a JSON array.
3. Clean speaker references so generic labels become real names.
4. Resolve relative temporal expressions against the session date.
5. Prefix each proposition with the normalized session date before indexing.
6. Retrieve with the PRISM hybrid stack, not with the extractor alone.

## Retrieval Setup To Keep

- **Retriever:** `PRISMv3Rerank`
- **Sparse retrieval:** BM25
- **Dense retrieval:** `all-MiniLM-L6-v2`
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

Best confirmed retrieval settings:

- **LoCoMo:** adversarial `k=5`, multi-hop `k=10`, all other categories `k=8`
- **LongMemEval:** multi-session `k=20`, all other categories `k=8` except
  single-session-user `k=5`

## What Worked

1. **The original 20k base mattered.**
   `sft4` came from the exact `train_sft_clean_merged.jsonl` base distribution.
   Runs that changed the base subset regressed.

2. **Four epochs was the sweet spot.**
   `sft4` is the local optimum the repo could actually reproduce.

3. **Absolute date anchoring helped.**
   Temporal repairs worked when the model saw explicit, normalized dates rather
   than benchmark-specific relative phrasing.

4. **Post-processing mattered.**
   Speaker cleanup plus relative-date resolution was necessary to turn raw
   outputs into stable memory records.

5. **Hybrid retrieval beat simpler retrieval.**
   BM25 + dense + reranking consistently outperformed BM25-only or dense-only
   approaches.

6. **Turn-local extraction was enough.**
   The model performed better without feeding long recent-context windows into
   the extractor.

7. **Multihop supervision preserved inferential behavior.**
   When temporal data was added, multihop QA was the only extra signal that
   reliably helped preserve inferential performance.

## What Did Not Work

1. **Relative-date training.**
   Training the extractor to emit benchmark-style relative dates hurt temporal
   performance instead of helping it.

2. **LoCoMo-domain SFT data.**
   Adding LoCoMo training conversations consistently regressed the model.

3. **More than 20k original LME examples.**
   Scaling the original noisy temporal labels to 50k amplified anchor loss and
   caused major regression.

4. **Small clean bases.**
   5k-base follow-on runs forgot too much and collapsed inferential behavior.

5. **Heavy QA multipliers.**
   High temporal or QA multipliers damaged adversarial precision and LongMemEval.

6. **High learning rates on follow-on QA runs.**
   Aggressive fine-tuning degraded the traits that made `sft4` good.

7. **Trying to push past the local optimum.**
   Most post-`sft4` training traded away adversarial performance for narrower
   gains.

## Release Rule

Release only this extraction skill and only this checkpoint publicly:

- `exp15_sft_qwen7b_4ep`

Treat all other checkpoints as internal ablations and learning artifacts, not as
parallel public releases.
