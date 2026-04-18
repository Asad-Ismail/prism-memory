[Back to Repo](../../README.md) · [Release Docs](README.md) · [Result Artifacts](../../results/README.md)

# PRISM-Memory Extraction Skill

**Hook:** Turn conversations into durable, searchable memory.

This is the single extraction skill the public release keeps.

- **Released model:** `PRISM-Memory 7B Adapter`
- **Base model:** `Qwen/Qwen2.5-7B-Instruct`
- **Role:** proposition extraction for long-term conversational memory
- **Why this one:** strongest confirmed overall release profile, strongest
  adversarial behavior, and best confirmed LongMemEval score among the release
  candidates

## Skill Definition

The extractor operates turn by turn and emits `0-5` atomic memory records per
turn. Each record should be a standalone fact about a person, event,
preference, plan, or property, with dates carried into the fact when available.

Canonical prompt:

```text
You are a memory extraction assistant. Given a conversation turn, extract 0-5 atomic, standalone facts. Each fact must be a complete sentence about a specific person, event, preference, or property. Include dates/times when mentioned. Skip greetings, filler, and questions. Output ONLY a JSON array of strings, e.g. ["fact1", "fact2"] or [].
```

## Inference Contract

1. Format the current turn with speaker and session date.
2. Extract `0-5` propositions as a JSON array.
3. Clean speaker references so generic labels become real names when possible.
4. Resolve relative temporal expressions against the session date.
5. Prefix each stored proposition with the normalized session date before indexing.
6. Pair the extractor with the hybrid retrieval stack, not with raw transcript search alone.

## Retrieval Setup To Keep

- **Retriever:** `PRISMv3Rerank`
- **Sparse retrieval:** BM25
- **Dense retrieval:** `all-MiniLM-L6-v2`
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

Best confirmed retrieval settings:

- **LoCoMo:** adversarial `k=5`, multi-hop `k=10`, all other categories `k=8`
- **LongMemEval:** multi-session `k=20`, all other categories `k=8` except
  single-session-user `k=5`

## What Held Up In The Repo

1. The stable `20,000`-example supervised base mattered more than aggressive
   benchmark-specific add-ons.
2. Four epochs was enough to reach the useful local optimum for this 7B line.
3. Explicit date anchoring helped. Benchmark-style relative-date imitation did not.
4. Post-processing mattered. Speaker cleanup and relative-date resolution made
   the extracted records usable.
5. Hybrid retrieval beat simpler sparse-only or dense-only retrieval.
6. Turn-local extraction worked better than feeding long recent-context windows
   into the extractor.

## What To Avoid

1. Benchmark-specific format hacks, especially relative-date answer imitation.
2. Narrow LoCoMo-style SFT add-ons that improve one slice and hurt balance.
3. Overtraining follow-on variants that trade adversarial precision for narrow gains.
4. Treating the extractor as a standalone answer model instead of a memory writer.

## Release Rule

Public surfaces should expose exactly one extraction behavior and one released
model. Other runs remain internal research artifacts.

Related docs:

- [datasets.md](datasets.md)
- [release-results.md](release-results.md)
- [technical-blog.md](technical-blog.md)
