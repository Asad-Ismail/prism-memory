[Back to Repo](../../README.md) · [Research Docs](README.md) · [Contracts](../../contract/README.md)

# Frontier Memory Research Program

This file is the operating note for the `frontier_memory` research harness that
lives alongside the public `PRISM-Memory` release.

## Goal

Find the best-performing memory system we can build right now, while keeping every iteration reviewable and every pivot explicit.

## Non-Negotiables

1. Keep an append-only log of experiments, decisions, and pivots.
2. Preserve frozen benchmark surfaces.
3. Prefer small mutations over broad rewrites.
4. Keep raw episodic traces as source of truth.
5. Treat semantic, episodic, and procedural memory as separate but connected concerns.

## Default Loop

1. Read the root `README.md`, [../../contract/SEARCH_CONTRACT.md](../../contract/SEARCH_CONTRACT.md), the latest entries in [../../logs/journal.md](../../logs/journal.md), and the active candidate YAML.
2. If runtime behavior is broken, fix the memory engine before adding more search surface.
3. Choose one hypothesis only.
4. Mutate a narrow surface:
   - consolidation thresholds
   - dedup / split / prune policy
   - retrieval routing
   - reliability scoring
   - prompt wording
   - orchestration placement
5. Run the cheapest benchmark tier that can falsify the hypothesis.
6. Score the result.
7. Append both a journal entry and a JSONL record.
8. Promote, reject, or pivot.

## First Priority

Before broad architecture search, stabilize a common comparison surface around these families:

1. `MACLA`-style hierarchical procedural memory.
2. `MemP`-style build / retrieve / update procedural memory.
3. `AWM`-style workflow induction.
4. `LEGOMem`-style orchestrator / worker memory placement.
5. `Evolving Memory`-style consolidation and failure-aware dreaming.

## What To Mutate

Prefer changing one or two of:

- `consolidation.passes`
- `consolidation.thresholds`
- `procedural.reliability`
- `retrieval.router`
- `retrieval.top_k`
- `controller.write_gates`
- `controller.architecture`
- `controller.use_policy`

## What Not To Mutate

1. Benchmark labels or gold data.
2. Historical log entries.
3. The scoring weights unless there is a documented contract change.
4. Multiple unrelated candidate families in one experiment.

## Pivot Rules

Pivot when one of these is true:

1. Three consecutive regressions from the same idea family.
2. Gains on `ALFWorld` do not transfer to `TravelPlanner`.
3. Procedural gains fail to move broader agent-memory tiers.
4. Consolidation improves average score but hurts shift or procedural recovery.

## Champion Rule

A candidate is only champion if it beats the incumbent on weighted cross-benchmark score and does not fail any hard guardrail.
