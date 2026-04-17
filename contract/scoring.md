# Scoring

## Philosophy

Use one scalar for selection, but never let the scalar hide catastrophic regressions.

The scalar is for ranking.
The guardrails are for veto.

## Required Inputs

Each benchmark should emit one normalized score in `[0, 1]`.

When a benchmark has multiple sub-metrics, reduce them to one benchmark score before aggregation and keep the detailed metrics in the raw result artifact.

## Tier Scores

### Tier 0

`tier0_score`

- `0.35 * alfworld_proxy`
- `0.35 * travelplanner_proxy`
- `0.30 * compiler_unit`

### Tier 1

`tier1_score`

- `0.25 * alfworld_seen_shard`
- `0.40 * alfworld_unseen_shard`
- `0.35 * travelplanner_dev_shard`

### Tier 2

`tier2_score`

- `0.22 * mem2actbench`
- `0.22 * memoryarena`
- `0.18 * memoryagentbench`
- `0.18 * proced_mem`
- `0.10 * shiftbench`
- `0.10 * efficiency_guardrail`

## Global Score

`global_score`

- `0.20 * tier0_score`
- `0.35 * tier1_score`
- `0.45 * tier2_score`

Tier 2 has the highest weight because narrow procedural benchmarks are not enough.

## Hard Guardrails

A candidate cannot be promoted if any of the following are true:

1. `alfworld_unseen_shard < incumbent - 0.02`
2. `travelplanner_dev_shard < incumbent - 0.02`
3. `shiftbench < incumbent - 0.03`
4. `proced_mem < incumbent - 0.03`
5. provenance / faithfulness checks fail

## Efficiency Guardrail

The `efficiency_guardrail` score should summarize:

1. token cost
2. latency
3. memory growth rate
4. replay / consolidation cost

If exact normalization is not yet available, record the raw values and set this score manually for early runs.

## Decision Rules

### Tier 0

Promote if:

1. `tier0_score > incumbent_tier0 + 0.01`, or
2. ties within `0.01` while materially improving efficiency or interpretability.

### Tier 1

Promote if:

1. `tier1_score > incumbent_tier1 + 0.01`, and
2. no hard guardrail fails.

### Tier 2 / Champion

Promote to champion if:

1. `global_score > incumbent_global + 0.01`, and
2. no hard guardrail fails, and
3. the run is fully logged.
