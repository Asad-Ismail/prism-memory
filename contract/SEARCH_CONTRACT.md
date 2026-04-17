# Search Contract

## Objective

Discover a memory system that performs strongly across:

- procedural tasks
- long-horizon planning
- multi-session memory use
- recovery under shift
- procedural retrieval and reuse

The search should optimize architecture and policy, not just prompts.

## Search Surface

Every candidate lives in six modules:

1. `encode`: how traces, facts, and actions are represented
2. `store`: what memory stores exist
3. `consolidate`: merge, split, prune, deprecate, and summarize passes
4. `retrieve`: routing and ranking logic
5. `use`: how memory is injected into planning / acting
6. `update`: when memory is written, repaired, or forgotten

## Required Memory Layers

Every serious candidate must declare explicit choices for:

1. `episodic`: append-only experience source of truth
2. `semantic`: distilled facts, entities, preferences, and updates
3. `procedural`: reusable routines, scripts, or playbooks

Optional but encouraged:

1. `tool_memory`
2. `orchestrator_memory`
3. `worker_memory`
4. `negative_constraints`

## Candidate Rules

Every candidate must:

1. Have a unique `candidate_id`.
2. Declare parent lineage.
3. State one main hypothesis.
4. List the exact mutation surface for the run.
5. Define benchmark tiers to run.
6. Record promotion status.

## Benchmark Tiers

### Tier 0: Cheap falsification

Purpose:

- reject bad consolidation or routing ideas quickly
- test replay and compiler logic before live benchmark spend

Typical surfaces:

- frozen trace replay
- compiler unit checks
- procedural retrieval proxy

### Tier 1: Core procedural benchmarks

Purpose:

- test whether procedural memory actually improves task execution

Required:

- `ALFWorld`
- `TravelPlanner`

### Tier 2: Broader agent-memory generalization

Purpose:

- make benchmark gaming obvious

Required:

- `Mem2ActBench`
- `MemoryArena`
- `MemoryAgentBench`
- `PROCED-MEM`
- `ShiftBench`

Optional:

- `EMemBench`
- `MemGUI-Bench`

## Promotion Rules

### Promote Tier 0 -> Tier 1

Only if:

1. Proxy score improves or ties within tolerance.
2. No critical compiler or replay regressions.
3. Consolidation does not destroy provenance.

### Promote Tier 1 -> Tier 2

Only if:

1. `ALFWorld` unseen and `TravelPlanner` both improve, or
2. one improves materially and the other stays within tolerance with a clear efficiency win.

### Champion

Only if:

1. Global weighted score beats incumbent.
2. No hard guardrail fails.
3. Results are logged with full lineage.

## Hard Guardrails

Reject a candidate if any of these happen:

1. Severe regression on any required Tier 1 benchmark.
2. Shift recovery collapse after consolidation.
3. Procedure library growth without measurable performance gain.
4. Deduplication removes critical rare cases.
5. Agent behavior becomes less faithful to source traces.

## Pivot Rules

Pivot the search family when:

1. Three runs in the same family regress.
2. `ALFWorld` rises but `TravelPlanner` stays flat.
3. Procedural metrics rise but broader agent-memory tiers do not.
4. Shift or procedural-retrieval metrics reveal over-consolidation.

## Nightly Dream Patch Loop

The consolidation job should emit explicit patch operations instead of silently rewriting memory:

1. extract negative constraints from failures
2. mark critical vs noise actions
3. merge near-duplicate routines
4. split over-broad or low-precision routines
5. compose higher-level playbooks
6. deprecate stale memories
7. compact summaries for retrieval

Each patch must remain linked to source traces.

## Logging Rules

For every run, append:

1. one machine-readable entry to `logs/experiments.jsonl`
2. one human-readable entry to `logs/journal.md`

No overwriting past runs. Corrections are new entries.
