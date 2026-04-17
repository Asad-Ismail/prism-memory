# Candidates

This directory contains policy YAMLs for the broader `frontier_memory` harness.

## Promoted Files

| File | Purpose |
|---|---|
| `bootstrap_v0.yaml` | baseline local synthetic-memory policy |
| `champion_latest.yaml` | current promoted synthetic-memory champion |
| `travelplanner_bootstrap.yaml` | TravelPlanner starting policy |
| `travelplanner_champion.yaml` | promoted TravelPlanner champion |
| `alfworld_champion.yaml` | promoted ALFWorld procedural-memory policy |
| `memoryarena_archive_champion.yaml` | promoted MemoryArena archive-backed policy |
| `openai_memory_champion.yaml` | provider-backed synthetic-memory champion |

## How These Are Used

- local benchmark runners load these YAMLs directly
- search scripts mutate them and promote winners
- the public PRISM extractor release is documented under `docs/release/`, but
  these policies remain useful for the broader runtime package

Generated children stay under `candidates/generated/` and are intentionally not
part of the public GitHub payload.
