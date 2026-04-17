# frontier_memory

`frontier_memory` is the runtime package behind the repo.

It includes the actual hybrid-memory implementation, search utilities, and the
benchmark adapters that the repo uses to test policies beyond the public PRISM
release.

## Module Groups

| Area | Files | Purpose |
|---|---|---|
| Core system | `system.py`, `types.py`, `config.py`, `utils.py` | shared runtime types and system assembly |
| Memory stores | `episodic.py`, `semantic.py`, `procedural.py`, `consolidation.py` | memory layers and consolidation behavior |
| Routing and search | `router.py`, `search.py` | retrieval routing and candidate mutation/evaluation flow |
| Synthetic benchmarks | `benchmarks.py`, `synthetic_benchmarks.py` | fast evaluation surface for iteration |
| External benchmark adapters | `travelplanner.py`, `travelplanner_search.py`, `alfworld_benchmark.py`, `memoryarena_*`, `memeval_adapter.py` | benchmark-specific runtime bridges |
| LLM integration | `llm_backend.py` | provider-backed answering and prompt profiles |

## How This Relates To PRISM-Memory

The public release is one narrow piece of this package:

- the extraction contract documented in [../docs/release/extraction-skill.md](../docs/release/extraction-skill.md)
- the confirmed artifacts in [../results/README.md](../results/README.md)
- the demo in [../space/README.md](../space/README.md)

The rest of the package exists so the repo can still evaluate memory policies on
real benchmark surfaces.
