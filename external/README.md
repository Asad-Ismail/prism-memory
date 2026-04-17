# External Benchmarks

This directory is reserved for optional local benchmark checkouts used by the
broader `frontier_memory` research harness.

## Typical Local Checkouts

| Project | Why you might need it |
|---|---|
| `TravelPlanner` | full official travel-planning evaluation |
| `alfworld` | full ALFWorld environment and task data |
| `MemP` | procedural-memory comparison code |
| `agent-workflow-memory` | workflow-memory comparison code |
| `evolving-memory` | consolidation-oriented comparison code |

## Important Rule

The GitHub-facing repo does not vendor these full checkouts. They stay local and
are ignored by `.gitignore`.

## Key Env Hooks

- `PRISM_TRAVELPLANNER_ROOT=/path/to/TravelPlanner`
- `MEMEVAL_ROOT=/path/to/MemEval`
- `BETTER_MEMORY_ROOT=/path/to/better_memory`

For TravelPlanner specifically:

- the public repo ships a tiny committed fixture under `tests/fixtures/travelplanner_root/`
- full official evaluation still expects a separate local TravelPlanner checkout

## Related Docs

- [../scripts/README.md](../scripts/README.md)
- [../tests/README.md](../tests/README.md)
- [../docs/research/README.md](../docs/research/README.md)

## Licensing

Each local checkout keeps its own upstream license file. The root repository
license applies to the code added in this repo, while third-party code used
locally remains governed by its original license.
