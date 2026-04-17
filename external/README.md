# External Benchmarks

This directory is reserved for optional local benchmark checkouts used by the
broader `frontier_memory` research harness.

## Typical Local Checkouts

- `TravelPlanner`
- `alfworld`
- `MemP`
- `agent-workflow-memory`
- `evolving-memory`

## GitHub Release Notes

The GitHub-facing repo does not vendor these checkouts. They stay local and are
ignored by `.gitignore`.

For TravelPlanner specifically:

- the public repo ships a tiny committed fixture under `tests/fixtures/travelplanner_root/`
- full official evaluation still expects a separate local TravelPlanner checkout
- point the harness at that checkout with `PRISM_TRAVELPLANNER_ROOT=/path/to/TravelPlanner`

## Licensing

Each local checkout keeps its own upstream license file. The root repository
license applies to the code added in this repo, while third-party code used
locally remains governed by its original license.
