# Tests

This directory contains the committed regression suite for the public repo.

## Coverage

| File | What it checks |
|---|---|
| `test_frontier_memory.py` | core synthetic-memory behavior and search logging |
| `test_travelplanner.py` | TravelPlanner policy behavior against the committed tiny fixture |
| `test_alfworld_benchmark.py` | ALFWorld planner logic and action-selection rules |
| `test_memoryarena_archive.py` | archive lookup helpers |
| `test_memoryarena_benchmark.py` | MemoryArena task-specific planner behavior |
| `test_memeval_adapter.py` | MemEval adapter normalization and override logic |

## Fixtures

The committed TravelPlanner fixture lives under `fixtures/` so the public repo
can run fast tests without downloading the full benchmark assets.

Run the full suite from the repo root:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```
