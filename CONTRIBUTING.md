# Contributing

Thanks for contributing to `PRISM-Memory`.

This repository has two layers:

1. the public `PRISM-Memory` release surface
2. the broader `frontier_memory` research harness

Changes should preserve clarity between those two layers.

See the navigation docs for the current public layout:

- `docs/README.md`
- `docs/release/README.md`
- `docs/research/README.md`

## Setup

Install the base package:

```bash
python -m pip install -e .
```

Install the development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

## Common Commands

Run the test suite:

```bash
make test
```

Run the demo locally:

```bash
make demo
```

Run the synthetic benchmark smoke test:

```bash
make synthetic-eval
```

## Optional Local Research Dependencies

Some scripts depend on local assets that are intentionally not part of the
public GitHub payload.

- `BETTER_MEMORY_ROOT`: path to the original `better_memory` workspace
- `MEMEVAL_ROOT`: path to the local MemEval checkout
- `PRISM_LOCOMO_PATH`: path to `locomo10.json`
- `PRISM_TRAVELPLANNER_VALIDATION_PATH`: path to a TravelPlanner validation
  fixture or dataset export

The tests already use a small committed TravelPlanner fixture, so they do not
need network access.

## Repo Hygiene

Do not commit:

- generated candidate variants under `candidates/generated/`
- local benchmark dumps under `logs/runs/`
- local smoke outputs like `logs/*_smoke.jsonl`
- oversized third-party benchmark assets ignored by `.gitignore`

If you add new benchmark adapters or external assets, update:

- `.gitignore`
- `external/README.md`
- `scripts/README.md`
- `README.md`

## Documentation Expectations

If you change the public release story, update the relevant docs together:

- `README.md`
- `docs/release/README.md`
- `docs/release/datasets.md`
- `docs/release/release-results.md`
- `docs/release/extraction-skill.md`
- `docs/release/technical-blog.md`
- `docs/release/model-card.md`

## Testing Expectations

Before opening a PR, run:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m compileall frontier_memory scripts space tests
```

## Style

- Prefer small, reviewable changes.
- Keep public docs concrete and benchmark-anchored.
- Keep release claims narrower than the evidence.
- Avoid hardcoded local paths unless they are guarded by env vars or clear local-only fallbacks.
