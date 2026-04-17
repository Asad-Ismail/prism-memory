# Release Scripts

These scripts exist only to build and confirm the public PRISM release surface.

## Files

| File | Purpose |
|---|---|
| `confirm_exp15_results.py` | replay the original Exp15 eval surface against the cached QA answers and write the public confirmation JSONs |
| `build_scenario_comparisons.py` | build the curated benchmark cases used by the demo from the shortlist JSON |
| `scenario_shortlist.json` | the curated shortlist of public demo cases |

## Outputs

These scripts write into [../../results/README.md](../../results/README.md):

- `results/confirmed_exp15_summary.json`
- `results/scenario_comparisons.json`
- `results/sft4.json`

## Notes

- These are not normal everyday benchmark runners.
- They depend on the original `better_memory` workspace via `BETTER_MEMORY_ROOT`.
- They are grouped here so the repo root stays clean and the release workflow is
  easy to find.
