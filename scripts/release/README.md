# Release Scripts

These scripts exist only to build and confirm the public PRISM release surface.

## Files

| File | Purpose |
|---|---|
| `confirm_release_results.py` | replay the original release eval surface against the cached QA answers and write the public confirmation JSONs |
| `generate_extraction_examples.py` | regenerate the held-out GPT-4.1-vs-PRISM extraction examples used by the main README and release docs |
| `generate_try_it_sessions.py` | generate bundled multi-turn example sessions with released-model outputs for the Space |
| `extraction_example_shortlist.json` | stable shortlist for the held-out extraction examples |
| `try_it_session_specs.json` | stable session specs for the interactive Space examples |

## Outputs

These scripts write into [../../results/README.md](../../results/README.md):

- `results/release_summary.json`
- `results/release_model.json`
- `results/extraction_examples.json`
- `results/try_it_sessions.json`

They also update:

- `docs/release/extraction-examples.md`

## Notes

- These are not normal everyday benchmark runners.
- They depend on the original `better_memory` workspace via `BETTER_MEMORY_ROOT`.
- They are grouped here so the repo root stays clean and the release workflow is
  easy to find.
