# Scripts

This directory holds the runnable entrypoints for setup, release packaging, and
benchmark evaluation.

## Setup And Packaging

| Script | Purpose |
|---|---|
| [setup.sh](setup.sh) | install the repo and Space dependencies |
| [deploy_model.sh](deploy_model.sh) | create a clean Hugging Face model bundle for the released LoRA adapter and optionally upload it |
| [deploy_space.sh](deploy_space.sh) | create a clean Hugging Face Space bundle and optionally upload it |

## Release Helpers

The release-only helpers live under [release/README.md](release/README.md).

## Benchmark Entry Points

| Script | Purpose |
|---|---|
| `run_synthetic_eval.py` | local synthetic memory benchmark |
| `run_search_iteration.py` | mutate and score candidate children |
| `run_travelplanner_eval.py` | TravelPlanner evaluation with a candidate policy |
| `run_travelplanner_search.py` | search loop around TravelPlanner policies |
| `run_alfworld_eval.py` | ALFWorld evaluation with the procedural-memory policy |
| `run_memoryarena_eval.py` | MemoryArena point eval |
| `run_memoryarena_suite.py` | MemoryArena public suite |
| `run_memeval_eval.py` | MemEval adapter over LoCoMo or LongMemEval |
| `run_openai_synthetic_eval.py` | provider-backed synthetic benchmark |
| `tune_openai_memory.py` | provider-backed tuning loop |
| `summarize_offline_frontier.py` | summarize promoted external-benchmark results |
| `score_candidate.py` | lightweight candidate score helper |

## Important Notes

- `TravelPlanner` full evaluation expects `PRISM_TRAVELPLANNER_ROOT` to point to
  a local checkout. The public repo only ships a tiny fixture.
- `MemEval` evaluation expects `MEMEVAL_ROOT` or a compatible local checkout.
- The Space bundle script reads the release docs and JSON artifacts directly
  from this repo, so stale paths here usually break demo publishing.
- The model bundle script reads the release adapter from `BETTER_MEMORY_ROOT`
  unless `PRISM_CHECKPOINT_DIR` is set explicitly.
