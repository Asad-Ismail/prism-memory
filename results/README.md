# Result Artifacts

This directory contains the tracked JSON artifacts that support the public
release story.

## Public Artifacts

| File | Purpose |
|---|---|
| `confirmed_exp15_summary.json` | confirmed release summary for `exp15_sft_qwen7b_4ep` |
| `scenario_comparisons.json` | curated benchmark cases used by the demo |
| `sft4.json` | single-checkpoint confirmation payload |
| `frontier_memory_benchmarks.json` | tracked summary of promoted external-benchmark results |

## Internal Comparison Artifact

| File | Purpose |
|---|---|
| `locomo_pairwise_question_diffs.json` | question-level diff between the release checkpoint and the closest runner-up |

See also:

- [../docs/release/release-results.md](../docs/release/release-results.md)
- [../space/README.md](../space/README.md)
- [../scripts/release/README.md](../scripts/release/README.md)

Large experimental logs and raw dumps belong under `logs/`, not here.
