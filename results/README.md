# Result Artifacts

This directory contains the release-facing evaluation outputs for
`PRISM-Memory`.

## Public Artifacts

- `confirmed_exp15_summary.json`: confirmed summary for the released checkpoint
- `scenario_comparisons.json`: curated benchmark cases for the demo
- `sft4.json`: single-checkpoint confirmation payload
- `frontier_memory_benchmarks.json`: tracked summary of the promoted external
  benchmark results used by the research harness

## Internal Comparison Artifact

- `locomo_pairwise_question_diffs.json`: question-level comparison between the
  release checkpoint and the closest internal runner-up

This directory is intended to stay small and readable. Large experimental logs
and raw benchmark dumps belong under `logs/`.
