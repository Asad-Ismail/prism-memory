# Result Artifacts

This directory contains the tracked JSON artifacts that support the public
release story.

## Public Artifacts

| File | Purpose |
|---|---|
| `release_summary.json` | confirmed summary for the released PRISM-Memory adapter |
| `release_model.json` | single-model confirmation payload for the released adapter |
| `extraction_examples.json` | selected held-out GPT-4.1-vs-PRISM extraction comparisons used by the main README and Space |
| `benchmark_cases.json` | curated benchmark cases used by the demo |
| `frontier_memory_benchmarks.json` | tracked summary of promoted external-benchmark results |

## Internal Comparison Artifact

| File | Purpose |
|---|---|
| `internal_locomo_pairwise_diffs.json` | question-level diff between the release model and the closest runner-up |

See also:

- [../docs/release/release-results.md](../docs/release/release-results.md)
- [../space/README.md](../space/README.md)
- [../scripts/release/README.md](../scripts/release/README.md)

Large experimental logs and raw dumps belong under `logs/`, not here.
