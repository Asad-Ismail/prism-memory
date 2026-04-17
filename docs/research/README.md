# Research Docs

These docs describe the broader `frontier_memory` research harness that lives
alongside the public `PRISM-Memory` release.

## Core Files

| File | Purpose |
|---|---|
| [program.md](program.md) | the operating note for the search loop |
| [../../contract/README.md](../../contract/README.md) | scoring contracts, benchmark tiers, and candidate templates |
| [../../logs/README.md](../../logs/README.md) | append-only journal and experiment JSONL rules |
| [../../candidates/README.md](../../candidates/README.md) | promoted policy files and what each one targets |
| [../../scripts/README.md](../../scripts/README.md) | runnable benchmark entrypoints for the harness |

## What This Layer Is For

This layer exists to:

1. search for better hybrid-memory policies
2. compare procedural, episodic, and semantic memory designs
3. keep benchmark and promotion logic reviewable
4. preserve experiment history without mixing it into the public release story

If you are only trying to understand the released PRISM extractor, start in
[../release/README.md](../release/README.md) instead.
