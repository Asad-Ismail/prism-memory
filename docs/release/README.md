# Release Docs

These are the documents that define the public `PRISM-Memory` release.

## Read These First

| File | Why it matters |
|---|---|
| [extraction-skill.md](extraction-skill.md) | the canonical extraction contract and the best checkpoint to pair with it |
| [extraction-examples.md](extraction-examples.md) | selected held-out GPT-4.1-vs-PRISM comparisons used on the main README |
| [release-results.md](release-results.md) | the confirmed metrics and the comparison logic behind the checkpoint choice |
| [memory-scenarios.md](memory-scenarios.md) | short end-to-end scenarios showing why the extractor is useful after memory is written |
| [datasets.md](datasets.md) | what data the release actually used and what did not help |
| [technical-blog.md](technical-blog.md) | the longer explanation of what worked, what regressed, and why |
| [model-card.md](model-card.md) | the source model card used for the public Hugging Face adapter release |

## Companion Repo Surfaces

| Path | Why you may want it |
|---|---|
| [../../results/README.md](../../results/README.md) | the JSON artifacts referenced by these docs |
| [../../space/README.md](../../space/README.md) | the public demo and Space bundle |
| [../../examples/README.md](../../examples/README.md) | small toy cases for explaining the extraction behavior |
| [../../scripts/release/README.md](../../scripts/release/README.md) | confirmation and scenario-building helpers |

## Scope

This release surface is deliberately narrower than the full repo:

- one extraction skill
- one checkpoint
- one public demo
- one benchmarked release story

The broader search harness and benchmark champions are still available elsewhere
in the repo, but they are not the primary public identity.
