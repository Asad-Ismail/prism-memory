# Release Docs

These documents define the public `PRISM-Memory` release.

## Read These First

| File | Why it matters |
|---|---|
| [datasets.md](datasets.md) | explains the synthetic corpus, the supervision format, and what is public today |
| [extraction-examples.md](extraction-examples.md) | selected held-out GPT-4.1-vs-PRISM comparisons used on the main README and the Space |
| [memory-scenarios.md](memory-scenarios.md) | short end-to-end scenarios showing why the stored memory is useful later |
| [extraction-skill.md](extraction-skill.md) | the canonical extraction contract and retrieval setup for the released model |
| [release-results.md](release-results.md) | the confirmed metrics and the reasoning behind the release choice |
| [technical-blog.md](technical-blog.md) | the longer write-up of what worked, what failed, and what the repo learned |
| [model-card.md](model-card.md) | the source model card for the public Hugging Face adapter release |

## Companion Repo Surfaces

| Path | Why you may want it |
|---|---|
| [../../results/README.md](../../results/README.md) | the JSON artifacts referenced by these docs |
| [../../space/README.md](../../space/README.md) | the public demo and Space bundle |
| [../../examples/README.md](../../examples/README.md) | small toy cases for explaining the extraction behavior |
| [../../scripts/release/README.md](../../scripts/release/README.md) | the helper scripts that regenerate the release artifacts |

## Scope

This release surface is deliberately narrower than the full repo:

- one released model
- one extraction skill
- one public demo
- one benchmarked release story

The broader search harness and other benchmark work still exist elsewhere in
the repo, but they are not the primary public identity.
