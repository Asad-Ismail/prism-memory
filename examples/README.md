# Examples

These examples make the release behavior concrete without requiring the LoRA
weights or the full benchmark data.

## Files

| File | Purpose |
|---|---|
| `sample_dialogue.txt` | a short conversation with a dated update |
| `sample_extraction.json` | the atomic facts the extractor should write |
| `sample_recall.md` | a retrieval-style recall question over those facts |

## How To Use Them

Use these files when you want to:

1. explain the extraction contract quickly
2. review prompt behavior on a stable toy case
3. sanity-check downstream recall logic before running a full benchmark

Related docs:

- [../docs/release/extraction-skill.md](../docs/release/extraction-skill.md)
- [../docs/release/technical-blog.md](../docs/release/technical-blog.md)
- [../space/README.md](../space/README.md)
