# Examples

These examples make the release contract concrete without requiring the LoRA
weights or the full benchmark data.

## Files

- `sample_dialogue.txt`: a short multi-turn conversation with a dated update
- `sample_extraction.json`: the atomic propositions the extractor should write
- `sample_recall.md`: a small retrieval-style recall task over those facts

## Why These Examples Exist

The public release is deliberately narrow: one extraction skill, one benchmarked
checkpoint, one demo. These files help explain the system at that same level:

1. what kind of dialogue goes in
2. what proposition memory should come out
3. what a downstream recall question looks like

Use them for docs, quick reviews, or prompt regression checks before running the
full benchmark stack.
