# Logs

This directory is append-only and belongs to the broader `frontier_memory`
research harness, not just the public `PRISM-Memory` release.

## Files

- `journal.md`: human-readable reasoning, pivots, failures, and next actions
- `experiments.jsonl`: one JSON object per experiment or meta-event
- `runs/`: local run outputs and benchmark dumps, usually kept out of the first
  public GitHub commit

## Rules

1. Never rewrite history.
2. If an old entry is wrong, add a correction entry.
3. Every run gets both a journal entry and a JSONL entry.
4. Log the hypothesis, mutation, benchmark tier, result, and decision.

## Minimal JSONL Shape

```json
{
  "date": "2026-04-16",
  "event": "experiment",
  "candidate_id": "bootstrap_v0",
  "parent_ids": [],
  "hypothesis": "procedural-first router improves ALFWorld proxy",
  "mutation": ["retrieval.router", "retrieval.top_k.procedural"],
  "benchmarks": {
    "alfworld_proxy": 0.0
  },
  "decision": "promote",
  "notes": "short free-text note"
}
```
