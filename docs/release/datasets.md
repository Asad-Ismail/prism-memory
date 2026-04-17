[Back to Repo](../../README.md) · [Release Docs](README.md) · [Release Results](release-results.md)

# PRISM-Memory Datasets

This file separates the data used by the public `PRISM-Memory` release from the
auxiliary datasets that were only useful for ablations.

## Released Training Recipe

The released checkpoint is `exp15_sft_qwen7b_4ep`.

The core recipe was:

1. Start from `Qwen/Qwen2.5-7B-Instruct`.
2. Fine-tune with LoRA on a `20k` sample from `train_sft.jsonl`.
3. Evaluate on held-out `LoCoMo` and held-out `LongMemEval`.

## Source Conversations

The underlying synthetic conversation source lives in the upstream
`better_memory/data/output/` directory.

| File | Kind | Split | Notes |
|---|---|---|---|
| `train.jsonl` | raw conversations | train | `2,329` synthetic multi-session conversations |
| `eval.jsonl` | raw conversations | eval | `584` held-out synthetic multi-session conversations |
| `metadata.json` | split metadata | all | counts by tier, agent type, and update regime |

The source generator was built to create long-horizon memory stress cases with
inserts, updates, deletes, and multi-session recall.

## Derived SFT Data

These are GPT-4.1-derived proposition labels built on top of the raw
conversations.

| File | Examples | Role | Release Status |
|---|---|---|---|
| `train_sft.jsonl` | `100,427` | primary SFT data | core release data |
| `train_sft_clean_merged.jsonl` | `20,000` | cleaned resume base matching `sft4` distribution | good follow-on base |
| `train_sft_temporal_resolved.jsonl` | `2,643` | temporal-fix add-on set | useful for targeted research, not the public base |
| `eval_sft.jsonl` | reference | GPT-4.1 PropMem extractions on eval conversations | evaluation reference only |

## Evaluation Surfaces

The released model was evaluated on two held-out surfaces:

| Benchmark | Held-out Surface | Notes |
|---|---|---|
| `LoCoMo` | conversations `conv-49` and `conv-50` | five categories: factual, temporal, inferential, multi-hop, adversarial |
| `LongMemEval` | held-out items stratified by question type | six categories, including temporal reasoning and knowledge updates |

Both the GPT-4.1 extraction baseline and the released 7B extractor were scored
with the same GPT-4.1 QA evaluator and the same cache-backed answer surface.

## Auxiliary LoCoMo Datasets

These files were used in ablations and targeted probes. They matter for the
research story, but they are not the main public training recipe.

| File | Examples | Intended Use | Outcome |
|---|---|---|---|
| `locomo_qa_supervised_factual.jsonl` | `512` | factual QA supervision | neutral to small benefit |
| `locomo_qa_supervised_multihop.jsonl` | `625` | multihop QA supervision | neutral to small benefit |
| `locomo_qa_supervised_temporal.jsonl` | `248` | temporal QA supervision with absolute dates | neutral to small benefit |
| `locomo_qa_supervised_inferential.jsonl` | `133` | inferential QA supervision | too small, hurt balance |
| `locomo_qa_supervised_temporal_relformat.jsonl` | `248` | temporal QA with benchmark-style relative dates | hurt |
| `locomo_sft_extra.jsonl` | `2,645` | LoCoMo-domain SFT add-on | hurt |
| `locomo_sft_extra_relformat.jsonl` | `3,178` | relative-date LoCoMo SFT add-on | hurt |

## Practical Takeaways

1. The best 7B model came from the stable `20k` `train_sft.jsonl` base, not
   from aggressive benchmark-specific add-ons.
2. Training on LoCoMo-domain conversations did not help generalization.
3. Relative-date output hacks made the extractor worse.
4. More original LME data was not automatically better because noisy temporal
   labels compounded the anchor-loss problem.

Related docs:

- [extraction-skill.md](extraction-skill.md)
- [release-results.md](release-results.md)
- [technical-blog.md](technical-blog.md)
