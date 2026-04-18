[Back to Repo](../../README.md) · [Release Docs](README.md) · [Result Artifacts](../../results/README.md)

# PRISM-Memory Release Results

This page summarizes the confirmed public release metrics and the internal
comparison evidence that informed the release choice.

## Released Model

- Model: `PRISM-Memory 7B Adapter`
- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Adapter type: LoRA
- Confirmed LoCoMo mean: `0.4981204463`
- Confirmed LongMemEval mean: `0.4767574431`
- QA cache hits during confirmation: `460`
- QA cache misses during confirmation: `0`

## Public Comparison

PRISM-Memory fine-tunes `Qwen/Qwen2.5-7B-Instruct` for the memory extraction
step that the PropMem reference gets from GPT-4.1.

| Benchmark | PRISM-Memory | GPT-4.1-based PropMem reference | Read |
|---|---:|---:|---|
| LongMemEval | `0.4768` | `0.4650` | PRISM wins |
| LoCoMo | `0.4981` | `0.5360` | PRISM trails, but stays competitive |

The QA layer is held constant. This is an extraction-step comparison, not an
end-to-end GPT-4.1 replacement claim.

## LoCoMo Breakdown

| Category | Score |
|---|---:|
| factual | `0.3339551926` |
| temporal | `0.4978785870` |
| inferential | `0.2605997475` |
| multi-hop | `0.5144477744` |
| adversarial | `0.8837209302` |

## LongMemEval Breakdown

| Category | Score |
|---|---:|
| knowledge-update | `0.5588405797` |
| multi-session | `0.1390977444` |
| single-session-assistant | `0.7656395892` |
| single-session-preference | `0.0519667456` |
| single-session-user | `0.9133333333` |
| temporal-reasoning | `0.4316666667` |

## Why This Model Was Released

The closest internal runner-up nearly tied the released model on overall
LoCoMo, but it lost on the broader release profile:

- lower LongMemEval score: `0.4689`
- weaker adversarial precision
- less balanced behavior across the full evaluation surface

Question-level comparison on held-out LoCoMo:

- disagreements: `152 / 400`
- questions favoring PRISM-Memory: `56`
- questions favoring the runner-up: `52`

That is close enough to be a real internal comparison, but not close enough to
justify two public models.

## Artifact Files

- [../../results/release_summary.json](../../results/release_summary.json)
- [../../results/release_model.json](../../results/release_model.json)
- [../../results/benchmark_cases.json](../../results/benchmark_cases.json)
- [../../results/internal_locomo_pairwise_diffs.json](../../results/internal_locomo_pairwise_diffs.json)

Related docs:

- [extraction-skill.md](extraction-skill.md)
- [extraction-examples.md](extraction-examples.md)
- [datasets.md](datasets.md)
- [model-card.md](model-card.md)
