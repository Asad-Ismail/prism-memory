# PRISM-Memory Release Results

This file summarizes the confirmed release metrics and the internal comparison
artifacts that informed the public checkpoint choice.

## Released Checkpoint

- Checkpoint: `exp15_sft_qwen7b_4ep`
- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Adapter type: LoRA
- Confirmed LoCoMo mean: `0.4981204463`
- Confirmed LongMemEval mean: `0.4767574431`
- QA cache hits during confirmation: `460`
- QA cache misses during confirmation: `0`

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

## Internal Comparison That Informed The Release

The closest runner-up was `inferential_from_temporal_heavy`.

- Confirmed LoCoMo mean: `0.4975893989`
- Confirmed LongMemEval mean: `0.4688992148`
- Pairwise LoCoMo disagreements vs `sft4`: `152 / 400`
- Question-level wins: `56` for `sft4`, `52` for the runner-up

The release decision stayed with `sft4` because it preserved the strongest
LongMemEval score and the strongest adversarial behavior.

## Artifact Files

- `results/confirmed_exp15_summary.json`
- `results/scenario_comparisons.json`
- `results/locomo_pairwise_question_diffs.json`
- `results/sft4.json`

