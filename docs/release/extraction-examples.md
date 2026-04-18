[Back to Repo](../../README.md) · [Release Docs](README.md) · [Result Artifacts](../../results/README.md)

# PRISM-Memory Extraction Examples

Selected held-out examples from the synthetic evaluation split.
The `GPT-4.1 reference` rows come from the supervised target memory labels.
The `PRISM-Memory 7B Adapter` rows were regenerated with greedy decoding using the same extraction prompt family used during evaluation.

These examples are illustrations, not the benchmark itself. Use
[release-results.md](release-results.md) for the aggregate numbers.

## Infrastructure bottlenecks stay structured

- Overlap score: `0.909`
- Session date: `2025-01-04 15:34:00`
- Note: Near-exact match on two operational facts from a single held-out turn.

**Turn**

> yeah, no real caching beyond basic Docker layer caching. Jenkins nodes have limited capacity, and we sometimes hit queue delays during peak commits.

**GPT-4.1 reference**

- No caching beyond basic Docker layer caching
- Jenkins nodes have limited capacity and experience queue delays during peak commits

**PRISM-Memory**

- No Docker caching beyond basic layer caching
- Jenkins nodes have limited capacity; peak commits cause queue delays

## Numeric constraints and preferences survive extraction

- Overlap score: `0.800`
- Session date: `2025-03-01 15:07:00`
- Note: The trained model keeps both the hard concurrency cap and the desired notification style.

**Turn**

> yeah, I think starting with incremental scans and parallel matrix jobs makes sense. We have 20 concurrent jobs max on GitHub Actions currently. Also want to keep Slack notifications from Snyk consistent with other pipeline alerts—aggregated and concise. Can you help draft the workflow?

**GPT-4.1 reference**

- GitHub Actions concurrency limit: 20 concurrent jobs
- Wants Snyk Slack notifications aggregated and concise, consistent with other pipeline alerts

**PRISM-Memory**

- GitHub Actions concurrency limit: 20 concurrent jobs
- Snyk Slack notifications should be aggregated and concise

## Current state and next-step plans stay grouped

- Overlap score: `0.799`
- Session date: `2025-09-09 15:26:00`
- Note: The extracted memory keeps the current configuration, the planned security step, and the rollout strategy in one compact bundle.

**Turn**

> yeah good point about resource overhead, we set CPU limits for all sidecars and monitor with Prometheus now. no mTLS yet, but it’s on the roadmap for phase two. as for routing, we want to start with canary deployments and traffic splitting, maybe some basic fault injection for testing.

**GPT-4.1 reference**

- Istio sidecar CPU limits set and monitored via Prometheus
- mTLS planned in phase two
- Plan to use canary deployments, traffic splitting, and basic fault injection

**PRISM-Memory**

- Sidecar CPU limits set and monitored via Prometheus
- Istio mTLS planned for phase two
- Routing strategy: canary deployments and traffic splitting; basic fault injection planned

## Regeneration

```bash
conda run -n pytorch_p310 python scripts/release/generate_extraction_examples.py
```

