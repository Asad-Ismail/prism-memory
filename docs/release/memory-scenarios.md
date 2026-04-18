[Back to Repo](../../README.md) · [Release Docs](README.md) · [Extraction Examples](extraction-examples.md)

# PRISM-Memory End-To-End Scenarios

These are compact product-style scenarios built from the public release
artifacts.

- The first two use the released held-out extraction examples.
- The last two use confirmed held-out benchmark cases from
  [../../results/benchmark_cases.json](../../results/benchmark_cases.json).

The point is not just that the extractor matches GPT-4.1-style labels. The
point is that a later system can ask a concrete question and get back a useful,
inspectable answer from stored memory.

## 1. Keep hard limits and notification preferences

**Conversation turn**

> yeah, I think starting with incremental scans and parallel matrix jobs makes sense. We have 20 concurrent jobs max on GitHub Actions currently. Also want to keep Slack notifications from Snyk consistent with other pipeline alerts, aggregated and concise.

**Stored memory**

- GitHub Actions concurrency limit: 20 concurrent jobs
- Snyk Slack notifications should be aggregated and concise

**Later question**

What is our GitHub Actions concurrency limit, and how should Snyk alerts look?

**Answer from memory**

20 concurrent jobs. Snyk alerts should be aggregated and concise.

**Why it matters**

This is the kind of operational detail that gets buried in chat but needs to
survive into later workflow drafts and agent actions.

## 2. Keep current state separate from the roadmap

**Conversation turn**

> yeah good point about resource overhead, we set CPU limits for all sidecars and monitor with Prometheus now. no mTLS yet, but it’s on the roadmap for phase two. as for routing, we want to start with canary deployments and traffic splitting, maybe some basic fault injection for testing.

**Stored memory**

- Sidecar CPU limits set and monitored via Prometheus
- Istio mTLS planned for phase two
- Routing strategy: canary deployments and traffic splitting; basic fault injection planned

**Later question**

Did we already enable mTLS, and what rollout strategy are we planning?

**Answer from memory**

mTLS is not enabled yet; it is planned for phase two. The rollout plan is
canary deployments and traffic splitting, with basic fault injection planned.

**Why it matters**

Memory systems often blur the current state with the planned state. This is the
kind of distinction that matters in deployment and incident work.

## 3. Answer dated questions instead of only remembering themes

**Question**

Which hobby did Sam take up in May 2023?

**Retrieved memory**

- Sam: [18 May 2023] Sam is considering trying painting as a new hobby.
- Sam: [24 May 2023] Sam has been considering trying painting as a new hobby.

**Answer from memory**

painting

**Why it matters**

A useful memory system should not just remember that someone talked about
hobbies. It should recover the dated fact that actually answers the later
question.

## 4. Refuse unsupported claims instead of inventing a reason

**Question**

Why did Dave get his guitar customized with a shiny finish?

**Retrieved memory**

- Dave: That guitar has a gorgeous purple hue. Why did you make it so shiny?
- Good pick! The customized purple glow gives it a unique look that really stands out.
- Dave: The guitar was in bad condition when Dave found it.

**Answer from memory**

None / unsupported

**Why it matters**

Memory systems are more useful when they can refuse cleanly. Here the retrieved
context talks about the guitar and the finish, but it never actually supports
the premise that Dave customized it for a specific reason.
