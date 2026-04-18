[Back to Repo](../../README.md) · [Release Docs](README.md) · [Extraction Examples](extraction-examples.md)

# PRISM-Memory End-To-End Scenarios

These are compact product-style scenarios built from the public release
artifacts.

- The first three come from the bundled interactive session artifact
  [../../results/try_it_sessions.json](../../results/try_it_sessions.json).
- The last one comes from the held-out extraction examples in
  [../../results/extraction_examples.json](../../results/extraction_examples.json).

The point is not just that the extractor matches GPT-4.1-style labels. The
point is that a later system can ask a concrete question and get back a useful,
inspectable answer from stored memory.

## 1. Keep hard limits and notification preferences

**Conversation turns**

> [2025-03-01] Dana: We have 20 concurrent jobs max on GitHub Actions right now. Keep Snyk Slack notifications aggregated and concise, not one alert per repo.  
> [2025-03-07] Dana: No mTLS yet. Put it in phase two after the canary rollout.

**Stored memory**

- GitHub Actions concurrency limit: 20 concurrent jobs
- Snyk Slack notifications aggregated and concise; no separate alerts per repo
- No mutual TLS enabled; mutual TLS will be implemented in phase two after the canary rollout

**Later question**

What is the current concurrency limit, how should Slack alerts behave, and when
is mTLS planned?

**Answer from memory**

The current GitHub Actions concurrency limit is 20 jobs. Slack notifications
should be aggregated and concise. mTLS is planned for phase two after the
canary rollout.

**Why it matters**

This is the kind of operational detail that gets buried in chat but needs to
survive into later workflow drafts and agent actions.

## 2. Replace stale travel plans with the current one

**Conversation turns**

> [2025-06-12] Maya: We booked the Lisbon trip for September 14. I want a quiet hotel near Alfama, and no red-eye flights.  
> [2025-07-02] Maya: Update the plan: Lisbon is off. We are going to Porto on September 21 instead, still no red-eye flights.

**Stored memory**

- Lisbon trip booked for September 14
- Prefers a quiet hotel near Alfama
- Prefers no red-eye flights
- Lisbon trip canceled; Porto trip scheduled for September 21
- No red-eye flights for Porto trip

**Later question**

Where is the trip now, on what date, and what flight constraint still applies?

**Answer from memory**

The current trip is to Porto on September 21, 2025, and red-eye flights are
still off limits.

**Why it matters**

Memory systems often keep both the old and new plan, but the downstream system
still has to recover the latest valid state.

## 3. Keep routines and health constraints available later

**Conversation turns**

> [2025-08-03] Sam: My doctor wants me to keep sodium under 2 grams a day. I started painting on weekends because it helps me decompress.  
> [2025-08-17] Sam: I bought watercolors and signed up for a Saturday class downtown.  
> [2025-09-01] Sam: Skip late-night coffee from now on; it wrecks my sleep.

**Stored memory**

- Doctor recommended Sam limit sodium intake to 2 grams per day
- Sam paints on weekends as a decompression activity
- Sam bought watercolors and signed up for a Saturday painting class downtown
- Sam decided to skip late-night coffee starting now

**Later question**

What hobby did Sam start, what class did he sign up for, and what health
constraints matter now?

**Answer from memory**

Sam started painting, signed up for a Saturday watercolor class downtown,
should keep sodium under 2 grams per day, and wants to avoid late-night coffee.

**Why it matters**

This is the difference between transcript search and usable memory. The durable
facts can drive reminders, planning, or coaching later.

## 4. Keep infrastructure bottlenecks structured

**Conversation turn**

> yeah, no real caching beyond basic Docker layer caching. Jenkins nodes have limited capacity, and we sometimes hit queue delays during peak commits.

**Stored memory**

- No Docker caching beyond basic layer caching
- Jenkins nodes have limited capacity; peak commits cause queue delays

**Later question**

What is our current caching setup, and why do builds sometimes queue up?

**Answer from memory**

There is no special caching beyond basic Docker layer caching. Builds queue up
because Jenkins nodes have limited capacity during peak commits.

**Why it matters**

The stored memory stays compact and directly useful. A later system does not
need to reread the full conversation turn to answer the operational question.
