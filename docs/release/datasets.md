[Back to Repo](../../README.md) · [Release Docs](README.md) · [Release Results](release-results.md)

# PRISM-Memory Training Data

The PRISM-Memory release is trained on **synthetic** multi-session
conversations with **GPT-4.1-derived** memory-writing labels. No real user chat
logs are part of the public release story.

## Dataset At A Glance

| Item | Count | What it means |
|---|---:|---|
| synthetic training conversations | `2,329` | multi-session conversations used to build the training label bank |
| synthetic held-out conversations | `584` | held-out conversations used for evaluation examples and reference labels |
| total generated conversations | `2,913` | train plus eval |
| supervised extraction examples | `100,427` | memory-writing examples derived from the synthetic conversations |
| released training subset | `20,000` | supervised examples used to train the public adapter |
| agent and task families | `6` | research, data analysis, QA, coding, planning, writing |

The synthetic conversation generator deliberately creates long-horizon memory
pressure:

- facts introduced early and queried later
- updated plans and corrected details
- deleted or invalidated information
- multi-session continuity
- mixtures of preferences, project state, dates, and operational facts

## How The Data Is Built

The training pipeline has two layers.

### 1. Synthetic conversation generation

The first layer creates multi-session conversations around realistic work and
assistant scenarios. Each conversation comes with scenario metadata, a persona,
multiple sessions, and explicit memory events such as inserts, updates, and
deletes.

Across the full corpus:

- `899` conversations are short
- `1,162` are medium
- `852` are long
- `897` are insert-only
- `937` include updates
- `435` include both updates and deletes

### 2. Supervised memory-writing labels

The second layer converts those conversations into supervised extraction
examples. Each example contains:

- retrieved memories seen so far
- recent conversation context
- the current user turn
- target memory operations that should be written from that turn

The released model learns this memory-writing step.

## What A Training Example Looks Like

One real synthetic scenario in the corpus is about **cloud infrastructure
performance optimization** for a low-latency trading platform.

**Synthetic scenario**

- domain: cloud infrastructure performance optimization
- persona: senior cloud systems engineer at a fintech startup
- conversation shape: two sessions, ten chunks, five later questions

**Synthetic user turn**

> Here’s the initial architecture outline: deploy microservices on AWS Fargate, use PostgreSQL 13 as the primary database, plan Kubernetes orchestration, use Redis for caching, keep API latency under 50ms, and redesign the system with a team of five engineers.

**Target memory records**

- Deploy microservices on AWS Fargate
- Orchestrate containers on a Kubernetes cluster (planned)
- Primary database: PostgreSQL 13
- Use Redis as an in-memory caching layer
- Latency target: API responses under 50ms

Later turns in the same conversation update that memory with new load targets,
TTL settings, and rollout constraints such as zero downtime.

## What Trained The Released Model

The public adapter was trained on `20,000` supervised extraction examples
sampled from the larger `100,427`-example label bank.

In plain terms, the model saw many examples of this pattern:

1. a conversation turn mentions several durable facts
2. the target output keeps only the memory-worthy facts
3. those facts are written as short standalone memory records

That is why the release behaves like a memory writer rather than a chat model.

## Evaluation Surfaces

The released model is evaluated on two held-out surfaces.

| Benchmark | Held-out surface | What it tests |
|---|---|---|
| `LoCoMo` | held-out conversations `conv-49` and `conv-50` | factual, temporal, inferential, multi-hop, and adversarial recall |
| `LongMemEval` | held-out items across six categories | knowledge updates, multi-session recall, single-session recall, and temporal reasoning |

Both the PRISM extractor and the GPT-4.1-based PropMem reference are scored
with the same QA layer, so the public comparison isolates the extraction step.

## What Is Public Today

Public now:

- the dataset design
- corpus counts
- example training records
- held-out extraction examples
- benchmark results and category breakdowns

Not public yet:

- the full raw synthetic conversation files
- the full supervised label bank
- the auxiliary ablation corpora used for follow-on experiments

## Practical Lessons From The Data

1. The strongest release model came from the stable `20,000`-example base, not
   from benchmark-specific add-ons.
2. Explicit date anchoring helped more than benchmark-style answer formatting.
3. More narrow benchmark data did not automatically improve generalization.
4. The supervision is most useful when it teaches durable facts, updates, and
   contradictions instead of stylistic imitation.

Related docs:

- [extraction-skill.md](extraction-skill.md)
- [release-results.md](release-results.md)
- [technical-blog.md](technical-blog.md)
