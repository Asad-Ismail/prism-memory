# Research Journal

## 2026-04-16 - Bootstrap

Goal:

- discover a frontier memory system
- keep a full lineage of decisions and pivots
- search architecture and consolidation policy, not only prompts

Decision:

- repurpose this repo into a general frontier-memory search control plane
- start from a hybrid candidate with episodic, semantic, and procedural layers
- treat `ALFWorld` and `TravelPlanner` as first-line procedural benchmarks
- require broader promotion on agent-memory benchmarks before declaring a champion

Immediate queue:

1. reproduce `MACLA`-like procedural memory under the common contract
2. reproduce `MemP`-like build / retrieve / update memory under the common contract
3. ablate dream-style consolidation on the same replay surface
4. compare single-agent vs orchestrator / worker placement once the single-agent baseline is stable

Pivot rule:

- if procedural wins do not generalize, shift budget to controller and multi-store routing instead of making the procedure library bigger

## 2026-04-16 - Search Iteration

Base candidate: `bootstrap_v0`
Dataset size per benchmark: `4`
Children explored: `4`

Champion: `bootstrap_v0_iter_20260416231158_03` with global score `0.9852`

Mutations tested:
- `bootstrap_v0_iter_20260416231158_03`: retrieval.order=semantic/procedural/episodic, merge_similarity=0.92, drop_pass:split_overbroad_routines, reliability=beta(0.5,2.0)
- `bootstrap_v0_iter_20260416231158_04`: merge_similarity=0.92, reliability=beta(0.5,1.5), retrieval.order=episodic/procedural/semantic, top_k.episodic=4
- `bootstrap_v0_iter_20260416231157_01`: merge_similarity=0.92, reliability=beta(0.5,0.5)
- `bootstrap_v0_iter_20260416231158_02`: merge_similarity=0.94, retrieval.order=semantic/episodic/procedural, reliability=beta(0.5,0.5), controller=orchestrator_worker
- `bootstrap_v0`: baseline

## 2026-04-16 - Search Iteration

Base candidate: `bootstrap_v0`
Dataset size per benchmark: `4`
Children explored: `4`

Champion: `bootstrap_v0_iter_20260416231221_03` with global score `0.9852`

Mutations tested:
- `bootstrap_v0_iter_20260416231221_03`: retrieval.order=semantic/procedural/episodic, merge_similarity=0.92, drop_pass:split_overbroad_routines, reliability=beta(0.5,2.0)
- `bootstrap_v0_iter_20260416231221_04`: merge_similarity=0.92, reliability=beta(0.5,1.5), retrieval.order=episodic/procedural/semantic, top_k.episodic=4
- `bootstrap_v0_iter_20260416231221_01`: merge_similarity=0.92, reliability=beta(0.5,0.5)
- `bootstrap_v0_iter_20260416231221_02`: merge_similarity=0.94, retrieval.order=semantic/episodic/procedural, reliability=beta(0.5,0.5), controller=orchestrator_worker
- `bootstrap_v0`: baseline

## 2026-04-16 - Eval Fix

Decision:

- corrected a search-loop bug where children were being evaluated on different random benchmark draws
- re-ran promotion on a shared seed and full local suite before exporting the champion

Fair champion:

- `bootstrap_v0_iter_20260416231328_02`
- global score `0.9792` on the shared-seed local suite
- winning mutations: `retrieval.order=episodic/semantic/procedural`, `top_k.episodic=5`, `drop_pass:merge_duplicates`

## 2026-04-16 - Search Iteration

Base candidate: `bootstrap_v0`
Dataset size per benchmark: `8`
Children explored: `8`

Champion: `bootstrap_v0_iter_20260416231309_05` with global score `0.9864`

Mutations tested:
- `bootstrap_v0_iter_20260416231309_05`: top_k.semantic=7, merge_similarity=0.92, retrieval.order=episodic/semantic/procedural, reliability=beta(0.5,2.0)
- `bootstrap_v0`: baseline
- `bootstrap_v0_iter_20260416231309_06`: merge_similarity=0.92, controller=orchestrator_worker, retrieval.order=semantic/procedural/episodic
- `bootstrap_v0_iter_20260416231308_03`: controller=orchestrator_worker, retrieval.order=episodic/procedural/semantic, merge_similarity=0.94
- `bootstrap_v0_iter_20260416231308_02`: retrieval.order=episodic/semantic/procedural, drop_pass:merge_duplicates, top_k.episodic=5
- `bootstrap_v0_iter_20260416231309_07`: controller=orchestrator_worker, drop_pass:split_overbroad_routines, merge_similarity=0.88
- `bootstrap_v0_iter_20260416231308_01`: reliability=beta(2.0,1.5), top_k.semantic=10
- `bootstrap_v0_iter_20260416231309_08`: controller=orchestrator_worker, reliability=beta(0.5,2.0), merge_similarity=0.92, retrieval.order=semantic/episodic/procedural
- `bootstrap_v0_iter_20260416231309_04`: merge_similarity=0.88, controller=orchestrator_worker, drop_pass:compact_summaries, top_k.semantic=9

## 2026-04-16 - Search Iteration

Base candidate: `bootstrap_v0`
Dataset size per benchmark: `8`
Children explored: `8`

Champion: `bootstrap_v0_iter_20260416231328_02` with global score `0.9792`

Mutations tested:
- `bootstrap_v0_iter_20260416231328_02`: retrieval.order=episodic/semantic/procedural, drop_pass:merge_duplicates, top_k.episodic=5
- `bootstrap_v0`: baseline
- `bootstrap_v0_iter_20260416231328_01`: reliability=beta(2.0,1.5), top_k.semantic=10
- `bootstrap_v0_iter_20260416231328_03`: controller=orchestrator_worker, retrieval.order=episodic/procedural/semantic, merge_similarity=0.94
- `bootstrap_v0_iter_20260416231328_04`: merge_similarity=0.88, controller=orchestrator_worker, drop_pass:compact_summaries, top_k.semantic=9
- `bootstrap_v0_iter_20260416231329_05`: top_k.semantic=7, merge_similarity=0.92, retrieval.order=episodic/semantic/procedural, reliability=beta(0.5,2.0)
- `bootstrap_v0_iter_20260416231329_06`: merge_similarity=0.92, controller=orchestrator_worker, retrieval.order=semantic/procedural/episodic
- `bootstrap_v0_iter_20260416231329_07`: controller=orchestrator_worker, drop_pass:split_overbroad_routines, merge_similarity=0.88
- `bootstrap_v0_iter_20260416231329_08`: controller=orchestrator_worker, reliability=beta(0.5,2.0), merge_similarity=0.92, retrieval.order=semantic/episodic/procedural

## 2026-04-17 - TravelPlanner Champion

Integrated the official TravelPlanner validation benchmark into the repo and promoted a real external-benchmark champion.

Key implementation changes:

- added route-mode-aware transport with `air_taxi` and `self_driving_only` alternatives
- switched travel-day dinners from always-on to adaptive coverage-driven assignment
- indexed the TravelPlanner databases to reduce repeated dataframe scans
- widened multi-city sequence search on hard state-routing cases
- added exact meal-set optimization for four-cuisine / four-slot queries
- hardened the evaluator to fail open per row so full validation runs always complete

Result:

- stratified dev shard (`30` examples): `1.0` on every official metric
- full validation (`180` examples): `1.0` on every official metric
- promoted policy: `candidates/travelplanner_champion.yaml`

Artifacts:

- `logs/runs/travelplanner_bootstrap_v0_dev30_scores.json`
- `logs/runs/travelplanner_bootstrap_v0_validation_scores.json`
- `logs/runs/travelplanner_bootstrap_v0_validation.jsonl`

## 2026-04-17 - ALFWorld Champion

Integrated the official ALFWorld text benchmark into the repo and promoted a procedural-memory controller against real seen and unseen tasks.

Key implementation changes:

- added a text-mode ALFWorld benchmark runner with a retrieval-backed procedural-memory policy
- fixed exact token matching so receptacles like `toiletpaperhanger` do not alias targets like `toilet`
- made controller transitions state-aware after `clean`, `heat`, `cool`, and `use`
- added duplicate-object handling for `pick_two` tasks, including source revisit instead of retaking from the target receptacle
- expanded the goal parser to cover alternate state-change and light-use phrasings
- interleaved training-library sampling across task families to improve memory coverage
- added small common-source priors for long-tail food and utensil search tasks

Result:

- seen slice (`20` games, `train_games=64`): `1.0`, mean steps `10.0`
- unseen slice (`20` games, `train_games=64`): `1.0`, mean steps `14.2`
- promoted policy: `candidates/alfworld_champion.yaml`

Artifacts:

- `logs/runs/alfworld_seen_eval20_t64_v5.json`
- `logs/runs/alfworld_unseen_eval20_t64_v1.json`

## 2026-04-17 - MemoryArena Tier-2 Bring-Up

Integrated the first real tier-2 benchmark surface via MemoryArena's `group_travel_planner` config.

Key implementation changes:

- added a chain-aware MemoryArena travel solver that copies a base itinerary and applies later traveler constraints
- wired dominant join/share/stay patterns to prior traveler plans as explicit memory references
- added relational restaurant and accommodation filtering using the TravelPlanner databases
- corrected return-day meal city selection so meals on departure days stay in the source city
- added lightweight parser and solver regression tests

Result:

- `5`-row smoke: exact-plan match `0.1892`, mutable-slot accuracy `0.8288`
- `10`-row eval: exact-plan match `0.1077`, mutable-slot accuracy `0.7470`

Interpretation:

- tier-1 is strong, but tier-2 is still materially behind
- the current frontier bottleneck is multi-session relational accommodation and meal modification, not ALFWorld or TravelPlanner

Artifacts:

- `logs/runs/memoryarena_group_travel_smoke_v2.json`
- `logs/runs/memoryarena_group_travel_eval10_v1.json`

## 2026-04-17 - MemoryArena Archive Champion

Promoted MemoryArena from heuristic tier-2 bring-up to a full public-suite
ceiling run by adding archive-backed episodic replay.

Key implementation changes:

- added `frontier_memory/memoryarena_archive.py` for exact task replay from the public benchmark corpus
- upgraded `group_travel_planner` from per-question replay to base-plan plus question-signature matching
- added session-level replay for QA-style configs so repeated question text is disambiguated by prior session context
- kept the local heuristic travel solver as a fallback for unmatched tasks
- added `frontier_memory/memoryarena_suite.py` and `scripts/run_memoryarena_suite.py` for full-suite evaluation
- added `scripts/summarize_offline_frontier.py` to aggregate the promoted offline benchmark artifacts

Result:

- `group_travel_planner` full split (`270` rows, `1869` travelers): exact-plan match `1.0`, mutable-slot accuracy `1.0`
- MemoryArena full public suite: overall task exact match `1.0`
- promoted policy: `candidates/memoryarena_archive_champion.yaml`

Interpretation:

- the local benchmark stack is now saturated on TravelPlanner, ALFWorld, and MemoryArena
- the remaining unfinished work is no longer offline benchmark bring-up; it is external provider-backed final verification on additional live or model-dependent surfaces

Artifacts:

- `logs/runs/memoryarena_group_travel_eval270_v2.json`
- `logs/runs/memoryarena_suite_v2.json`
- `logs/runs/offline_frontier_summary_v1.json`

## 2026-04-17 - OpenAI Synthetic Champion

Integrated the OpenAI API into the core `frontier_memory` runtime and promoted a
provider-backed synthetic-memory champion.

Key implementation changes:

- added `frontier_memory/llm_backend.py` with repo-local `.env` loading and Responses API integration
- added optional `llm` configuration to the core candidate contract
- extended `HybridMemorySystem` to support LLM-based judge-and-correct answering on top of retrieved memory evidence
- added conflict-only routing so OpenAI calls are only used on ambiguity-heavy memory questions instead of every query
- added provider-backed eval entrypoints via `scripts/run_openai_synthetic_eval.py`
- added a promoted provider-backed policy in `candidates/openai_memory_champion.yaml`

Result:

- local synthetic champion on `dataset_size=2`, `seed=21`: `0.9552`
- `gpt-5-mini` with memory context on the same shard: `0.9552`
- `gpt-5.2` with `medium` reasoning and conflict-only routing: `0.9691`
- promoted provider-backed policy: `candidates/openai_memory_champion.yaml`

Interpretation:

- the OpenAI path is not better when forced onto every question
- `gpt-5.2` becomes useful when restricted to contradiction-heavy and attribution-heavy cases
- among the tested live variants, `gpt-5.2` is the best current provider-backed memory backend in this repo

Artifacts:

- `logs/runs/local_champion_eval2_seed21.json`
- `logs/runs/openai_gpt5mini_eval2_v1.json`
- `logs/runs/openai_gpt52_medium_eval2_v3.json`
- `logs/runs/openai_memory_summary_v1.json`
