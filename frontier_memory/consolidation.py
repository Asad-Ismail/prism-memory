from __future__ import annotations

from .config import CandidateConfig
from .episodic import EpisodicStore
from .procedural import ProceduralStore
from .semantic import SemanticStore


class DreamConsolidator:
    def __init__(self, candidate: CandidateConfig) -> None:
        self.candidate = candidate

    def run(
        self,
        episodic: EpisodicStore,
        semantic: SemanticStore,
        procedural: ProceduralStore,
    ) -> None:
        passes = set(self.candidate.get("consolidation", "passes", default=[]))
        thresholds = self.candidate.get("consolidation", "thresholds", default={})
        merge_similarity = float(thresholds.get("merge_similarity", 0.90))
        min_evidence = int(thresholds.get("min_evidence_for_refinement", 2))
        prune_threshold = float(thresholds.get("prune_confidence_below", 0.15))

        if "merge_duplicates" in passes:
            semantic.merge_duplicates(merge_similarity)
        if "compose_playbooks" in passes or "mark_critical_vs_noise" in passes:
            use_failures = bool(
                self.candidate.get(
                    "memory",
                    "procedural",
                    "build",
                    "use_failures",
                    default=True,
                )
            )
            procedural.learn_from_events(
                episodic.events,
                min_evidence=min_evidence,
                use_failures=use_failures,
            )
        if "split_overbroad_routines" in passes or "deprecate_stale" in passes:
            procedural.split_overbroad(prune_threshold)
        if "compact_summaries" in passes:
            episodic.compact_summaries()
