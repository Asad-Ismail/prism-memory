from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


DEFAULT_CONFIG: Dict[str, Any] = {
    "candidate_id": "unnamed_candidate",
    "memory": {
        "episodic": {
            "enabled": True,
            "provenance_required": True,
        },
        "semantic": {
            "enabled": True,
            "representation": "proposition_graph",
            "canonicalization": {
                "enabled": True,
                "merge_similarity": 0.90,
                "temporal_update_policy": "newest_with_provenance",
            },
        },
        "procedural": {
            "enabled": True,
            "unit_type": "hierarchical_playbook",
            "build": {
                "mode": "hybrid",
                "use_successes": True,
                "use_failures": True,
            },
            "reliability": {
                "model": "beta_posterior",
                "prior_alpha": 1.0,
                "prior_beta": 1.0,
            },
        },
        "tool_memory": {
            "enabled": True,
        },
    },
    "consolidation": {
        "cadence": "nightly",
        "passes": [
            "extract_negative_constraints",
            "mark_critical_vs_noise",
            "merge_duplicates",
            "split_overbroad_routines",
            "compose_playbooks",
            "deprecate_stale",
            "compact_summaries",
        ],
        "thresholds": {
            "merge_similarity": 0.90,
            "min_evidence_for_refinement": 2,
            "stale_days": 30,
            "prune_confidence_below": 0.15,
        },
    },
    "retrieval": {
        "router": "hybrid",
        "order": ["procedural", "semantic", "episodic"],
        "planning_mode": "direct_then_decompose",
        "top_k": {
            "procedural": 5,
            "semantic": 8,
            "episodic": 3,
        },
        "rerank": {
            "enabled": True,
            "method": "token_overlap",
        },
    },
    "controller": {
        "architecture": "single_agent",
        "use_policy": "inject_minimal_supporting_context",
        "update_policy": "validate_then_replace",
        "write_gates": {
            "require_trace_link": True,
            "require_environment_feedback": False,
            "allow_unverified_summaries": False,
        },
    },
    "llm": {
        "enabled": False,
        "provider": "openai",
        "mode": "judge_and_correct",
        "routing_policy": "conflict_only",
        "fallback_to_heuristic": True,
        "model": "gpt-5.2",
        "reasoning_effort": "medium",
        "max_output_tokens": 96,
        "verbosity": "low",
        "prompt_profile": "default",
        "refusal_text": "I don't know.",
        "context": {
            "semantic_top_k": 8,
            "procedural_top_k": 3,
            "episodic_top_k": 5,
            "include_heuristic_answer": True,
        },
    },
    "mutation_surface": {
        "allowed": [],
        "forbidden": [],
    },
}


@dataclass
class CandidateConfig:
    raw: Dict[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> "CandidateConfig":
        source = Path(path)
        data = yaml.safe_load(source.read_text()) or {}
        return cls(_deep_merge(DEFAULT_CONFIG, data))

    def clone(self) -> "CandidateConfig":
        return CandidateConfig(copy.deepcopy(self.raw))

    @property
    def candidate_id(self) -> str:
        return str(self.raw.get("candidate_id", "unnamed_candidate"))

    @property
    def parent_ids(self) -> list[str]:
        return list(self.raw.get("parent_ids", []))

    def get(self, *path: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def set(self, path: Sequence[str], value: Any) -> None:
        if not path:
            raise ValueError("path must contain at least one key")
        node = self.raw
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value

    def dump(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(self.raw, sort_keys=False))

    def allowed_mutations(self) -> list[str]:
        allowed = self.get("mutation_surface", "allowed", default=[])
        return [str(item) for item in allowed]

    def summary(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "retrieval_order": self.get("retrieval", "order", default=[]),
            "top_k": self.get("retrieval", "top_k", default={}),
            "controller_architecture": self.get("controller", "architecture"),
            "use_policy": self.get("controller", "use_policy"),
            "llm_enabled": self.get("llm", "enabled", default=False),
            "llm_model": self.get("llm", "model", default=None),
            "merge_similarity": self.get(
                "consolidation",
                "thresholds",
                "merge_similarity",
                default=0.90,
            ),
            "passes": self.get("consolidation", "passes", default=[]),
        }

    @staticmethod
    def dotted_path(parts: Iterable[str]) -> str:
        return ".".join(parts)


def load_candidate(path: str | Path) -> CandidateConfig:
    return CandidateConfig.from_file(path)
