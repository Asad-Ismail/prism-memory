from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from .config import CandidateConfig
from .consolidation import DreamConsolidator
from .episodic import EpisodicStore
from .llm_backend import build_llm_backend
from .procedural import ProceduralStore
from .router import QueryRouter
from .semantic import SemanticStore
from .types import MemoryEvent, QueryPlan, SemanticFact
from .utils import extract_numbers, normalize_text, token_overlap_score


class HybridMemorySystem:
    def __init__(self, candidate: CandidateConfig) -> None:
        self.candidate = candidate
        prior_alpha = float(
            candidate.get(
                "memory",
                "procedural",
                "reliability",
                "prior_alpha",
                default=1.0,
            )
        )
        prior_beta = float(
            candidate.get(
                "memory",
                "procedural",
                "reliability",
                "prior_beta",
                default=1.0,
            )
        )
        self.episodic = EpisodicStore()
        self.semantic = SemanticStore()
        self.procedural = ProceduralStore(prior_alpha=prior_alpha, prior_beta=prior_beta)
        self.router = QueryRouter()
        self.consolidator = DreamConsolidator(candidate)
        self.llm_backend = build_llm_backend(candidate)
        self._dirty = False

    def reset(self) -> None:
        self.episodic.reset()
        self.semantic.reset()
        self.procedural.reset()
        self._dirty = False

    def ingest(self, turn: Any) -> None:
        event = self._coerce_event(turn)
        self.episodic.append(event)
        self.semantic.ingest(event)
        self._dirty = True

    def answer(self, question: str) -> str:
        self._ensure_consolidated()
        plan = self.router.route(question, self.semantic.subjects())
        heuristic = self._answer_heuristically(plan, question)
        if self.llm_backend is None:
            return heuristic or "I don't know."

        llm_context = self._collect_llm_context(plan, question)
        if not self._should_use_llm(plan, question, heuristic, llm_context):
            return heuristic or "I don't know."

        llm_answer = self.llm_backend.answer(
            question=question,
            evidence_text=llm_context["text"],
            heuristic_answer=heuristic,
        )
        if llm_answer and llm_answer != "I don't know.":
            return llm_answer
        if self.candidate.get("llm", "fallback_to_heuristic", default=True):
            return heuristic or "I don't know."
        return llm_answer or "I don't know."

    def _answer_heuristically(self, plan: QueryPlan, question: str) -> Optional[str]:
        if plan.intent == "chain":
            response = self._answer_chain(plan, question)
            if response:
                return response
        if plan.intent == "direct_relation":
            response = self._answer_direct_relation(plan, question)
            if response:
                return response
        if plan.intent == "transition":
            response = self._answer_transition(plan)
            if response:
                return response
        if plan.intent == "historical":
            response = self._answer_temporal(plan, question, current=False)
            if response:
                return response
        if plan.intent == "current":
            response = self._answer_temporal(plan, question, current=True)
            if response:
                return response
        if plan.intent in {"ever", "boolean"}:
            response = self._answer_boolean(plan, question)
            if response:
                return response
        return self._answer_general(plan, question)

    def _collect_llm_context(self, plan: QueryPlan, question: str) -> dict[str, object]:
        subject = plan.subject
        semantic_top_k = int(
            self.candidate.get(
                "llm",
                "context",
                "semantic_top_k",
                default=self.candidate.get("retrieval", "top_k", "semantic", default=8),
            )
        )
        procedural_top_k = int(
            self.candidate.get(
                "llm",
                "context",
                "procedural_top_k",
                default=self.candidate.get("retrieval", "top_k", "procedural", default=5),
            )
        )
        episodic_top_k = int(
            self.candidate.get(
                "llm",
                "context",
                "episodic_top_k",
                default=self.candidate.get("retrieval", "top_k", "episodic", default=3),
            )
        )

        semantic_hits = self.semantic.search(
            question,
            subject=subject,
            relation_hints=plan.relation_hints,
            active_only=None,
            top_k=semantic_top_k,
        )
        procedural_hits = self.procedural.retrieve(question, top_k=procedural_top_k)
        episodic_hits = self.episodic.retrieve(question, subject=subject, top_k=episodic_top_k)

        sections = [
            f"Intent: {plan.intent}",
            f"Subject: {subject or '(unknown)'}",
        ]

        if semantic_hits:
            semantic_lines = []
            for fact in semantic_hits:
                semantic_lines.append(
                    "  - "
                    f"turn={fact.turn_index}; active={fact.active}; subject={fact.subject}; "
                    f"relation={fact.relation}; value={fact.value}; source={fact.source_text}"
                )
            sections.append("Semantic memory:\n" + "\n".join(semantic_lines))

        if procedural_hits:
            procedural_lines = []
            for proc in procedural_hits:
                step_preview = " -> ".join(proc.steps[:3])
                procedural_lines.append(
                    "  - "
                    f"reliability={proc.reliability:.2f}; trigger={' '.join(proc.trigger_terms)}; "
                    f"steps={step_preview}"
                )
            sections.append("Procedural memory:\n" + "\n".join(procedural_lines))

        if episodic_hits:
            episodic_lines = []
            for event in episodic_hits:
                timestamp = str(event.metadata.get("timestamp", "")).strip()
                session_id = str(event.metadata.get("session_id", "")).strip()
                metadata_bits = []
                if session_id:
                    metadata_bits.append(f"session={session_id}")
                if timestamp:
                    metadata_bits.append(f"time={timestamp}")
                metadata_prefix = "; ".join(metadata_bits)
                if metadata_prefix:
                    metadata_prefix = f"{metadata_prefix}; "
                episodic_lines.append(
                    "  - "
                    f"turn={event.turn_index}; {metadata_prefix}speaker={event.speaker}; text={event.text}"
                )
            sections.append("Episodic memory:\n" + "\n".join(episodic_lines))

        if self.episodic.summaries:
            summary_lines = []
            for summary in self.episodic.summaries[:2]:
                summary_lines.append(
                    "  - "
                    f"turns={summary.start_turn}-{summary.end_turn}; summary={summary.text}"
                )
            sections.append("Episodic summaries:\n" + "\n".join(summary_lines))

        return {
            "text": "\n\n".join(sections),
            "semantic_hits": semantic_hits,
            "procedural_hits": procedural_hits,
            "episodic_hits": episodic_hits,
        }

    def _should_use_llm(
        self,
        plan: QueryPlan,
        question: str,
        heuristic: Optional[str],
        llm_context: dict[str, object],
    ) -> bool:
        policy = str(self.candidate.get("llm", "routing_policy", default="always"))
        if policy == "always":
            return True
        if not heuristic or heuristic == "I don't know.":
            return True

        semantic_hits = list(llm_context.get("semantic_hits", []))
        episodic_hits = list(llm_context.get("episodic_hits", []))

        active_hits = [fact for fact in semantic_hits if fact.active]
        inactive_hits = [fact for fact in semantic_hits if not fact.active]
        lowered_question = question.lower()

        if policy == "conflict_only":
            if active_hits and inactive_hits:
                return True
            if lowered_question.startswith(("who", "which")) and plan.subject is None and len({fact.subject for fact in semantic_hits[:4]}) >= 2:
                return True
            if plan.intent in {"current", "historical", "ever", "boolean"} and len(semantic_hits) >= 2:
                if len({fact.relation for fact in semantic_hits[:3]}) <= 1:
                    return True
            if lowered_question.startswith(("who", "which")) and len({event.speaker for event in episodic_hits[:4]}) >= 3:
                return True
            return False

        return False

    def _ensure_consolidated(self) -> None:
        if not self._dirty:
            return
        self.consolidator.run(self.episodic, self.semantic, self.procedural)
        self._dirty = False

    def _coerce_event(self, turn: Any) -> MemoryEvent:
        if isinstance(turn, MemoryEvent):
            return turn
        return MemoryEvent(
            speaker=str(getattr(turn, "speaker")),
            text=str(getattr(turn, "text")),
            turn_index=int(getattr(turn, "turn_index")),
            metadata=dict(getattr(turn, "metadata", {})),
        )

    def _answer_chain(self, plan: QueryPlan, question: str) -> Optional[str]:
        if plan.subject is None:
            return None
        fact = self.semantic.follow_chain(
            root_subject=plan.subject,
            chain_relations=plan.chain_relations,
            terminal_hint=plan.terminal_hint,
        )
        if fact is None:
            return None
        return self._verbalize_fact(question, fact)

    def _answer_direct_relation(self, plan: QueryPlan, question: str) -> Optional[str]:
        if plan.subject is None:
            return None
        fact = self.semantic.current_fact(plan.subject, plan.relation_hints)
        if fact is None:
            return None
        return self._verbalize_fact(question, fact)

    def _answer_transition(self, plan: QueryPlan) -> Optional[str]:
        if plan.subject is None:
            return None
        turn_index = self.semantic.transition_turn(plan.subject, plan.relation_hints)
        if turn_index is None:
            return None
        return f"Around turn {turn_index}."

    def _answer_temporal(self, plan: QueryPlan, question: str, *, current: bool) -> Optional[str]:
        if plan.subject is None:
            return None
        if current:
            fact = self.semantic.current_fact(plan.subject, plan.relation_hints)
        else:
            fact = self.semantic.historical_fact(plan.subject, plan.relation_hints)
        if fact is None:
            return None
        return self._verbalize_fact(question, fact)

    def _answer_boolean(self, plan: QueryPlan, question: str) -> Optional[str]:
        if plan.subject is None:
            return None
        if plan.intent == "ever":
            if self.semantic.fact_supports(plan.subject, plan.target_text or "", plan.relation_hints):
                return "yes"
            return "no"

        if plan.relation_hints:
            fact = self.semantic.current_fact(plan.subject, plan.relation_hints)
            if fact is not None:
                if any(rel in fact.relation for rel in ("relationship", "commute", "job", "city")):
                    return self._verbalize_fact(question, fact)
                return "yes"

        if plan.target_text and self.semantic.fact_supports(plan.subject, plan.target_text, plan.relation_hints):
            return "yes"

        candidates = self.semantic.search(
            question,
            subject=plan.subject,
            relation_hints=plan.relation_hints,
            active_only=None,
            top_k=5,
        )
        if candidates:
            return self._verbalize_fact(question, candidates[0])
        return None

    def _answer_general(self, plan: QueryPlan, question: str) -> Optional[str]:
        subject = plan.subject
        top_k_semantic = int(self.candidate.get("retrieval", "top_k", "semantic", default=8))
        semantic_hits = self.semantic.search(
            question,
            subject=subject,
            relation_hints=plan.relation_hints,
            active_only=None,
            top_k=top_k_semantic,
        )
        if semantic_hits:
            return self._verbalize_fact(question, semantic_hits[0])

        top_k_procedural = int(self.candidate.get("retrieval", "top_k", "procedural", default=5))
        procedural_hits = self.procedural.retrieve(question, top_k=top_k_procedural)
        if procedural_hits:
            return procedural_hits[0].steps[0]

        top_k_episodic = int(self.candidate.get("retrieval", "top_k", "episodic", default=3))
        episodic_hits = self.episodic.retrieve(question, subject=subject, top_k=top_k_episodic)
        if episodic_hits:
            return self._extract_answer_from_text(question, episodic_hits[0].text)
        return None

    def _verbalize_fact(self, question: str, fact: SemanticFact) -> str:
        normalized_question = normalize_text(question)
        if fact.relation == "attr:commute" and normalized_question.startswith("how"):
            return fact.value
        if fact.answer_hint:
            if normalized_question.startswith(("is ", "has ", "did ", "does ", "was ")):
                if fact.relation.startswith("attr:relationship"):
                    return f"Yes, {fact.subject} is {fact.value}."
                return fact.value
            return fact.answer_hint
        return self._extract_answer_from_text(question, fact.value or fact.source_text)

    def _extract_answer_from_text(self, question: str, text: str) -> str:
        lowered_question = normalize_text(question)
        normalized_text = text.strip().rstrip(".")

        if lowered_question.startswith("when"):
            patterns = [
                r"(last [A-Za-z]+)",
                r"(on [A-Za-z]+)",
                r"(in the [A-Za-z]+)",
                r"(before [A-Za-z]+)",
                r"(after [A-Za-z]+)",
                r"(at age \d+)",
                r"(on the weekend)",
            ]
            for pattern in patterns:
                match = re.search(pattern, normalized_text, flags=re.IGNORECASE)
                if match:
                    return match.group(1)

        if lowered_question.startswith("where"):
            match = re.search(r"\b(?:in|to|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", normalized_text)
            if match:
                return match.group(1)

        if lowered_question.startswith("who"):
            for pattern in (
                r"named ([A-Z][a-z]+)",
                r"with ([A-Z][a-z]+)",
                r"to ([A-Z][a-z]+)",
                r"met ([A-Za-z ]+)",
            ):
                match = re.search(pattern, normalized_text)
                if match:
                    return match.group(1)

        if lowered_question.startswith("what colour") or "color" in lowered_question:
            match = re.search(r"\b([a-z]+)\s+car\b", normalized_text.lower())
            if match:
                return match.group(1)

        if lowered_question.startswith(("does ", "did ", "has ", "is ", "was ")):
            return normalized_text

        return normalized_text


def simple_answer_score(question: str, prediction: str, reference: str) -> float:
    pred = prediction.strip()
    ref = reference.strip()
    if not pred:
        return 0.0

    pred_norm = normalize_text(pred)
    ref_norm = normalize_text(ref)

    if pred_norm == ref_norm:
        return 1.0
    if ref_norm in pred_norm or pred_norm in ref_norm:
        return 1.0

    if ref_norm in {"yes", "no"}:
        if ref_norm in pred_norm.split():
            return 1.0
        if ("yes" in pred_norm and ref_norm == "no") or ("no" in pred_norm and ref_norm == "yes"):
            return 0.0

    if normalize_text(question).startswith("when") or "turn" in ref_norm:
        pred_numbers = extract_numbers(pred)
        ref_numbers = extract_numbers(ref)
        if pred_numbers and ref_numbers:
            if any(abs(left - right) <= 5 for left in pred_numbers for right in ref_numbers):
                return 1.0

    lexical = token_overlap_score(pred, ref)
    if lexical >= 0.60:
        return 1.0
    if lexical >= 0.30:
        return 0.5
    if token_overlap_score(pred, question) > 0.90:
        return 0.0
    return 0.0
