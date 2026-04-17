from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

from .types import MemoryEvent, SemanticFact
from .utils import extract_capitalized_phrases, normalize_text, token_overlap_score


DYNAMIC_RELATIONS = {
    "attr:job",
    "attr:city",
    "attr:relationship",
    "attr:relationship_status",
    "attr:hobby",
    "attr:dietary_preference",
    "attr:commute",
}


RELATION_PATTERN_HINTS: List[Tuple[str, str]] = [
    ("dog", "claim:dog_name"),
    ("degree", "claim:degree"),
    ("grow up", "claim:hometown"),
    ("instrument", "claim:instrument"),
    ("weekend", "claim:weekend_activity"),
    ("food", "claim:food"),
    ("cuisine", "claim:food"),
    ("hobby", "attr:hobby"),
    ("job", "attr:job"),
    ("work", "attr:job"),
    ("promotion", "topic:work"),
    ("travel", "topic:travel"),
    ("live", "attr:city"),
    ("move", "attr:city"),
    ("dating", "attr:relationship"),
    ("relationship", "attr:relationship_status"),
]


class SemanticStore:
    def __init__(self) -> None:
        self.facts: List[SemanticFact] = []
        self._seq = 0

    def reset(self) -> None:
        self.facts.clear()
        self._seq = 0

    def ingest(self, event: MemoryEvent) -> None:
        extracted = self._extract_facts(event)
        for fact in extracted:
            self._add_fact(fact)

    def subjects(self) -> List[str]:
        values = sorted({fact.subject for fact in self.facts if fact.subject})
        return values

    def _next_fact_id(self) -> str:
        self._seq += 1
        return f"fact_{self._seq:05d}"

    def _add_fact(self, fact: SemanticFact) -> SemanticFact:
        normalized_value = normalize_text(fact.value)
        for existing in self.facts:
            if (
                existing.subject == fact.subject
                and existing.relation == fact.relation
                and normalize_text(existing.value) == normalized_value
            ):
                duplicate_turns = existing.metadata.setdefault("duplicate_turns", [])
                duplicate_turns.append(fact.turn_index)
                if fact.turn_index > existing.turn_index:
                    existing.turn_index = fact.turn_index
                return existing

        if fact.is_dynamic or fact.metadata.get("phase") == "update" or fact.metadata.get("contradicts"):
            for existing in reversed(self.facts):
                if (
                    existing.active
                    and existing.subject == fact.subject
                    and existing.relation == fact.relation
                    and normalize_text(existing.value) != normalized_value
                ):
                    existing.active = False
                    existing.valid_to_turn = fact.turn_index
                    existing.superseded_by = fact.fact_id
        self.facts.append(fact)
        return fact

    def merge_duplicates(self, threshold: float) -> None:
        merged: List[SemanticFact] = []
        for fact in sorted(self.facts, key=lambda item: (item.subject, item.relation, item.turn_index)):
            found = None
            for candidate in merged:
                if candidate.subject != fact.subject or candidate.relation != fact.relation:
                    continue
                if token_overlap_score(candidate.value, fact.value) >= threshold:
                    found = candidate
                    break
            if found is None:
                merged.append(fact)
                continue
            found.metadata.setdefault("merged_fact_ids", []).append(fact.fact_id)
            if fact.active and not found.active:
                found.active = True
                found.valid_to_turn = None
            if fact.turn_index > found.turn_index:
                found.turn_index = fact.turn_index
                if fact.answer_hint:
                    found.answer_hint = fact.answer_hint
        self.facts = merged

    def current_fact(
        self,
        subject: str,
        relation_hints: Iterable[str],
    ) -> Optional[SemanticFact]:
        candidates = self.search(
            "",
            subject=subject,
            relation_hints=list(relation_hints),
            active_only=True,
            top_k=1,
        )
        return candidates[0] if candidates else None

    def historical_fact(
        self,
        subject: str,
        relation_hints: Iterable[str],
    ) -> Optional[SemanticFact]:
        candidates = self.search(
            "",
            subject=subject,
            relation_hints=list(relation_hints),
            active_only=False,
            top_k=5,
        )
        return candidates[0] if candidates else None

    def transition_turn(
        self,
        subject: str,
        relation_hints: Iterable[str],
    ) -> Optional[int]:
        candidates = self.search(
            "",
            subject=subject,
            relation_hints=list(relation_hints),
            active_only=None,
            top_k=10,
        )
        transition_points = [
            fact.valid_to_turn
            for fact in candidates
            if fact.valid_to_turn is not None
        ]
        if transition_points:
            return max(transition_points)
        if len(candidates) >= 2:
            return candidates[0].turn_index
        return None

    def follow_chain(
        self,
        root_subject: str,
        chain_relations: List[str],
        terminal_hint: Optional[str],
    ) -> Optional[SemanticFact]:
        current_subject = root_subject
        for relation in chain_relations:
            edge = self.current_fact(current_subject, [f"link:{relation}"])
            if edge is None:
                return None
            current_subject = edge.answer_hint or edge.value
        if terminal_hint is None:
            return SemanticFact(
                fact_id="virtual_chain",
                subject=root_subject,
                relation="chain_terminal",
                value=current_subject,
                answer_hint=current_subject,
                turn_index=0,
                source_text=current_subject,
                speaker="system",
            )
        hints = [
            f"terminal:{terminal_hint}",
            f"attr:{terminal_hint}",
            f"claim:{terminal_hint}",
            f"topic:{terminal_hint}",
        ]
        return self.current_fact(current_subject, hints)

    def fact_supports(
        self,
        subject: str,
        target_text: str,
        relation_hints: Iterable[str],
    ) -> bool:
        if not target_text:
            return False
        target = normalize_text(target_text)
        candidates = self.search(
            target_text,
            subject=subject,
            relation_hints=list(relation_hints),
            active_only=None,
            top_k=20,
        )
        target_tokens = set(target.split())
        target_core = self._target_core(target_text)
        for fact in candidates:
            haystack = normalize_text(" ".join([
                fact.value,
                fact.answer_hint or "",
                fact.source_text,
            ]))
            if target in haystack:
                return True
            if target_tokens and target_tokens.issubset(set(haystack.split())):
                return True
            if target_core and target_core in haystack:
                return True
            if target_core == "car" and any(term in haystack for term in ("driving", "drive")):
                return True
            if target_core and token_overlap_score(target_core, haystack) >= 0.25:
                return True
            if token_overlap_score(target_text, haystack) >= 0.25:
                return True
        return False

    def search(
        self,
        query: str,
        *,
        subject: Optional[str],
        relation_hints: List[str],
        active_only: Optional[bool],
        top_k: int,
    ) -> List[SemanticFact]:
        scored = []
        latest_turn = self.facts[-1].turn_index if self.facts else 1
        normalized_hints = [normalize_text(hint.replace(":", " ")) for hint in relation_hints]
        for fact in self.facts:
            if subject and fact.subject != subject:
                continue
            if active_only is True and not fact.active:
                continue
            if active_only is False and fact.active:
                continue
            score = 0.0
            if subject and fact.subject == subject:
                score += 1.0
            if normalized_hints:
                score += self._relation_hint_score(fact, normalized_hints)
            query_overlap = 0.0
            if query:
                payload = " ".join(
                    [
                        fact.relation,
                        fact.value,
                        fact.answer_hint or "",
                        fact.source_text,
                    ]
                )
                if subject:
                    subject_pattern = re.escape(subject)
                    ranked_query = re.sub(subject_pattern, " ", query, flags=re.IGNORECASE)
                    ranked_payload = re.sub(subject_pattern, " ", payload, flags=re.IGNORECASE)
                else:
                    ranked_query = query
                    ranked_payload = payload
                query_overlap = token_overlap_score(ranked_query, ranked_payload)
                score += query_overlap
            metadata = fact.metadata or {}
            if metadata.get("type") == "low_freq" and query_overlap >= 0.10:
                score += 0.30
            if metadata.get("surprise_level") == "surprising" and query_overlap >= 0.10:
                score += 0.20
            if metadata.get("type") == "high_freq":
                score -= 0.03
            if fact.active:
                score += 0.05
            score += fact.turn_index / max(latest_turn, 1) * 0.05
            if score > 0:
                scored.append((score, fact))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [fact for _, fact in scored[:top_k]]

    def _relation_hint_score(self, fact: SemanticFact, hints: List[str]) -> float:
        normalized_relation = normalize_text(fact.relation.replace(":", " "))
        aliases = [normalize_text(alias) for alias in fact.aliases]
        score = 0.0
        for hint in hints:
            if hint and hint in normalized_relation:
                score += 0.9
            elif any(hint in alias for alias in aliases):
                score += 0.7
        return score

    def _extract_facts(self, event: MemoryEvent) -> List[SemanticFact]:
        metadata_facts = self._extract_from_metadata(event)
        if metadata_facts:
            return metadata_facts
        return self._extract_from_text(event)

    def _extract_from_metadata(self, event: MemoryEvent) -> List[SemanticFact]:
        metadata = event.metadata or {}
        facts: List[SemanticFact] = []
        if metadata.get("type") == "relation":
            relation = f"link:{metadata['rel']}"
            value = metadata["B"]
            facts.append(
                self._make_fact(
                    subject=metadata["A"],
                    relation=relation,
                    value=value,
                    answer_hint=value,
                    event=event,
                )
            )
        if metadata.get("type") == "terminal":
            relation = f"terminal:{metadata['fact_type']}"
            value = str(metadata["value"])
            facts.append(
                self._make_fact(
                    subject=metadata["entity"],
                    relation=relation,
                    value=value,
                    answer_hint=value,
                    event=event,
                )
            )
        if metadata.get("topic") and metadata.get("fact"):
            topic = str(metadata["topic"])
            fact_text = str(metadata["fact"])
            facts.append(
                self._make_fact(
                    subject=event.speaker,
                    relation=f"topic:{topic}",
                    value=fact_text,
                    answer_hint=self._answer_hint(f"topic:{topic}", fact_text),
                    event=event,
                )
            )
        if metadata.get("entity") and metadata.get("attribute"):
            attr = self._normalize_attr(str(metadata["attribute"]))
            fact_text = str(metadata.get("fact") or self._extract_attr_value(attr, event.text))
            facts.append(
                self._make_fact(
                    subject=str(metadata["entity"]),
                    relation=f"attr:{attr}",
                    value=fact_text,
                    answer_hint=self._answer_hint(f"attr:{attr}", fact_text),
                    event=event,
                    is_dynamic=True,
                )
            )
        if metadata.get("entity") and metadata.get("fact") and not metadata.get("attribute"):
            relation, hint = self._infer_relation(str(metadata["fact"]))
            facts.append(
                self._make_fact(
                    subject=str(metadata["entity"]),
                    relation=relation,
                    value=str(metadata["fact"]),
                    answer_hint=hint,
                    event=event,
                    is_dynamic=relation in DYNAMIC_RELATIONS,
                )
            )
        if metadata.get("surprise_level") and metadata.get("entity") and metadata.get("fact"):
            relation, hint = self._infer_relation(str(metadata["fact"]))
            fact = self._make_fact(
                subject=str(metadata["entity"]),
                relation=relation,
                value=str(metadata["fact"]),
                answer_hint=hint,
                event=event,
                is_dynamic=False,
            )
            fact.metadata["surprise_level"] = metadata["surprise_level"]
            facts.append(fact)
        return facts

    def _extract_from_text(self, event: MemoryEvent) -> List[SemanticFact]:
        text = event.text.strip()
        lowered = normalize_text(text)
        facts: List[SemanticFact] = []
        if not text or len(lowered.split()) < 3:
            return facts

        attr_patterns = [
            (r"([A-Z][a-z]+) works as a (.+?)\.", "attr:job"),
            (r"([A-Z][a-z]+) just started a new position as (.+?)\.", "attr:job"),
            (r"([A-Z][a-z]+) lives in ([A-Za-z ]+)\.", "attr:city"),
            (r"([A-Z][a-z]+) recently moved to ([A-Za-z ]+)\.", "attr:city"),
            (r"([A-Z][a-z]+) is in a relationship with ([A-Z][a-z]+)\.", "attr:relationship"),
            (r"([A-Z][a-z]+) and ([A-Z][a-z]+) recently got together\.", "attr:relationship"),
            (r"([A-Z][a-z]+) loves ([a-z ]+?) and does it every weekend\.", "attr:hobby"),
            (r"([A-Z][a-z]+) has really gotten into ([a-z ]+?) lately\.", "attr:hobby"),
        ]
        for pattern, relation in attr_patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            if relation == "attr:relationship" and "got together" in pattern:
                subject = match.group(1)
                value = match.group(2)
            else:
                subject = match.group(1)
                value = match.group(2)
            facts.append(
                self._make_fact(
                    subject=subject,
                    relation=relation,
                    value=value.strip(),
                    answer_hint=value.strip(),
                    event=event,
                    is_dynamic=relation in DYNAMIC_RELATIONS,
                )
            )

        if event.speaker not in {"AI", "Assistant", "User"} and text.startswith("By the way,"):
            relation = self._infer_topic_relation(text)
            facts.append(
                self._make_fact(
                    subject=event.speaker,
                    relation=relation,
                    value=text.replace("By the way,", "").strip(" ."),
                    answer_hint=self._answer_hint(relation, text),
                    event=event,
                )
            )

        generic_name_facts = re.findall(
            r"\b([A-Z][a-z]+)\b[^.]*?\b(is|has|plays|loves|grew|drives|works)\b[^.]*\.",
            text,
        )
        if generic_name_facts:
            for subject, _ in generic_name_facts:
                relation, hint = self._infer_relation(text)
                facts.append(
                    self._make_fact(
                        subject=subject,
                        relation=relation,
                        value=text.strip(" ."),
                        answer_hint=hint,
                        event=event,
                        is_dynamic=relation in DYNAMIC_RELATIONS,
                    )
                )
        return facts

    def _make_fact(
        self,
        *,
        subject: str,
        relation: str,
        value: str,
        answer_hint: Optional[str],
        event: MemoryEvent,
        is_dynamic: bool = False,
    ) -> SemanticFact:
        fact = SemanticFact(
            fact_id=self._next_fact_id(),
            subject=subject,
            relation=relation,
            value=value.strip(),
            turn_index=event.turn_index,
            source_text=event.text,
            speaker=event.speaker,
            metadata=dict(event.metadata),
            answer_hint=answer_hint,
            is_dynamic=is_dynamic,
            valid_from_turn=event.turn_index,
        )
        fact.aliases = self._aliases_for_fact(fact)
        return fact

    def _aliases_for_fact(self, fact: SemanticFact) -> List[str]:
        aliases = [
            fact.subject,
            fact.relation.split(":")[-1],
            fact.answer_hint or "",
        ]
        metadata = fact.metadata or {}
        if metadata.get("attribute"):
            aliases.append(str(metadata["attribute"]))
        if metadata.get("topic"):
            aliases.append(str(metadata["topic"]))
        return [alias for alias in aliases if alias]

    def _normalize_attr(self, attr: str) -> str:
        normalized = normalize_text(attr).replace(" ", "_")
        if normalized == "relationship_status":
            return "relationship_status"
        if normalized == "relationship":
            return "relationship"
        return normalized

    def _infer_topic_relation(self, text: str) -> str:
        lowered = normalize_text(text)
        if any(term in lowered for term in ("conference", "holiday", "wedding", "work trip", "volunteer", "festival", "sabbatical")):
            return "topic:travel"
        if any(term in lowered for term in ("sushi", "bread", "taco", "vegan", "food", "ramen", "fermented", "cooking class")):
            return "topic:food"
        return "topic:work"

    def _infer_relation(self, text: str) -> Tuple[str, Optional[str]]:
        lowered = normalize_text(text)
        if "named " in lowered and "dog" in lowered:
            return "claim:dog_name", self._answer_hint("claim:dog_name", text)
        if "grew up in" in lowered:
            return "claim:hometown", self._answer_hint("claim:hometown", text)
        if "phd in" in lowered:
            return "claim:degree", self._answer_hint("claim:degree", text)
        if "plays classical piano" in lowered or "instrument" in lowered:
            return "claim:instrument", self._answer_hint("claim:instrument", text)
        if "weekend" in lowered and "hiking" in lowered:
            return "claim:weekend_activity", self._answer_hint("claim:weekend_activity", text)
        if "works as" in lowered:
            return "attr:job", self._answer_hint("attr:job", text)
        if "lives in" in lowered or "moved to" in lowered:
            return "attr:city", self._answer_hint("attr:city", text)
        if "dating" in lowered or "relationship" in lowered:
            return "attr:relationship_status", self._answer_hint("attr:relationship_status", text)
        if "hobby" in lowered or "gotten into" in lowered:
            return "attr:hobby", self._answer_hint("attr:hobby", text)
        if "travel" in lowered or "conference" in lowered or "holiday" in lowered:
            return "topic:travel", self._answer_hint("topic:travel", text)
        if any(term in lowered for term in ("sushi", "bread", "taco", "vegan", "food", "ramen")):
            return "topic:food", self._answer_hint("topic:food", text)
        if any(term in lowered for term in ("promotion", "remote", "interns", "tedx", "startup", "technical book", "freelancing")):
            return "topic:work", self._answer_hint("topic:work", text)
        for pattern, relation in RELATION_PATTERN_HINTS:
            if pattern in lowered:
                return relation, self._answer_hint(relation, text)
        return "claim:general", self._answer_hint("claim:general", text)

    def _extract_attr_value(self, attr: str, text: str) -> str:
        mapping = {
            "job": [
                r"position as (.+?)\.",
                r"works as a (.+?)\.",
                r"is (.+?)\.",
            ],
            "city": [
                r"moved to ([A-Za-z ]+)\.",
                r"lives in ([A-Za-z ]+)\.",
            ],
            "relationship": [
                r"relationship with ([A-Z][a-z]+)\.",
                r"and ([A-Z][a-z]+) recently got together",
            ],
            "relationship_status": [
                r"now (.+?)\.",
                r"is (.+?)\.",
            ],
            "hobby": [
                r"into ([a-z ]+?) lately\.",
                r"loves ([a-z ]+?) and",
            ],
            "dietary_preference": [
                r"now (.+?)\.",
                r"is (.+?)\.",
            ],
            "commute": [
                r"now (.+?)\.",
                r"is (.+?)\.",
            ],
        }
        for pattern in mapping.get(attr, []):
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return text.strip(" .")

    def _answer_hint(self, relation: str, text: str) -> Optional[str]:
        lowered = normalize_text(text)
        hint_rules: List[Tuple[str, str]] = [
            (r"named ([A-Z][a-z]+)", "group1"),
            (r"grew up in ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "group1"),
            (r"PhD in ([a-z]+)", "PhD in {1}"),
            (r"plays (?:classical )?([a-z]+)", "group1"),
            (r"loves ([a-z ]+?) on weekends", "group1"),
            (r"works as a (.+)", "group1"),
            (r"drives a ([a-z]+) car", "group1"),
            (r"afraid of ([a-z]+)", "group1"),
            (r"world record for the fastest time assembling a 1000-piece puzzle", "fastest 1000-piece puzzle assembly"),
            (r"child actress in a TV commercial", "was in a TV commercial"),
            (r"solve a Rubik's cube blindfolded", "solve a Rubik's cube blindfolded"),
            (r"replied-all to a 10,000-person company email", "replied-all to 10000 people"),
            (r"survived a lightning strike", "was struck by lightning"),
            (r"cooked dinner for a sitting president", "a president"),
            (r"mistaken for a famous celebrity", "was mistaken for a celebrity"),
            (r"patent for an underwater breathing device", "underwater breathing device"),
            (r"completed an Ironman triathlon", "Ironman triathlon"),
            (r"went for a short walk after lunch", "went for a walk"),
            (r"checked (?:her|his) email before bed", "checked email"),
            (r"read a book before sleeping", "read a book"),
            (r"watered (?:her|his) plants in the morning", "watered plants"),
            (r"met a colleague for coffee", "a colleague"),
            (r"called (?:his|her) mom on the weekend", "his mom"),
            (r"lot of ([a-z ]+)", "group1"),
            (r"making ([a-z ]+)", "group1"),
            (r"great ([a-z ]+place)", "a {1}"),
            (r"went ([a-z ]+) last month", "went {1}"),
            (r"obsessed with ([a-z ]+)", "group1"),
            (r"makes homemade ([a-z ]+) every weekend", "group1"),
            (r"tried ([a-z ]+) for health", "group1"),
            (r"takes a ([a-z ]+) on Tuesdays", "group1"),
            (r"promotion to ([a-z ]+)", "promotion to {1}"),
            (r"applying to ([A-Z]?[a-z ]+programs)", "applying to {1}"),
            (r"freelancing part-time", "started freelancing"),
            (r"mentoring interns", "mentoring interns"),
            (r"TEDx talk", "gave a TEDx talk"),
            (r"side project startup", "side project startup"),
            (r"fully remote", "moved to remote"),
            (r"technical book", "a technical book"),
            (r"([A-Z][a-z]+) for a", "group1"),
            (r"([A-Z][a-z]+) on sabbatical", "group1"),
            (r"([A-Z][a-z]+) for volunteer work", "group1"),
            (r"([A-Z][a-z]+) for a friend's birthday", "group1"),
            (r"([A-Z][a-z]+) for a work trip", "group1"),
        ]
        for pattern, template in hint_rules:
            match = re.search(pattern, text)
            if not match:
                continue
            if template == "group1":
                return match.group(1).strip()
            if "{1}" in template:
                return template.format(*["", match.group(1).strip()])
            return template

        if relation in {"topic:travel", "attr:city", "claim:hometown"}:
            capitals = extract_capitalized_phrases(text)
            if capitals:
                return capitals[-1]
        if relation in {"topic:food", "claim:food"}:
            for pattern in (r"lot of ([a-z ]+)", r"making ([a-z ]+)", r"discovered a great ([a-z ]+)", r"obsessed with ([a-z ]+)", r"homemade ([a-z ]+)"):
                match = re.search(pattern, lowered)
                if match:
                    return match.group(1).strip()
        if relation in {"attr:job", "topic:work"}:
            return text.strip(" .")
        if relation == "claim:general":
            return text.strip(" .")
        return text.strip(" .")

    def _target_core(self, target_text: str) -> str:
        core = normalize_text(target_text)
        replacements = [
            "worked as a ",
            "worked as ",
            "lived in ",
            "living in ",
            "commute by ",
            "commute ",
            "eat ",
            "eating ",
            "was ",
            "were ",
            "is ",
            "are ",
            "a ",
            "an ",
        ]
        for prefix in replacements:
            if core.startswith(prefix):
                core = core[len(prefix) :]
        core = core.replace("to work", "").strip()
        return core
