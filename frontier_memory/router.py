from __future__ import annotations

import re
from typing import Iterable, List, Optional

from .types import QueryPlan
from .utils import extract_capitalized_phrases, normalize_text


RELATIONS = ["colleague", "neighbour", "friend", "cousin", "classmate"]
QUESTION_WORDS = {"What", "Where", "Who", "When", "How", "Is", "Did", "Does", "Has", "Was"}


class QueryRouter:
    def route(self, question: str, known_subjects: Iterable[str]) -> QueryPlan:
        subject = self._extract_subject(question, known_subjects)
        chain = self._parse_chain(question)
        if chain is not None:
            terminal_hint = self._terminal_hint(chain["terminal_phrase"])
            return QueryPlan(
                intent="chain",
                subject=chain["subject"],
                relation_hints=[f"link:{relation}" for relation in chain["relations"]],
                chain_relations=chain["relations"],
                terminal_hint=terminal_hint,
                wants_current=True,
            )

        lower = normalize_text(question)
        relation_hints = self._infer_relation_hints(lower)
        wants_history = any(
            phrase in lower
            for phrase in ("before", "used to", "historical", "prior", "previous")
        )
        wants_current = any(
            phrase in lower
            for phrase in ("current", "currently", " now", "these days", "still", "live now")
        )
        wants_boolean = lower.startswith(("is ", "has ", "did ", "does ", "was "))
        target_text = self._extract_target_text(question, subject)

        if lower.startswith("when") and any(
            word in lower for word in ("switch", "move", "change", "pick up")
        ):
            intent = "transition"
        elif "ever" in lower:
            intent = "ever"
            wants_boolean = True
        elif lower.startswith("who does") and "work with" in lower:
            intent = "direct_relation"
            relation_hints = ["link:colleague"]
        elif "next door" in lower and lower.startswith("who"):
            intent = "direct_relation"
            relation_hints = ["link:neighbour"]
        elif lower.startswith("who is") and "close friend" in lower:
            intent = "direct_relation"
            relation_hints = ["link:friend"]
        elif wants_history:
            intent = "historical"
        elif wants_current:
            intent = "current"
        elif wants_boolean:
            intent = "boolean"
        else:
            intent = "general"

        return QueryPlan(
            intent=intent,
            subject=subject,
            relation_hints=relation_hints,
            target_text=target_text,
            wants_boolean=wants_boolean,
            wants_history=wants_history,
            wants_current=wants_current,
        )

    def _extract_subject(self, question: str, known_subjects: Iterable[str]) -> Optional[str]:
        known = list(known_subjects)
        for subject in known:
            if subject in question:
                return subject
        capitalized = [
            token
            for token in extract_capitalized_phrases(question)
            if token not in QUESTION_WORDS
        ]
        return capitalized[0] if capitalized else None

    def _infer_relation_hints(self, lower_question: str) -> List[str]:
        hints: List[str] = []
        mapping = {
            "work": ["attr:job", "topic:work", "terminal:job"],
            "job": ["attr:job", "topic:work", "terminal:job"],
            "promotion": ["topic:work"],
            "professional": ["topic:work"],
            "planning": ["topic:work"],
            "summer": ["topic:work"],
            "writing": ["topic:work"],
            "travel": ["topic:travel"],
            "live": ["attr:city", "terminal:city", "claim:hometown", "topic:travel"],
            "move": ["attr:city", "terminal:city"],
            "dating": ["attr:relationship", "attr:relationship_status"],
            "relationship": ["attr:relationship", "attr:relationship_status"],
            "hobby": ["attr:hobby", "terminal:hobby"],
            "weekend": ["claim:weekend_activity"],
            "dog": ["claim:dog_name"],
            "degree": ["claim:degree"],
            "instrument": ["claim:instrument"],
            "piano": ["claim:instrument"],
            "fear": ["claim:fear"],
            "afraid": ["claim:fear"],
            "food": ["topic:food", "claim:food"],
            "cuisine": ["topic:food", "claim:food"],
            "discover": ["topic:food"],
            "discovered": ["topic:food"],
            "enjoying": ["topic:food"],
            "making": ["topic:food"],
            "dietary": ["topic:food"],
            "twin": ["claim:twin"],
            "record": ["claim:world_record"],
            "remote": ["attr:commute"],
        }
        for trigger, relation_hints in mapping.items():
            if trigger in lower_question:
                hints.extend(relation_hints)
        if "work with" in lower_question:
            hints.append("link:colleague")
        if "next door" in lower_question:
            hints.append("link:neighbour")
        if "close friend" in lower_question:
            hints.append("link:friend")
        if "cousin" in lower_question:
            hints.append("link:cousin")
        if "classmate" in lower_question:
            hints.append("link:classmate")
        deduped = []
        seen = set()
        for hint in hints:
            if hint not in seen:
                seen.add(hint)
                deduped.append(hint)
        return deduped

    def _extract_target_text(self, question: str, subject: Optional[str]) -> Optional[str]:
        if subject is None:
            return None
        text = question.strip().rstrip("?")
        if " ever " in text.lower():
            suffix = re.split(r"\bever\b", text, flags=re.IGNORECASE, maxsplit=1)[-1]
            return suffix.strip()
        if subject in text:
            suffix = text.split(subject, 1)[1]
            suffix = suffix.strip()
            suffix = suffix.lstrip("'s ").strip()
            return suffix or None
        return None

    def _parse_chain(self, question: str) -> Optional[dict]:
        stripped = question.strip().rstrip("?")
        lowered = normalize_text(stripped)
        if lowered.startswith(("what does ", "where does ", "who is ", "who does ")) and "'s " in stripped:
            prefix_removed = re.sub(r"^(What|Where|Who|How)\s+(does|is)\s+", "", stripped)
            segments = prefix_removed.split("'s ")
            if len(segments) < 2:
                return None
            subject = segments[0].strip()
            relations: List[str] = []
            terminal_phrase = ""
            for segment in segments[1:]:
                clean = segment.strip()
                matched = False
                for relation in RELATIONS:
                    if clean.startswith(relation):
                        relations.append(relation)
                        tail = clean[len(relation) :].strip()
                        if tail:
                            terminal_phrase = tail
                        matched = True
                        break
                if not matched and not terminal_phrase:
                    terminal_phrase = clean
            if relations:
                return {
                    "subject": subject,
                    "relations": relations,
                    "terminal_phrase": terminal_phrase,
                }
        return None

    def _terminal_hint(self, terminal_phrase: str) -> Optional[str]:
        lowered = normalize_text(terminal_phrase)
        if not lowered:
            return None
        if "live" in lowered:
            return "city"
        if "work" in lowered:
            return "job"
        if "hobby" in lowered or "enjoy" in lowered:
            return "hobby"
        return None
