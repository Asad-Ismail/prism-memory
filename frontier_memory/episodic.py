from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .types import MemoryEvent
from .utils import compact_whitespace, token_overlap_score


@dataclass
class EpisodicSummary:
    start_turn: int
    end_turn: int
    text: str


class EpisodicStore:
    def __init__(self) -> None:
        self.events: List[MemoryEvent] = []
        self.summaries: List[EpisodicSummary] = []

    def reset(self) -> None:
        self.events.clear()
        self.summaries.clear()

    def append(self, event: MemoryEvent) -> None:
        event.salience = self._salience(event)
        self.events.append(event)

    def _salience(self, event: MemoryEvent) -> float:
        score = 0.05
        metadata = event.metadata or {}
        if metadata:
            score += 0.10
        if metadata.get("phase") == "update":
            score += 0.35
        if metadata.get("type") in {"low_freq", "relation", "terminal"}:
            score += 0.25
        if metadata.get("surprise_level") == "surprising":
            score += 0.35
        lowered = event.text.lower()
        if any(term in lowered for term in ("big update", "by the way", "also", "random thing")):
            score += 0.10
        if event.speaker not in {"AI", "Assistant"}:
            score += 0.05
        return score

    def retrieve(
        self,
        query: str,
        *,
        subject: Optional[str] = None,
        top_k: int = 3,
    ) -> List[MemoryEvent]:
        scored = []
        latest_turn = self.events[-1].turn_index if self.events else 1
        for event in self.events:
            score = token_overlap_score(query, event.text)
            score += event.salience
            if subject and (subject == event.speaker or subject in event.text):
                score += 0.75
            if event.turn_index == latest_turn:
                score += 0.05
            else:
                score += event.turn_index / max(latest_turn, 1) * 0.05
            if score > 0:
                scored.append((score, event))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [event for _, event in scored[:top_k]]

    def compact_summaries(self, chunk_size: int = 20) -> None:
        self.summaries.clear()
        for start in range(0, len(self.events), chunk_size):
            window = self.events[start : start + chunk_size]
            content = [
                f"{event.turn_index}:{event.speaker}:{compact_whitespace(event.text)}"
                for event in window
                if event.salience >= 0.15
            ]
            if not content:
                continue
            self.summaries.append(
                EpisodicSummary(
                    start_turn=window[0].turn_index,
                    end_turn=window[-1].turn_index,
                    text=" | ".join(content[:6]),
                )
            )
