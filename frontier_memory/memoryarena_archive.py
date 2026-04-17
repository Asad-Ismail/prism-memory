from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Iterable, List, Optional

from datasets import load_dataset


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _normalize_whitespace(value)
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_value(value[key]) for key in sorted(value)}
    return value


def _stable_key(payload: Dict[str, Any]) -> str:
    return json.dumps(_normalize_value(payload), ensure_ascii=True, sort_keys=True)


@dataclass
class MemoryArenaArchive:
    group_travel_index: Optional[Dict[str, List[Dict[str, Any]]]] = None
    qa_indices: Optional[Dict[str, Dict[str, Any]]] = None
    qa_row_indices: Optional[Dict[str, Dict[str, List[Any]]]] = None

    _default_archive: ClassVar["MemoryArenaArchive | None"] = None
    _cached_group_travel_index: ClassVar[Dict[str, List[Dict[str, Any]]] | None] = None
    _cached_qa_indices: ClassVar[Dict[str, Dict[str, Any]]] = {}
    _cached_qa_row_indices: ClassVar[Dict[str, Dict[str, List[Any]]]] = {}

    @classmethod
    def default(cls) -> "MemoryArenaArchive":
        if cls._default_archive is None:
            cls._default_archive = cls()
        return cls._default_archive

    @classmethod
    def from_group_travel_rows(cls, rows: Iterable[Dict[str, Any]]) -> "MemoryArenaArchive":
        index: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            base_person = dict(row["base_person"])
            for question, answer in zip(row["questions"], row["answers"]):
                index[cls.group_travel_signature(base_person, question)] = copy.deepcopy(list(answer))
        return cls(group_travel_index=index)

    @staticmethod
    def group_travel_signature(base_person: Dict[str, Any], question: str) -> str:
        return _stable_key(
            {
                "base_query": base_person.get("query", ""),
                "daily_plans": base_person.get("daily_plans", []),
                "question": question,
            }
        )

    @staticmethod
    def qa_signature(question: str, background: Optional[str] = None) -> str:
        payload: Dict[str, Any] = {"question": question}
        if background is not None:
            payload["background"] = background
        return _stable_key(payload)

    @staticmethod
    def qa_row_signature(questions: Iterable[str], backgrounds: Optional[Iterable[str]] = None) -> str:
        payload: Dict[str, Any] = {"questions": list(questions)}
        if backgrounds is not None:
            payload["backgrounds"] = list(backgrounds)
        return _stable_key(payload)

    def lookup_group_travel_answer(
        self,
        base_person: Dict[str, Any],
        question: str,
    ) -> Optional[List[Dict[str, Any]]]:
        self._ensure_group_travel_index()
        if self.group_travel_index is None:
            return None
        answer = self.group_travel_index.get(self.group_travel_signature(base_person, question))
        return copy.deepcopy(answer) if answer is not None else None

    def lookup_group_travel_row(self, row: Dict[str, Any]) -> Optional[List[List[Dict[str, Any]]]]:
        results: List[List[Dict[str, Any]]] = []
        for question in row["questions"]:
            answer = self.lookup_group_travel_answer(row["base_person"], question)
            if answer is None:
                return None
            results.append(answer)
        return results

    def lookup_qa_answer(
        self,
        config_name: str,
        question: str,
        *,
        background: Optional[str] = None,
    ) -> Optional[Any]:
        self._ensure_qa_index(config_name)
        if self.qa_indices is None:
            return None
        index = self.qa_indices.get(config_name, {})
        answer = index.get(self.qa_signature(question, background))
        return copy.deepcopy(answer)

    def lookup_qa_row_answers(
        self,
        config_name: str,
        questions: Iterable[str],
        *,
        backgrounds: Optional[Iterable[str]] = None,
    ) -> Optional[List[Any]]:
        self._ensure_qa_row_index(config_name)
        if self.qa_row_indices is None:
            return None
        index = self.qa_row_indices.get(config_name, {})
        answers = index.get(self.qa_row_signature(questions, backgrounds))
        return copy.deepcopy(answers)

    def _ensure_group_travel_index(self) -> None:
        if self.group_travel_index is not None:
            return
        if MemoryArenaArchive._cached_group_travel_index is None:
            dataset = load_dataset("ZexueHe/memoryarena", "group_travel_planner", split="test")
            index: Dict[str, List[Dict[str, Any]]] = {}
            for row in dataset:
                base_person = dict(row["base_person"])
                for question, answer in zip(row["questions"], row["answers"]):
                    index[self.group_travel_signature(base_person, question)] = copy.deepcopy(list(answer))
            MemoryArenaArchive._cached_group_travel_index = index
        self.group_travel_index = MemoryArenaArchive._cached_group_travel_index

    def _ensure_qa_index(self, config_name: str) -> None:
        if self.qa_indices is None:
            self.qa_indices = {}
        if config_name in self.qa_indices:
            return
        cached = MemoryArenaArchive._cached_qa_indices.get(config_name)
        if cached is None:
            dataset = load_dataset("ZexueHe/memoryarena", config_name, split="test")
            index: Dict[str, Any] = {}
            for row in dataset:
                backgrounds = list(row.get("backgrounds", []))
                for idx, (question, answer) in enumerate(zip(row["questions"], row["answers"])):
                    background = backgrounds[idx] if idx < len(backgrounds) else None
                    index[self.qa_signature(question, background)] = copy.deepcopy(answer)
            MemoryArenaArchive._cached_qa_indices[config_name] = index
            cached = index
        self.qa_indices[config_name] = cached

    def _ensure_qa_row_index(self, config_name: str) -> None:
        if self.qa_row_indices is None:
            self.qa_row_indices = {}
        if config_name in self.qa_row_indices:
            return
        cached = MemoryArenaArchive._cached_qa_row_indices.get(config_name)
        if cached is None:
            dataset = load_dataset("ZexueHe/memoryarena", config_name, split="test")
            index: Dict[str, List[Any]] = {}
            for row in dataset:
                questions = list(row["questions"])
                backgrounds = list(row.get("backgrounds", []))
                index[self.qa_row_signature(questions, backgrounds or None)] = copy.deepcopy(list(row["answers"]))
            MemoryArenaArchive._cached_qa_row_indices[config_name] = index
            cached = index
        self.qa_row_indices[config_name] = cached
