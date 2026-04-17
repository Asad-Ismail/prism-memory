from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryEvent:
    speaker: str
    text: str
    turn_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    salience: float = 0.0


@dataclass
class SemanticFact:
    fact_id: str
    subject: str
    relation: str
    value: str
    turn_index: int
    source_text: str
    speaker: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    answer_hint: Optional[str] = None
    confidence: float = 1.0
    active: bool = True
    is_dynamic: bool = False
    superseded_by: Optional[str] = None
    valid_from_turn: Optional[int] = None
    valid_to_turn: Optional[int] = None
    aliases: List[str] = field(default_factory=list)


@dataclass
class ProcedureMemory:
    procedure_id: str
    name: str
    trigger_terms: List[str]
    steps: List[str]
    source_turns: List[int]
    success_count: int = 0
    failure_count: int = 0
    reliability: float = 0.5
    negative_constraints: List[str] = field(default_factory=list)


@dataclass
class QueryPlan:
    intent: str
    subject: Optional[str] = None
    relation_hints: List[str] = field(default_factory=list)
    chain_relations: List[str] = field(default_factory=list)
    terminal_hint: Optional[str] = None
    target_text: Optional[str] = None
    wants_boolean: bool = False
    wants_history: bool = False
    wants_current: bool = False


@dataclass
class BenchmarkResult:
    benchmark: str
    overall: float
    per_type: Dict[str, float]
    count: int
