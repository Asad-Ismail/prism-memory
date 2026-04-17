from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List

from .types import MemoryEvent, ProcedureMemory
from .utils import beta_mean, dedupe_preserve_order, tokenize, token_overlap_score


class ProceduralStore:
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.procedures: List[ProcedureMemory] = []
        self._seq = 0

    def reset(self) -> None:
        self.procedures.clear()
        self._seq = 0

    def learn_from_events(
        self,
        events: Iterable[MemoryEvent],
        *,
        min_evidence: int,
        use_failures: bool,
    ) -> None:
        grouped = defaultdict(list)
        ordered_events = [event for event in events if event.metadata or len(tokenize(event.text)) >= 5]
        for index in range(max(len(ordered_events) - 1, 0)):
            first = ordered_events[index]
            second = ordered_events[index + 1]
            signature_tokens = dedupe_preserve_order(tokenize(first.text)[:3] + tokenize(second.text)[:3])
            if not signature_tokens:
                continue
            signature = " ".join(signature_tokens[:4])
            grouped[signature].append((first, second))

        procedures: List[ProcedureMemory] = []
        for signature, pairs in grouped.items():
            if len(pairs) < min_evidence:
                continue
            self._seq += 1
            steps = dedupe_preserve_order(
                [pair[0].text.strip() for pair in pairs[:2]] + [pairs[0][1].text.strip()]
            )
            source_turns = [pair[0].turn_index for pair in pairs] + [pair[1].turn_index for pair in pairs]
            failures = 0
            constraints = []
            if use_failures:
                for first, second in pairs:
                    for event in (first, second):
                        if event.metadata.get("failure") or "failed" in event.text.lower():
                            failures += 1
                            constraints.append(event.text.strip())
            success_count = len(pairs)
            reliability = beta_mean(
                self.prior_alpha,
                self.prior_beta,
                success_count=success_count,
                failure_count=failures,
            )
            procedures.append(
                ProcedureMemory(
                    procedure_id=f"proc_{self._seq:04d}",
                    name=f"playbook:{signature}",
                    trigger_terms=signature.split(),
                    steps=steps,
                    source_turns=sorted(set(source_turns)),
                    success_count=success_count,
                    failure_count=failures,
                    reliability=reliability,
                    negative_constraints=constraints[:3],
                )
            )
        self.procedures = procedures

    def split_overbroad(self, prune_threshold: float) -> None:
        refined: List[ProcedureMemory] = []
        for procedure in self.procedures:
            if procedure.reliability < prune_threshold:
                continue
            if len(procedure.steps) <= 4:
                refined.append(procedure)
                continue
            midpoint = len(procedure.steps) // 2
            for index, chunk in enumerate((procedure.steps[:midpoint], procedure.steps[midpoint:]), start=1):
                if not chunk:
                    continue
                refined.append(
                    ProcedureMemory(
                        procedure_id=f"{procedure.procedure_id}_part{index}",
                        name=f"{procedure.name}:part{index}",
                        trigger_terms=procedure.trigger_terms,
                        steps=chunk,
                        source_turns=procedure.source_turns,
                        success_count=procedure.success_count,
                        failure_count=procedure.failure_count,
                        reliability=procedure.reliability,
                        negative_constraints=procedure.negative_constraints,
                    )
                )
        self.procedures = refined

    def retrieve(self, query: str, top_k: int) -> List[ProcedureMemory]:
        scored = []
        for procedure in self.procedures:
            payload = " ".join(procedure.trigger_terms + procedure.steps + procedure.negative_constraints)
            score = token_overlap_score(query, payload)
            score += procedure.reliability * 0.25
            if score > 0:
                scored.append((score, procedure))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [procedure for _, procedure in scored[:top_k]]
