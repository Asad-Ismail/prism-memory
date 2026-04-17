from __future__ import annotations

import unittest

from frontier_memory.config import load_candidate
from frontier_memory.memeval_adapter import (
    apply_memeval_overrides,
    compute_memeval_summary,
    dialogue_turns_to_events,
    normalize_memeval_prediction,
)


class MemEvalAdapterTests(unittest.TestCase):
    def test_dialogue_turns_to_events_preserves_metadata(self) -> None:
        events = dialogue_turns_to_events(
            [
                {
                    "speaker": "Alice",
                    "text": "I moved to Berlin.",
                    "dia_id": "session_2_4",
                    "timestamp": "1:56 pm on 8 May, 2023",
                }
            ]
        )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.turn_index, 1)
        self.assertEqual(event.metadata["session_id"], "session_2")
        self.assertEqual(event.metadata["timestamp"], "1:56 pm on 8 May, 2023")

    def test_normalize_memeval_prediction_maps_refusals_to_none(self) -> None:
        self.assertEqual(normalize_memeval_prediction("I don't know."), "None")
        self.assertEqual(normalize_memeval_prediction("Not mentioned in the chat"), "None")
        self.assertEqual(normalize_memeval_prediction("Berlin"), "Berlin")

    def test_apply_memeval_overrides_updates_openai_candidate(self) -> None:
        candidate = load_candidate("candidates/openai_memory_champion.yaml")
        tuned = apply_memeval_overrides(candidate)

        self.assertEqual(tuned.get("llm", "prompt_profile"), "memeval_short_answer")
        self.assertEqual(tuned.get("llm", "refusal_text"), "None")
        self.assertGreaterEqual(tuned.get("llm", "context", "episodic_top_k"), 8)

    def test_compute_memeval_summary_handles_judge_and_accuracy(self) -> None:
        summary = compute_memeval_summary(
            [
                {
                    "sample_id": "a",
                    "category_name": "Temporal",
                    "f1": 1.0,
                    "judge_relevant": 1,
                    "judge_complete": 1,
                    "judge_accurate": 1,
                    "longmemeval_correct": 1,
                },
                {
                    "sample_id": "b",
                    "category_name": "Temporal",
                    "f1": 0.5,
                    "judge_relevant": 0,
                    "judge_complete": 1,
                    "judge_accurate": 0,
                    "longmemeval_correct": 0,
                },
            ]
        )

        self.assertAlmostEqual(summary["overall_f1_mean"], 0.75)
        self.assertAlmostEqual(summary["judge"]["judge_pass_rate"], (0.5 + 1.0 + 0.5) / 3)
        self.assertAlmostEqual(summary["longmemeval_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
