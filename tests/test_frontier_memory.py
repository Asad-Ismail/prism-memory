from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from frontier_memory.benchmarks import run_synthetic_suite
from frontier_memory.config import load_candidate
from frontier_memory.search import run_search_iteration
from frontier_memory.system import HybridMemorySystem
from frontier_memory.synthetic_benchmarks import EntityChainBenchmark, TemporalDriftBenchmark


ROOT = Path(__file__).resolve().parents[1]


class FrontierMemoryTests(unittest.TestCase):
    def test_temporal_memory_handles_updates(self) -> None:
        candidate = load_candidate(ROOT / "candidates/bootstrap_v0.yaml")
        memory = HybridMemorySystem(candidate)
        example = TemporalDriftBenchmark(seed=11).generate_example(attribute="job", update_turn=15)
        for turn in example.conversation:
            memory.ingest(turn)

        answers = {qa.question_type: memory.answer(qa.question) for qa in example.qa_pairs}
        self.assertIn("Around turn 15", answers["transition_timing"])
        self.assertEqual(answers["current_state"], example.qa_pairs[0].answer)
        self.assertEqual(answers["historical"], example.qa_pairs[1].answer)

    def test_chain_memory_composes_relations(self) -> None:
        candidate = load_candidate(ROOT / "candidates/bootstrap_v0.yaml")
        memory = HybridMemorySystem(candidate)
        example = EntityChainBenchmark(seed=13).generate_example(depth=3)
        for turn in example.conversation:
            memory.ingest(turn)
        self.assertEqual(memory.answer(example.qa_pairs[0].question), example.qa_pairs[0].answer)
        self.assertEqual(memory.answer(example.qa_pairs[1].question), example.qa_pairs[1].answer)

    def test_synthetic_suite_smoke(self) -> None:
        result = run_synthetic_suite(ROOT / "candidates/bootstrap_v0.yaml", dataset_size=2, seed=5)
        self.assertGreaterEqual(result["global_score"], 0.85)
        self.assertIn("surprise_recall", result["benchmarks"])
        self.assertIn("temporal_drift", result["benchmarks"])

    def test_search_iteration_writes_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir) / "logs"
            generated_dir = Path(tmp_dir) / "generated"
            champion_path = Path(tmp_dir) / "champion.yaml"
            summary = run_search_iteration(
                ROOT / "candidates/bootstrap_v0.yaml",
                num_children=2,
                dataset_size=1,
                seed=3,
                logs_dir=log_dir,
                generated_dir=generated_dir,
                champion_path=champion_path,
            )
            self.assertIn("best_candidate_id", summary)
            self.assertTrue((log_dir / "experiments.jsonl").exists())
            self.assertTrue((log_dir / "journal.md").exists())
            run_files = list((log_dir / "runs").glob("*.json"))
            self.assertGreaterEqual(len(run_files), 1)
            self.assertTrue(champion_path.exists())
            with (log_dir / "experiments.jsonl").open() as handle:
                first_line = json.loads(handle.readline())
            self.assertEqual(first_line["event"], "search_iteration")
            self.assertGreaterEqual(len(list(generated_dir.glob("*.yaml"))), 2)

    def test_mock_llm_backend_can_override_heuristic_answer(self) -> None:
        candidate = load_candidate(ROOT / "candidates/bootstrap_v0.yaml").clone()
        candidate.set(("llm", "enabled"), True)
        candidate.set(("llm", "provider"), "mock")
        candidate.set(("llm", "mock_mode"), "fixed")
        candidate.set(("llm", "mock_answer"), "mock-final-answer")
        memory = HybridMemorySystem(candidate)
        example = TemporalDriftBenchmark(seed=11).generate_example(attribute="job", update_turn=15)
        for turn in example.conversation:
            memory.ingest(turn)

        answer = memory.answer(example.qa_pairs[0].question)

        self.assertEqual(answer, "mock-final-answer")


if __name__ == "__main__":
    unittest.main()
