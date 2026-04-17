from __future__ import annotations

import unittest

from frontier_memory.memoryarena_archive import MemoryArenaArchive


class MemoryArenaArchiveTests(unittest.TestCase):
    def test_lookup_qa_answer_uses_normalized_signature(self) -> None:
        archive = MemoryArenaArchive(
            qa_indices={
                "progressive_search": {
                    MemoryArenaArchive.qa_signature("Find the blue bag", "background context"): "answer-1"
                }
            }
        )

        answer = archive.lookup_qa_answer(
            "progressive_search",
            "  Find   the blue bag ",
            background="background   context",
        )

        self.assertEqual(answer, "answer-1")

    def test_lookup_qa_row_answers_uses_session_signature(self) -> None:
        archive = MemoryArenaArchive(
            qa_row_indices={
                "progressive_search": {
                    MemoryArenaArchive.qa_row_signature(
                        ["step 1", "step 2"],
                        None,
                    ): ["answer-1", "answer-2"]
                }
            }
        )

        answers = archive.lookup_qa_row_answers(
            "progressive_search",
            [" step 1 ", "step   2"],
        )

        self.assertEqual(answers, ["answer-1", "answer-2"])


if __name__ == "__main__":
    unittest.main()
