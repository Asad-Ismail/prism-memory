from __future__ import annotations

import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "travelplanner_validation_fixture.jsonl"
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "travelplanner_root"
os.environ.setdefault("PRISM_TRAVELPLANNER_VALIDATION_PATH", str(FIXTURE_PATH))
os.environ.setdefault("PRISM_TRAVELPLANNER_ROOT", str(FIXTURE_ROOT))

from frontier_memory.travelplanner import (
    TravelMemoryPlanner,
    TravelPlannerPolicy,
    select_validation_rows,
)
from frontier_memory.travelplanner_search import dump_policy, load_policy


class TravelPlannerTests(unittest.TestCase):
    def test_select_validation_rows_is_stratified(self) -> None:
        rows = select_validation_rows(limit=9, seed=7)
        counts = Counter(row["days"] for row in rows)
        self.assertEqual(counts[3], 3)
        self.assertEqual(counts[5], 3)
        self.assertEqual(counts[7], 3)

    def test_policy_roundtrip(self) -> None:
        policy = TravelPlannerPolicy(
            policy_id="roundtrip",
            city_pool_size=11,
            prefer_flight_multiplier=0.9,
            transport_mode="self_driving_only",
            travel_dinner_policy="never",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "policy.yaml"
            dump_policy(policy, path)
            loaded = load_policy(path)
        self.assertEqual(loaded.policy_id, policy.policy_id)
        self.assertEqual(loaded.city_pool_size, 11)
        self.assertEqual(loaded.transport_mode, "self_driving_only")
        self.assertEqual(loaded.travel_dinner_policy, "never")

    def test_plan_query_matches_day_structure(self) -> None:
        planner = TravelMemoryPlanner()
        row = TravelMemoryPlanner.load_validation(limit=1)[0]
        policy = TravelPlannerPolicy(policy_id="shape_smoke")
        plan = planner.plan_query(row, policy)
        self.assertEqual(len(plan), row["days"])
        self.assertIn("from ", plan[0]["current_city"])
        self.assertIn("from ", plan[-1]["current_city"])
        self.assertNotEqual(plan[0]["transportation"], "-")
        self.assertNotEqual(plan[-1]["transportation"], "-")


if __name__ == "__main__":
    unittest.main()
