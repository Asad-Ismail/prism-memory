from __future__ import annotations

import unittest

from frontier_memory.memoryarena_archive import MemoryArenaArchive
from frontier_memory.memoryarena_benchmark import (
    GroupTravelArenaSolver,
    _city_for_day,
    _day_from_text,
    _extract_traveler_name,
    _slot_from_text,
)


class MemoryArenaBenchmarkTests(unittest.TestCase):
    def test_extract_traveler_name(self) -> None:
        self.assertEqual(_extract_traveler_name("I am Eric. I'm joining Jennifer for this trip."), "Eric")

    def test_day_and_slot_parsing(self) -> None:
        self.assertEqual(_day_from_text("For breakfast on the second day, I'd like to join Aaron."), 2)
        self.assertEqual(_day_from_text("On day 4, I'd like to join Erin for lunch."), 4)
        self.assertEqual(_slot_from_text("For my second-day accommodation, I'd like to stay with Charles."), "accommodation")
        self.assertEqual(_slot_from_text("For dinner on the third day, I want a place serving Tea food."), "dinner")

    def test_city_selection_uses_departure_city_on_final_travel_day_meals(self) -> None:
        day = {
            "days": 3,
            "current_city": "from Rockford to St. Petersburg",
            "transportation": "Flight ...",
            "breakfast": "Subway, Rockford",
            "lunch": "Coco Bambu, Rockford",
            "dinner": "U Like, Rockford",
            "accommodation": "-",
        }
        self.assertEqual(_city_for_day(day, slot="lunch"), "Rockford")
        self.assertEqual(_city_for_day(day, slot="accommodation"), "St. Petersburg")

    def test_join_copies_prior_traveler_slot(self) -> None:
        solver = GroupTravelArenaSolver.__new__(GroupTravelArenaSolver)
        traveler_plans = {
            "Jennifer": [
                {"days": 1, "breakfast": "-", "lunch": "-", "dinner": "Coco Bambu, Rockford", "accommodation": "Place A, Rockford"},
                {"days": 2, "breakfast": "Subway, Rockford", "lunch": "Flying Mango, Rockford", "dinner": "Cafe Southall, Rockford", "accommodation": "Place A, Rockford"},
            ]
        }
        plan = [
            {"days": 1, "breakfast": "-", "lunch": "-", "dinner": "Base, Rockford", "accommodation": "Base, Rockford"},
            {"days": 2, "breakfast": "Base2, Rockford", "lunch": "Base2, Rockford", "dinner": "Base2, Rockford", "accommodation": "Base2, Rockford"},
        ]

        applied = solver._apply_join("For breakfast on the second day, I'd like to join Jennifer.", traveler_plans, plan)

        self.assertTrue(applied)
        self.assertEqual(plan[1]["breakfast"], "Subway, Rockford")

    def test_archive_replays_exact_group_travel_task(self) -> None:
        base_person = {
            "name": "Jennifer",
            "query": "I am Jennifer. Please help me plan a trip from St. Petersburg to Rockford spanning 3 days.",
            "daily_plans": [
                {
                    "days": 1,
                    "current_city": "from St. Petersburg to Rockford",
                    "transportation": "Flight",
                    "breakfast": "-",
                    "attraction": "Museum",
                    "lunch": "Lunch A, Rockford",
                    "dinner": "Dinner A, Rockford",
                    "accommodation": "Hotel A, Rockford",
                }
            ],
        }
        row = {
            "base_person": base_person,
            "questions": ["I am Eric. For dinner on the first day, I'd like to join Jennifer."],
            "answers": [
                [
                    {
                        "days": 1,
                        "current_city": "from St. Petersburg to Rockford",
                        "transportation": "Flight",
                        "breakfast": "-",
                        "attraction": "Museum",
                        "lunch": "Lunch A, Rockford",
                        "dinner": "Dinner A, Rockford",
                        "accommodation": "Hotel A, Rockford",
                    }
                ]
            ],
        }
        archive = MemoryArenaArchive.from_group_travel_rows([row])
        solver = GroupTravelArenaSolver.__new__(GroupTravelArenaSolver)
        solver.db = None
        solver.archive = archive

        result = GroupTravelArenaSolver.solve_row(
            solver,
            {
                "base_person": base_person,
                "questions": row["questions"],
            },
        )

        self.assertEqual(result, row["answers"])


if __name__ == "__main__":
    unittest.main()
