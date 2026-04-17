from __future__ import annotations

import unittest
from pathlib import Path

try:
    from frontier_memory.alfworld_benchmark import (
        AlfworldPolicy,
        RetrievalProceduralMemoryAgent,
        _select_training_games,
        parse_goal_spec,
    )
    ALFWORLD_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on optional deps.
    ALFWORLD_IMPORT_ERROR = exc


def _goal_observation(goal: str) -> str:
    return f"-= Welcome to TextWorld, ALFRED! =-\n\nYour task is to: {goal}.\n"


@unittest.skipIf(ALFWORLD_IMPORT_ERROR is not None, f"optional ALFWorld deps unavailable: {ALFWORLD_IMPORT_ERROR}")
class AlfworldBenchmarkTests(unittest.TestCase):
    def test_select_training_games_interleaves_families(self) -> None:
        files = [
            str(Path("/tmp/train/pick_and_place_simple-Apple-x-x/trial_1/game.tw-pddl")),
            str(Path("/tmp/train/pick_and_place_simple-Banana-x-x/trial_2/game.tw-pddl")),
            str(Path("/tmp/train/pick_heat_then_place_in_recep-Mug-x-x/trial_3/game.tw-pddl")),
            str(Path("/tmp/train/pick_heat_then_place_in_recep-Pan-x-x/trial_4/game.tw-pddl")),
        ]

        selected = _select_training_games(files, 3)

        self.assertEqual(selected[0], files[0])
        self.assertEqual(selected[1], files[2])
        self.assertEqual(selected[2], files[1])

    def test_parse_goal_spec_handles_alternative_state_goal_phrasings(self) -> None:
        cool_goal = parse_goal_spec("put a cool apple in microwave")
        self.assertEqual(cool_goal.task_family, "pick_cool_then_place_in_recep")
        self.assertEqual(cool_goal.required_appliance, "fridge")

        clean_goal = parse_goal_spec("clean some tomato and put it in countertop")
        self.assertEqual(clean_goal.task_family, "pick_clean_then_place_in_recep")
        self.assertEqual(clean_goal.required_appliance, "sinkbasin")

        light_goal = parse_goal_spec("look at alarmclock under the desklamp")
        self.assertEqual(light_goal.task_family, "look_at_obj_in_light")
        self.assertEqual(light_goal.required_appliance, "desklamp")

    def test_processed_object_routes_to_target_instead_of_repeating_state_change(self) -> None:
        agent = RetrievalProceduralMemoryAgent([], AlfworldPolicy())
        agent.reset(_goal_observation("put a hot mug in cabinet"))
        agent.held_objects = ["mug 1"]
        agent.processed_object = True
        agent.current_location = "microwave 1"

        action = agent.act(
            "You heat the mug 1 using the microwave 1.",
            [
                "heat mug 1 with microwave 1",
                "go to cabinet 1",
                "go to cabinet 2",
            ],
        )

        self.assertEqual(action, "go to cabinet 1")

    def test_target_matching_does_not_confuse_toilet_with_toiletpaperhanger(self) -> None:
        agent = RetrievalProceduralMemoryAgent([], AlfworldPolicy())
        agent.reset(_goal_observation("put two toiletpaper in toilet"))
        agent.held_objects = ["toiletpaper 4"]
        agent.current_location = "toiletpaperhanger 1"

        action = agent.act(
            "You are carrying the toiletpaper 4.",
            [
                "move toiletpaper 4 to toiletpaperhanger 1",
                "go to toilet 1",
            ],
        )

        self.assertEqual(action, "go to toilet 1")

    def test_agent_opens_current_container_before_exploring_elsewhere(self) -> None:
        agent = RetrievalProceduralMemoryAgent([], AlfworldPolicy())
        agent.reset(_goal_observation("put some candle on toilet"))
        agent.current_location = "cabinet 1"
        agent.visited_locations = {"cabinet 1"}

        action = agent.act(
            "You arrive at cabinet 1. The cabinet 1 is closed.",
            [
                "open cabinet 1",
                "go to shelf 1",
            ],
        )

        self.assertEqual(action, "open cabinet 1")

    def test_multi_object_task_revisits_known_source_instead_of_retaking_from_target(self) -> None:
        agent = RetrievalProceduralMemoryAgent([], AlfworldPolicy())
        agent.reset(_goal_observation("put two toiletpaper in toilet"))
        agent.placed_count = 1
        agent.current_location = "toilet 1"
        agent.discovered_sources = {"cabinet 1"}

        action = agent.act(
            "You arrive at toilet 1.",
            [
                "take toiletpaper 2 from toilet 1",
                "go to cabinet 1",
            ],
        )

        self.assertEqual(action, "go to cabinet 1")

    def test_common_source_prior_prefers_countertop_for_food_search(self) -> None:
        agent = RetrievalProceduralMemoryAgent([], AlfworldPolicy())
        agent.reset(_goal_observation("put a cool apple in microwave"))
        agent.visited_locations = {"microwave 1", "fridge 1"}

        action = agent.act(
            "You are in the kitchen.",
            [
                "go to cabinet 1",
                "go to countertop 1",
            ],
        )

        self.assertEqual(action, "go to countertop 1")


if __name__ == "__main__":
    unittest.main()
