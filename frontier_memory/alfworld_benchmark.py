from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import textworld
import textworld.gym

from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredExpert, AlfredInfos, AlfredTWEnv

from .utils import token_overlap_score


ALFWORLD_ROOT = Path.home() / ".cache" / "alfworld"
COMMON_SOURCE_PRIORS: Dict[str, List[str]] = {
    "apple": ["countertop", "fridge", "diningtable"],
    "lettuce": ["fridge", "countertop"],
    "tomato": ["countertop", "fridge"],
    "potato": ["countertop", "fridge"],
    "spatula": ["countertop", "drawer", "cabinet"],
    "plate": ["cabinet", "countertop", "shelf", "drawer"],
    "mug": ["coffeemachine", "countertop", "cabinet"],
}


@dataclass
class AlfworldPolicy:
    policy_id: str = "retrieval_v0"
    train_games: int = 64
    eval_games: int = 20
    max_steps: int = 50
    exact_match_bonus: float = 2.0
    prev_action_bonus: float = 0.5
    goal_weight: float = 1.0
    observation_weight: float = 2.0
    template_bonus: float = 0.75
    novelty_penalty: float = 0.15

    def summary(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "train_games": self.train_games,
            "eval_games": self.eval_games,
            "max_steps": self.max_steps,
            "goal_weight": self.goal_weight,
            "observation_weight": self.observation_weight,
            "template_bonus": self.template_bonus,
        }


@dataclass
class TrajectoryStep:
    goal: str
    task_family: str
    observation: str
    prev_action: str
    action: str
    step_index: int
    game_file: str


@dataclass
class GameOutcome:
    game_file: str
    goal: str
    task_family: str
    won: bool
    steps: int
    actions: List[str]


@dataclass
class AlfworldEvalResult:
    policy: AlfworldPolicy
    split: str
    num_games: int
    success_rate: float
    mean_steps: float
    outcomes: List[GameOutcome] = field(default_factory=list)


@dataclass
class GoalSpec:
    goal: str
    task_family: str
    object_name: str
    target_receptacle: str = ""
    required_state: str = ""
    required_appliance: str = ""
    count: int = 1


def _alfworld_config(*, num_train_games: int = -1, num_eval_games: int = -1) -> Dict[str, Any]:
    root = ALFWORLD_ROOT
    return {
        "dataset": {
            "data_path": str(root / "json_2.1.1" / "train"),
            "eval_id_data_path": str(root / "json_2.1.1" / "valid_seen"),
            "eval_ood_data_path": str(root / "json_2.1.1" / "valid_unseen"),
            "num_train_games": num_train_games,
            "num_eval_games": num_eval_games,
        },
        "logic": {
            "domain": str(root / "logic" / "alfred.pddl"),
            "grammar": str(root / "logic" / "alfred.twl2"),
        },
        "env": {
            "type": "AlfredTWEnv",
            "domain_randomization": False,
            "task_types": [1, 2, 3, 4, 5, 6],
            "expert_timeout_steps": 150,
            "expert_type": "handcoded",
            "goal_desc_human_anns_prob": 0.0,
        },
        "general": {
            "training_method": "dagger",
        },
        "dagger": {
            "training": {
                "max_nb_steps_per_episode": 50,
            },
        },
    }


def extract_goal(observation: str) -> str:
    marker = "Your task is to:"
    if marker not in observation:
        return ""
    return observation.split(marker, 1)[1].split("\n", 1)[0].strip().rstrip(".")


def task_family_from_goal(goal: str) -> str:
    return parse_goal_spec(goal).task_family


def parse_goal_spec(goal: str) -> GoalSpec:
    lowered = goal.lower()
    if lowered.startswith("put two "):
        match = re.search(r"put two ([a-z]+) in ([a-z]+)", lowered)
        if match:
            return GoalSpec(
                goal=goal,
                task_family="pick_two_obj_and_place",
                object_name=match.group(1),
                target_receptacle=match.group(2),
                count=2,
            )
    if lowered.startswith("examine ") and " with " in lowered:
        match = re.search(r"examine the ([a-z]+) with the ([a-z]+)", lowered)
        if match:
            return GoalSpec(
                goal=goal,
                task_family="look_at_obj_in_light",
                object_name=match.group(1),
                required_appliance=match.group(2),
            )
    if lowered.startswith("look at ") and " under the " in lowered:
        match = re.search(r"look at ([a-z]+) under the ([a-z]+)", lowered)
        if match:
            return GoalSpec(
                goal=goal,
                task_family="look_at_obj_in_light",
                object_name=match.group(1),
                required_appliance=match.group(2),
            )
    if "put a clean " in lowered:
        match = re.search(r"put a clean ([a-z]+) in ([a-z]+)", lowered)
        if match:
            return GoalSpec(
                goal=goal,
                task_family="pick_clean_then_place_in_recep",
                object_name=match.group(1),
                target_receptacle=match.group(2),
                required_state="clean",
                required_appliance="sinkbasin",
            )
    if lowered.startswith("clean ") and "put it in " in lowered:
        match = re.search(r"clean (?:some |a )?([a-z]+) and put it in ([a-z]+)", lowered)
        if match:
            return GoalSpec(
                goal=goal,
                task_family="pick_clean_then_place_in_recep",
                object_name=match.group(1),
                target_receptacle=match.group(2),
                required_state="clean",
                required_appliance="sinkbasin",
            )
    if "put a hot " in lowered:
        match = re.search(r"put a hot ([a-z]+) in ([a-z]+)", lowered)
        if match:
            return GoalSpec(
                goal=goal,
                task_family="pick_heat_then_place_in_recep",
                object_name=match.group(1),
                target_receptacle=match.group(2),
                required_state="hot",
                required_appliance="microwave",
            )
    if lowered.startswith("heat ") and "put it in " in lowered:
        match = re.search(r"heat (?:some |a )?([a-z]+) and put it in ([a-z]+)", lowered)
        if match:
            return GoalSpec(
                goal=goal,
                task_family="pick_heat_then_place_in_recep",
                object_name=match.group(1),
                target_receptacle=match.group(2),
                required_state="hot",
                required_appliance="microwave",
            )
    if "put a cool " in lowered:
        match = re.search(r"put a cool ([a-z]+) in ([a-z]+)", lowered)
        if match:
            return GoalSpec(
                goal=goal,
                task_family="pick_cool_then_place_in_recep",
                object_name=match.group(1),
                target_receptacle=match.group(2),
                required_state="cool",
                required_appliance="fridge",
            )
    if "cool " in lowered and "put it in " in lowered:
        match = re.search(r"cool (?:some |a )?([a-z]+) and put it in ([a-z]+)", lowered)
        if match:
            return GoalSpec(
                goal=goal,
                task_family="pick_cool_then_place_in_recep",
                object_name=match.group(1),
                target_receptacle=match.group(2),
                required_state="cool",
                required_appliance="fridge",
            )
    match = re.search(r"put (?:a |some )?([a-z]+) (?:in|on) ([a-z]+)", lowered)
    if match:
        return GoalSpec(
            goal=goal,
            task_family="pick_and_place_simple",
            object_name=match.group(1),
            target_receptacle=match.group(2),
        )
    return GoalSpec(goal=goal, task_family="pick_and_place_simple", object_name="")


def command_template(command: str) -> str:
    lowered = command.lower().strip()
    for prefix, template in [
        ("go to ", "go_to"),
        ("take ", "take"),
        ("move ", "move"),
        ("open ", "open"),
        ("close ", "close"),
        ("clean ", "clean"),
        ("heat ", "heat"),
        ("cool ", "cool"),
        ("use ", "use"),
    ]:
        if lowered.startswith(prefix):
            return template
    if lowered in {"look", "inventory", "help"}:
        return lowered
    return "other"


def _normalize_textworld_command(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _normalize_token(text: str) -> str:
    token = re.sub(r"[^a-z]", "", text.lower())
    if token.endswith("s") and len(token) > 3:
        token = token[:-1]
    return token


def _word_tokens(text: str) -> List[str]:
    return [token for token in (_normalize_token(part) for part in re.findall(r"[a-z]+", text.lower())) if token]


def _single_game_env(game_file: str, *, include_expert: bool, max_steps: int):
    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        extras=["gamefile", "expert_plan"] if include_expert else ["gamefile"],
    )
    wrappers: List[Any] = [AlfredDemangler(shuffle=False), AlfredInfos]
    if include_expert:
        wrappers.append(AlfredExpert("handcoded"))
    env_id = textworld.gym.register_games(
        [game_file],
        request_infos,
        batch_size=1,
        asynchronous=False,
        max_episode_steps=max_steps,
        wrappers=wrappers,
    )
    return textworld.gym.make(env_id)


def _task_family_from_game_file(game_file: str) -> str:
    return Path(game_file).parents[1].name.split("-", 1)[0]


def _select_training_games(game_files: Sequence[str], num_games: int) -> List[str]:
    files = list(game_files)
    if num_games < 0 or num_games >= len(files):
        return files

    grouped: Dict[str, List[str]] = {}
    family_order: List[str] = []
    for game_file in files:
        family = _task_family_from_game_file(game_file)
        if family not in grouped:
            grouped[family] = []
            family_order.append(family)
        grouped[family].append(game_file)

    interleaved: List[str] = []
    max_family_size = max(len(group) for group in grouped.values())
    for offset in range(max_family_size):
        for family in family_order:
            group = grouped[family]
            if offset < len(group):
                interleaved.append(group[offset])
            if len(interleaved) >= num_games:
                return interleaved
    return interleaved[:num_games]


def collect_expert_library(*, num_games: int, max_steps: int = 50) -> List[TrajectoryStep]:
    config = _alfworld_config(num_train_games=num_games, num_eval_games=1)
    env_wrapper = AlfredTWEnv(config, train_eval="train")
    entries: List[TrajectoryStep] = []

    for game_file in _select_training_games(env_wrapper.game_files, num_games):
        env = _single_game_env(game_file, include_expert=True, max_steps=max_steps)
        try:
            obs, infos = env.reset()
            goal = extract_goal(obs[0])
            task_family = task_family_from_goal(goal)
            prev_action = ""
            done = False
            step_index = 0
            while not done and step_index < max_steps:
                expert_plan = infos.get("extra.expert_plan", [[]])[0]
                if not expert_plan:
                    break
                action = expert_plan[0]
                entries.append(
                    TrajectoryStep(
                        goal=goal,
                        task_family=task_family,
                        observation=obs[0],
                        prev_action=prev_action,
                        action=action,
                        step_index=step_index,
                        game_file=game_file,
                    )
                )
                obs, scores, dones, infos = env.step([action])
                done = bool(dones[0])
                prev_action = action
                step_index += 1
        finally:
            env.close()
    return entries


class RetrievalProceduralMemoryAgent:
    def __init__(self, library: Sequence[TrajectoryStep], policy: AlfworldPolicy) -> None:
        self.library = list(library)
        self.policy = policy
        self.source_priors = self._build_source_priors(self.library)
        self.goal = ""
        self.goal_spec = GoalSpec(goal="", task_family="pick_and_place_simple", object_name="")
        self.task_family = ""
        self.prev_action = ""
        self.step_index = 0
        self.action_counts: Dict[str, int] = {}
        self.visited_locations: set[str] = set()
        self.opened_receptacles: set[str] = set()
        self.held_objects: List[str] = []
        self.discovered_sources: set[str] = set()
        self.processed_object = False
        self.placed_count = 0
        self.current_location = ""

    def reset(self, initial_observation: str) -> None:
        self.goal = extract_goal(initial_observation)
        self.goal_spec = parse_goal_spec(self.goal)
        self.task_family = self.goal_spec.task_family
        self.prev_action = ""
        self.step_index = 0
        self.action_counts = {}
        self.visited_locations = set()
        self.opened_receptacles = set()
        self.held_objects = []
        self.discovered_sources = set()
        self.processed_object = False
        self.placed_count = 0
        self.current_location = ""

    @staticmethod
    def _build_source_priors(library: Sequence[TrajectoryStep]) -> Dict[tuple[str, str], Dict[str, int]]:
        priors: Dict[tuple[str, str], Dict[str, int]] = {}
        for entry in library:
            lowered = entry.action.lower()
            if not lowered.startswith("take ") or " from " not in lowered:
                continue
            object_part = lowered.split("take ", 1)[1].split(" from ", 1)[0]
            source_part = lowered.split(" from ", 1)[1]
            object_tokens = _word_tokens(object_part)
            source_tokens = _word_tokens(source_part)
            if not object_tokens or not source_tokens:
                continue
            key = (entry.task_family, object_tokens[0])
            source_counts = priors.setdefault(key, {})
            source_counts[source_tokens[0]] = source_counts.get(source_tokens[0], 0) + 1
        return priors

    def act(self, observation: str, admissible_commands: Sequence[str]) -> str:
        if not admissible_commands:
            return "look"

        heuristic = self._heuristic_action(observation, admissible_commands)
        if heuristic is not None:
            return self._commit(heuristic)

        candidates = [entry for entry in self.library if entry.task_family == self.task_family]
        if not candidates:
            return self._commit(admissible_commands[0])

        scored_entries = sorted(
            candidates,
            key=lambda entry: self._entry_score(entry, observation),
            reverse=True,
        )

        normalized_admissible = {_normalize_textworld_command(cmd): cmd for cmd in admissible_commands}
        for entry in scored_entries[:50]:
            normalized_action = _normalize_textworld_command(entry.action)
            if normalized_action in normalized_admissible:
                action = normalized_admissible[normalized_action]
                if self.action_counts.get(action, 0) < 3:
                    return self._commit(action)

        best_action = max(
            admissible_commands,
            key=lambda cmd: self._fallback_score(cmd, scored_entries[:20]),
        )
        return self._commit(best_action)

    def _commit(self, action: str) -> str:
        self.prev_action = action
        self.step_index += 1
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        self._update_state(action)
        return action

    def _update_state(self, action: str) -> None:
        lowered = action.lower()
        if lowered.startswith("go to "):
            location = lowered.split("go to ", 1)[1]
            self.current_location = location
            self.visited_locations.add(location)
        elif lowered.startswith("open "):
            self.opened_receptacles.add(lowered.split("open ", 1)[1])
        elif lowered.startswith("take "):
            target = lowered.split("take ", 1)[1].split(" from ", 1)[0]
            self.held_objects.append(target)
            if (
                _normalize_token(target) == _normalize_token(self.goal_spec.object_name)
                and " from " in lowered
            ):
                self.discovered_sources.add(lowered.split(" from ", 1)[1])
        elif lowered.startswith("move "):
            move_payload = lowered.split("move ", 1)[1]
            target = move_payload.split(" to ", 1)[0]
            self.held_objects = [item for item in self.held_objects if item != target]
            destination = move_payload.split(" to ", 1)[1] if " to " in move_payload else ""
            if (
                _normalize_token(target) == _normalize_token(self.goal_spec.object_name)
                and (not self.goal_spec.target_receptacle or _normalize_token(self.goal_spec.target_receptacle) in _word_tokens(destination))
            ):
                self.placed_count += 1
        elif any(lowered.startswith(prefix) for prefix in ("clean ", "heat ", "cool ", "use ")):
            self.processed_object = True

    def _entry_score(self, entry: TrajectoryStep, observation: str) -> float:
        score = 0.0
        score += self.policy.goal_weight * token_overlap_score(self.goal, entry.goal)
        score += self.policy.observation_weight * token_overlap_score(observation, entry.observation)
        if self.prev_action and self.prev_action == entry.prev_action:
            score += self.policy.prev_action_bonus
        if entry.step_index == self.step_index:
            score += 0.25
        return score

    def _fallback_score(self, command: str, top_entries: Sequence[TrajectoryStep]) -> float:
        score = token_overlap_score(command, self.goal)
        template = command_template(command)
        if template in {"look", "help", "inventory"}:
            score -= 1.0
        if self.action_counts.get(command, 0) > 0:
            score -= self.policy.novelty_penalty * self.action_counts[command]
        for entry in top_entries:
            if command_template(entry.action) == template:
                score += self.policy.template_bonus
                score += 0.5 * token_overlap_score(command, entry.action)
        return score

    def _heuristic_action(self, observation: str, admissible_commands: Sequence[str]) -> Optional[str]:
        spec = self.goal_spec
        goal_object = _normalize_token(spec.object_name)
        target_receptacle = _normalize_token(spec.target_receptacle)
        appliance = _normalize_token(spec.required_appliance)
        holding_target = any(_normalize_token(item) == goal_object for item in self.held_objects)

        def contains_token(command: str, token: str) -> bool:
            return bool(token) and token in _word_tokens(command)

        def suffix_after(command: str, prefix: str) -> str:
            lowered = command.lower()
            if not lowered.startswith(prefix):
                return ""
            return lowered.split(prefix, 1)[1]

        def take_source(command: str) -> str:
            lowered = command.lower()
            if not lowered.startswith("take ") or " from " not in lowered:
                return ""
            return lowered.split(" from ", 1)[1]

        def current_matches(token: str) -> bool:
            return bool(token) and token in _word_tokens(self.current_location)

        def source_bias(command: str) -> int:
            priors = self.source_priors.get((self.task_family, goal_object), {})
            if not priors:
                return 0
            return max((count for token, count in priors.items() if contains_token(command, token)), default=0)

        def common_source_bias(command: str) -> int:
            priors = COMMON_SOURCE_PRIORS.get(goal_object, [])
            for rank, token in enumerate(priors):
                if contains_token(command, token):
                    return len(priors) - rank
            return 0

        def first_matching(commands: Sequence[str], predicate, *, max_repeats: int = 3) -> Optional[str]:
            for command in commands:
                if predicate(command) and self.action_counts.get(command, 0) < max_repeats:
                    return command
            return None

        # 1. Finish the task if the exact placement command is available.
        move_cmd = first_matching(
            admissible_commands,
            lambda cmd: command_template(cmd) == "move"
            and contains_token(cmd, goal_object)
            and contains_token(cmd, target_receptacle),
        )
        if move_cmd and (not spec.required_state or self.processed_object or spec.task_family == "pick_two_obj_and_place"):
            return move_cmd

        # 2. For light tasks, route to the lamp/tool and use it once the object is in hand.
        if spec.task_family == "look_at_obj_in_light" and holding_target:
            use_cmd = first_matching(admissible_commands, lambda cmd: command_template(cmd) == "use")
            if use_cmd:
                return use_cmd
            if spec.required_appliance and not current_matches(appliance):
                go_to_light = first_matching(
                    admissible_commands,
                    lambda cmd: command_template(cmd) == "go_to" and contains_token(cmd, appliance),
                )
                if go_to_light:
                    return go_to_light

        # 3. Apply the required state transformation once, then transition to placement.
        if holding_target and spec.required_appliance and not self.processed_object:
            if spec.required_appliance and not current_matches(appliance):
                go_to_appliance = first_matching(
                    admissible_commands,
                    lambda cmd: command_template(cmd) == "go_to" and contains_token(cmd, appliance),
                )
                if go_to_appliance:
                    return go_to_appliance
            needed_template = {
                "clean": "clean",
                "hot": "heat",
                "cool": "cool",
            }.get(spec.required_state, "use")
            state_cmd = first_matching(
                admissible_commands,
                lambda cmd: command_template(cmd) == needed_template and contains_token(cmd, goal_object),
            )
            if state_cmd:
                return state_cmd

        # 4. If holding the object, route toward the target receptacle and open it if needed.
        if holding_target:
            open_target = first_matching(
                admissible_commands,
                lambda cmd: command_template(cmd) == "open"
                and contains_token(cmd, target_receptacle)
                and suffix_after(cmd, "open ") not in self.opened_receptacles,
            )
            if open_target and current_matches(target_receptacle):
                return open_target
            go_to_target = first_matching(
                admissible_commands,
                lambda cmd: command_template(cmd) == "go_to"
                and contains_token(cmd, target_receptacle)
                and not current_matches(target_receptacle),
            )
            if go_to_target:
                return go_to_target
            if open_target:
                return open_target

        # 5. Take the target object when it is visible.
        take_cmd = first_matching(
            admissible_commands,
            lambda cmd: command_template(cmd) == "take"
            and contains_token(cmd, goal_object)
            and not (
                spec.count > 1
                and self.placed_count > 0
                and contains_token(take_source(cmd), target_receptacle)
            ),
        )
        if take_cmd and (spec.count == 1 or self.placed_count < spec.count):
            return take_cmd

        # 6. In multi-object tasks, revisit known source receptacles before exploring new ones.
        if spec.count > 1 and self.placed_count < spec.count and self.discovered_sources:
            revisit_source = first_matching(
                admissible_commands,
                lambda cmd: command_template(cmd) == "go_to"
                and suffix_after(cmd, "go to ") in self.discovered_sources
                and suffix_after(cmd, "go to ") != self.current_location,
            )
            if revisit_source:
                return revisit_source

        # 7. If the target object is mentioned in the current observation, open unopened containers first.
        if goal_object and goal_object in _normalize_token(observation):
            open_current = first_matching(
                admissible_commands,
                lambda cmd: command_template(cmd) == "open" and cmd.lower().split("open ", 1)[1] not in self.opened_receptacles,
            )
            if open_current:
                return open_current

        # 8. Search the current container before leaving for the next unexplored location.
        current_open = first_matching(
            admissible_commands,
            lambda cmd: command_template(cmd) == "open"
            and self.current_location
            and suffix_after(cmd, "open ") == self.current_location
            and suffix_after(cmd, "open ") not in self.opened_receptacles,
        )
        if current_open:
            return current_open

        # 9. Explore unseen locations, preferring source priors and goal-relevant destinations.
        unseen_go_to = [
            cmd
            for cmd in admissible_commands
            if command_template(cmd) == "go_to" and cmd.lower().split("go to ", 1)[1] not in self.visited_locations
        ]
        if unseen_go_to:
            prioritized = sorted(
                unseen_go_to,
                key=lambda cmd: (
                    source_bias(cmd),
                    common_source_bias(cmd),
                    contains_token(cmd, goal_object),
                    contains_token(cmd, target_receptacle),
                    contains_token(cmd, appliance),
                    self._fallback_score(cmd, []),
                ),
                reverse=True,
            )
            return prioritized[0]

        # 10. Open unseen containers before falling back to retrieval.
        unopened = [
            cmd
            for cmd in admissible_commands
            if command_template(cmd) == "open" and cmd.lower().split("open ", 1)[1] not in self.opened_receptacles
        ]
        if unopened:
            return max(unopened, key=lambda cmd: (source_bias(cmd), common_source_bias(cmd), self._fallback_score(cmd, [])))

        return None


def evaluate_retrieval_policy(
    policy: AlfworldPolicy,
    *,
    split: str,
    write_json_path: Optional[str | Path] = None,
) -> AlfworldEvalResult:
    library = collect_expert_library(num_games=policy.train_games, max_steps=policy.max_steps)
    agent = RetrievalProceduralMemoryAgent(library, policy)

    if split not in {"valid_seen", "valid_unseen"}:
        raise ValueError(f"Unsupported split: {split}")

    config = _alfworld_config(num_train_games=policy.train_games, num_eval_games=policy.eval_games)
    train_eval = "eval_in_distribution" if split == "valid_seen" else "eval_out_of_distribution"
    env_wrapper = AlfredTWEnv(config, train_eval=train_eval)
    outcomes: List[GameOutcome] = []

    for game_file in env_wrapper.game_files[: policy.eval_games]:
        env = _single_game_env(game_file, include_expert=False, max_steps=policy.max_steps)
        actions: List[str] = []
        try:
            obs, infos = env.reset()
            agent.reset(obs[0])
            done = False
            won = False
            steps = 0
            while not done and steps < policy.max_steps:
                admissible_commands = infos["admissible_commands"][0]
                action = agent.act(obs[0], admissible_commands)
                actions.append(action)
                obs, scores, dones, infos = env.step([action])
                done = bool(dones[0])
                won = bool(scores[0] > 0 or infos.get("won", [False])[0])
                steps += 1
                if won:
                    break
            outcomes.append(
                GameOutcome(
                    game_file=game_file,
                    goal=agent.goal,
                    task_family=agent.task_family,
                    won=won,
                    steps=steps,
                    actions=actions,
                )
            )
        finally:
            env.close()

    success_rate = sum(1 for outcome in outcomes if outcome.won) / max(1, len(outcomes))
    mean_steps = sum(outcome.steps for outcome in outcomes) / max(1, len(outcomes))
    result = AlfworldEvalResult(
        policy=policy,
        split=split,
        num_games=len(outcomes),
        success_rate=success_rate,
        mean_steps=mean_steps,
        outcomes=outcomes,
    )

    if write_json_path is not None:
        path = Path(write_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "policy_id": policy.policy_id,
                    "split": split,
                    "num_games": result.num_games,
                    "success_rate": result.success_rate,
                    "mean_steps": result.mean_steps,
                    "summary": policy.summary(),
                    "outcomes": [
                        {
                            "game_file": outcome.game_file,
                            "goal": outcome.goal,
                            "task_family": outcome.task_family,
                            "won": outcome.won,
                            "steps": outcome.steps,
                            "actions": outcome.actions,
                        }
                        for outcome in outcomes
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return result
