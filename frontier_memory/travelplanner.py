from __future__ import annotations

import importlib
import json
import os
import random
import sys
from ast import literal_eval
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from datasets import load_dataset


TRAVELPLANNER_ENV = "PRISM_TRAVELPLANNER_ROOT"
TRAVELPLANNER_VALIDATION_ENV = "PRISM_TRAVELPLANNER_VALIDATION_PATH"
EXTERNAL_TRAVELPLANNER_ROOT = Path(__file__).resolve().parents[1] / "external" / "TravelPlanner"
FIXTURE_TRAVELPLANNER_ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "travelplanner_root"


def resolve_travelplanner_root() -> Path:
    env_root = os.getenv(TRAVELPLANNER_ENV)
    if env_root:
        return Path(env_root)
    external_db = EXTERNAL_TRAVELPLANNER_ROOT / "database" / "database"
    if external_db.exists():
        return EXTERNAL_TRAVELPLANNER_ROOT
    return FIXTURE_TRAVELPLANNER_ROOT


def _extract_before_parenthesis(text: str) -> str:
    if "(" in text:
        return text.split("(", 1)[0].strip()
    return text.strip()


def _city_from_item(text: str) -> Tuple[str, str]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) < 2:
        return text.strip(), "-"
    return ",".join(parts[:-1]).strip(), _extract_before_parenthesis(parts[-1])


def _as_dict_local_constraint(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return literal_eval(value)


@dataclass
class TravelPlannerPolicy:
    policy_id: str = "balanced_v0"
    city_pool_size: int = 8
    city_cost_weight: float = 1.0
    city_transport_weight: float = 0.7
    city_restaurant_weight: float = 0.15
    city_attraction_weight: float = 0.1
    city_cuisine_weight: float = 0.8
    restaurant_rating_weight: float = 2.0
    required_cuisine_bonus: float = 40.0
    nonrequired_meal_penalty: float = 0.0
    dinner_on_travel_days: bool = True
    breakfast_on_final_day: bool = False
    top_k_sequences: int = 32
    prefer_flight_multiplier: float = 1.0
    transport_mode: str = "auto"
    travel_dinner_policy: str = "adaptive"

    def summary(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "city_pool_size": self.city_pool_size,
            "city_cost_weight": self.city_cost_weight,
            "city_transport_weight": self.city_transport_weight,
            "city_restaurant_weight": self.city_restaurant_weight,
            "city_attraction_weight": self.city_attraction_weight,
            "city_cuisine_weight": self.city_cuisine_weight,
            "restaurant_rating_weight": self.restaurant_rating_weight,
            "required_cuisine_bonus": self.required_cuisine_bonus,
            "prefer_flight_multiplier": self.prefer_flight_multiplier,
            "transport_mode": self.transport_mode,
            "travel_dinner_policy": self.travel_dinner_policy,
        }


@dataclass
class TransportChoice:
    mode: str
    text: str
    cost: float


@dataclass
class CityBlockPlan:
    city: str
    arrival_transport: TransportChoice
    accommodation_text: str
    accommodation_cost_per_night: float
    arrival_dinner: str
    breakfast: str
    lunch: str
    dinner: str
    stay_attraction: str


@dataclass
class TravelPlanResult:
    query_index: int
    query: Dict[str, Any]
    plan: List[Dict[str, Any]]
    estimated_cost: float
    policy_id: str
    error: Optional[str] = None


@dataclass
class TravelEvaluationResult:
    policy: TravelPlannerPolicy
    scores: Dict[str, float]
    objective: float
    output_path: Optional[str] = None
    sample_plans: List[TravelPlanResult] = field(default_factory=list)
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)


class TravelPlannerDB:
    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or resolve_travelplanner_root()
        nested_root = self.root / "database" / "database"
        self.database_root = nested_root if nested_root.exists() else self.root / "database"
        self.flights = pd.read_csv(self.database_root / "flights" / "clean_Flights_2022.csv").dropna()
        self.restaurants = pd.read_csv(self.database_root / "restaurants" / "clean_restaurant_2022.csv").dropna()
        self.accommodations = pd.read_csv(self.database_root / "accommodations" / "clean_accommodations_2022.csv").dropna()
        self.attractions = pd.read_csv(self.database_root / "attractions" / "attractions.csv").dropna()
        self.distances = pd.read_csv(self.database_root / "googleDistanceMatrix" / "distance.csv")
        pairs = [
            line.strip().split("\t")
            for line in (self.database_root / "background" / "citySet_with_states.txt").read_text().splitlines()
            if line.strip()
        ]
        self.city_to_state = {city: state for city, state in pairs}
        self.state_to_cities: Dict[str, List[str]] = {}
        for city, state in pairs:
            self.state_to_cities.setdefault(state, []).append(city)
        self._restaurants_empty = self.restaurants.iloc[0:0]
        self._attractions_empty = self.attractions.iloc[0:0]
        self._accommodations_empty = self.accommodations.iloc[0:0]
        self._restaurants_by_city = {
            str(city): frame.reset_index(drop=True)
            for city, frame in self.restaurants.groupby("City", sort=False)
        }
        self._attractions_by_city = {
            str(city): frame.reset_index(drop=True)
            for city, frame in self.attractions.groupby("City", sort=False)
        }
        self._accommodations_by_city = {
            str(city): frame.reset_index(drop=True)
            for city, frame in self.accommodations.groupby("city", sort=False)
        }
        self._flights_by_route_date = {
            (str(origin), str(destination), str(date)): frame.reset_index(drop=True)
            for (origin, destination, date), frame in self.flights.groupby(
                ["OriginCityName", "DestCityName", "FlightDate"],
                sort=False,
            )
        }
        self._distance_by_pair = {
            (str(row["origin"]), str(row["destination"])): (
                row["distance"],
                row["duration"],
            )
            for _, row in self.distances.iterrows()
        }

    def cities_in_state(self, state: str) -> List[str]:
        return list(self.state_to_cities.get(state, []))

    def restaurants_in_city(self, city: str) -> pd.DataFrame:
        return self._restaurants_by_city.get(city, self._restaurants_empty)

    def attractions_in_city(self, city: str) -> pd.DataFrame:
        return self._attractions_by_city.get(city, self._attractions_empty)

    def accommodations_in_city(self, city: str) -> pd.DataFrame:
        return self._accommodations_by_city.get(city, self._accommodations_empty)

    def flights_between(self, origin: str, destination: str, date: str) -> pd.DataFrame:
        return self._flights_by_route_date.get((origin, destination, date), self.flights.iloc[0:0])

    def _ground_cost(self, origin: str, destination: str, *, multiplier: float) -> Optional[Tuple[int, str]]:
        distance_info = self._distance_by_pair.get((origin, destination))
        if distance_info is None:
            return None
        distance, duration = distance_info
        if pd.isna(distance) or pd.isna(duration):
            return None
        if str(distance).strip().lower() == "nan" or str(duration).strip().lower() == "nan":
            return None
        if "day" in str(duration).lower():
            return None
        cost = int(eval(str(distance).replace("km", "").replace(",", "")) * multiplier)
        return cost, str(duration)

    def taxi_cost(self, origin: str, destination: str) -> Optional[Tuple[int, str]]:
        return self._ground_cost(origin, destination, multiplier=1.0)

    def self_driving_cost(self, origin: str, destination: str) -> Optional[Tuple[int, str]]:
        return self._ground_cost(origin, destination, multiplier=0.05)


class TravelPlannerOfficialEvaluator:
    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or resolve_travelplanner_root()
        self._commonsense_module = None
        self._hard_module = None

    def _ensure_modules(self) -> None:
        if self._commonsense_module is not None and self._hard_module is not None:
            return
        if not (self.root / "evaluation").exists():
            raise FileNotFoundError(
                "TravelPlanner evaluation files not found. "
                f"Set {TRAVELPLANNER_ENV} to a full TravelPlanner checkout if you want to run the official evaluator."
            )
        cwd = os.getcwd()
        sys.path.insert(0, str(self.root / "evaluation"))
        sys.path.insert(0, str(self.root))
        try:
            os.chdir(self.root / "evaluation")
            self._commonsense_module = importlib.import_module("commonsense_constraint")
            self._hard_module = importlib.import_module("hard_constraint")
        finally:
            os.chdir(cwd)

    def evaluate_plans(self, query_data_list: Sequence[Dict[str, Any]], plans: Sequence[List[Dict[str, Any]]]) -> Dict[str, float]:
        self._ensure_modules()
        commonsense_mod = self._commonsense_module
        hard_mod = self._hard_module

        delivery_cnt = 0
        commonsense_pass_total = 0
        hard_pass_total = 0
        commonsense_total = 0
        hard_total = 0
        final_commonsense_cnt = 0
        final_hard_cnt = 0
        final_all_cnt = 0

        for query_data, plan in zip(query_data_list, plans):
            tested_plan = {"plan": plan}
            if tested_plan["plan"]:
                delivery_cnt += 1
                commonsense_info = commonsense_mod.evaluation(query_data, tested_plan["plan"])
            else:
                commonsense_info = None

            if commonsense_info and commonsense_info["is_not_absent"][0] and commonsense_info["is_valid_information_in_sandbox"][0]:
                hard_info = hard_mod.evaluation(query_data, tested_plan["plan"])
            else:
                hard_info = None

            if commonsense_info:
                all_true = True
                for _, value in commonsense_info.items():
                    if value[0] is not None:
                        commonsense_total += 1
                        if value[0]:
                            commonsense_pass_total += 1
                        else:
                            all_true = False
                if all_true:
                    final_commonsense_cnt += 1

            if hard_info:
                all_true = True
                for _, value in hard_info.items():
                    if value[0] is not None:
                        hard_total += 1
                        if value[0]:
                            hard_pass_total += 1
                        else:
                            all_true = False
                if all_true:
                    final_hard_cnt += 1

            if commonsense_info and hard_info:
                commonsense_ok = all((value[0] is None or value[0] is True) for value in commonsense_info.values())
                hard_ok = all((value[0] is None or value[0] is True) for value in hard_info.values())
                if commonsense_ok and hard_ok:
                    final_all_cnt += 1

        total = len(query_data_list)
        return {
            "Delivery Rate": delivery_cnt / total,
            "Commonsense Constraint Micro Pass Rate": commonsense_pass_total / max(commonsense_total, 1),
            "Commonsense Constraint Macro Pass Rate": final_commonsense_cnt / total,
            "Hard Constraint Micro Pass Rate": hard_pass_total / max(hard_total, 1),
            "Hard Constraint Macro Pass Rate": final_hard_cnt / total,
            "Final Pass Rate": final_all_cnt / total,
        }


class TravelMemoryPlanner:
    def __init__(self, db: Optional[TravelPlannerDB] = None) -> None:
        self.db = db or TravelPlannerDB()

    @staticmethod
    def load_validation(limit: Optional[int] = None) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        fixture_path = os.environ.get(TRAVELPLANNER_VALIDATION_ENV)
        if fixture_path:
            with Path(fixture_path).expanduser().open() as handle:
                for index, line in enumerate(handle):
                    if limit is not None and index >= limit:
                        break
                    row = json.loads(line)
                    row["local_constraint"] = _as_dict_local_constraint(row["local_constraint"])
                    if isinstance(row.get("date"), str):
                        row["date"] = literal_eval(row["date"])
                    rows.append(row)
            return rows

        dataset = load_dataset("osunlp/TravelPlanner", "validation")["validation"]
        for index, item in enumerate(dataset):
            if limit is not None and index >= limit:
                break
            row = dict(item)
            row["local_constraint"] = _as_dict_local_constraint(row["local_constraint"])
            if isinstance(row.get("date"), str):
                row["date"] = literal_eval(row["date"])
            rows.append(row)
        return rows

    def plan_dataset(self, rows: Sequence[Dict[str, Any]], policy: TravelPlannerPolicy) -> List[TravelPlanResult]:
        results: List[TravelPlanResult] = []
        for index, row in enumerate(rows):
            plan = self.plan_query(row, policy)
            results.append(
                TravelPlanResult(
                    query_index=index + 1,
                    query=row,
                    plan=plan,
                    estimated_cost=self.estimate_cost(plan, row),
                    policy_id=policy.policy_id,
                )
            )
        return results

    def plan_query(self, row: Dict[str, Any], policy: TravelPlannerPolicy) -> List[Dict[str, Any]]:
        best_plan: Optional[List[Dict[str, Any]]] = None
        best_rank: Optional[Tuple[bool, float, float, int]] = None
        last_error: Optional[Exception] = None

        for route_mode in self._route_mode_candidates(row, policy):
            try:
                sequence = self._choose_city_sequence(row, policy, route_mode)
                blocks = self._build_city_blocks(row, sequence, policy, route_mode)
                plan = self._assemble_plan(row, sequence, blocks, policy, route_mode)
                estimated_cost = self.estimate_cost(plan, row)
                rank = (
                    estimated_cost > row["budget"],
                    max(0.0, estimated_cost - row["budget"]),
                    estimated_cost,
                    0 if route_mode == "self_driving_only" else 1,
                )
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_plan = plan
            except ValueError as exc:
                last_error = exc

        if best_plan is None:
            raise ValueError(f"Unable to build plan for query: {row['query']}") from last_error
        return best_plan

    def estimate_cost(self, plan: List[Dict[str, Any]], row: Dict[str, Any]) -> float:
        total = 0.0
        people = row["people_number"]
        for unit in plan:
            transportation = unit["transportation"]
            if transportation and transportation != "-":
                if transportation.startswith("Flight Number:"):
                    flight_no = transportation.split("Flight Number: ", 1)[1].split(",", 1)[0].strip()
                    flight = self.db.flights[self.db.flights["Flight Number"] == flight_no]
                    if len(flight) > 0:
                        total += float(flight.iloc[0]["Price"]) * people
                elif transportation.startswith("Taxi,"):
                    origin, destination = self._extract_from_to(transportation)
                    cost_info = self.db.taxi_cost(origin, destination) if origin and destination else None
                    if cost_info is not None:
                        total += float(cost_info[0]) * max(1, (people + 3) // 4)
                elif transportation.startswith("Self-driving,"):
                    origin, destination = self._extract_from_to(transportation)
                    cost_info = self.db.self_driving_cost(origin, destination) if origin and destination else None
                    if cost_info is not None:
                        total += float(cost_info[0]) * max(1, (people + 4) // 5)

            for key in ("breakfast", "lunch", "dinner"):
                value = unit[key]
                if value and value != "-":
                    name, city = _city_from_item(value)
                    frame = self.db.restaurants
                    hit = frame[(frame["Name"] == name) & (frame["City"] == city)]
                    if len(hit) > 0:
                        total += float(hit.iloc[0]["Average Cost"]) * people

            accommodation = unit["accommodation"]
            if accommodation and accommodation != "-":
                name, city = _city_from_item(accommodation)
                frame = self.db.accommodations
                hit = frame[(frame["NAME"] == name) & (frame["city"] == city)]
                if len(hit) > 0:
                    occupancy = int(hit.iloc[0]["maximum occupancy"])
                    total += float(hit.iloc[0]["price"]) * max(1, (people + occupancy - 1) // occupancy)
        return total

    def _route_mode_candidates(self, row: Dict[str, Any], policy: TravelPlannerPolicy) -> List[str]:
        if policy.transport_mode != "auto":
            return [policy.transport_mode]
        modes = ["air_taxi"]
        if row["local_constraint"].get("transportation") != "no self-driving":
            modes.append("self_driving_only")
        return modes

    def _choose_city_sequence(self, row: Dict[str, Any], policy: TravelPlannerPolicy, route_mode: str) -> List[str]:
        visiting_city_number = row["visiting_city_number"]
        if visiting_city_number == 1:
            return [_extract_before_parenthesis(row["dest"])]

        state = row["dest"]
        candidate_cities = [city for city in self.db.cities_in_state(state) if city != row["org"]]
        ranked = []
        for city in candidate_cities:
            city_score = self._city_rank_score(city, row, policy, route_mode)
            if city_score is not None:
                ranked.append((city_score, city))
        ranked.sort(key=lambda item: item[0])
        score_by_city = {city: score for score, city in ranked}
        if len(ranked) < visiting_city_number:
            raise ValueError(f"Unable to find enough valid cities for {row['query']}")

        tried_pool_sizes: List[int] = []
        for pool_size in [
            policy.city_pool_size,
            max(policy.city_pool_size, 12),
            max(policy.city_pool_size, 16),
            len(ranked),
        ]:
            pool_size = min(len(ranked), pool_size)
            if pool_size in tried_pool_sizes:
                continue
            tried_pool_sizes.append(pool_size)
            pool = [city for _, city in ranked[:pool_size]]
            best_sequence = None
            best_score = float("inf")
            all_sequences = list(permutations(pool, visiting_city_number))
            all_sequences.sort(key=lambda sequence: sum(score_by_city[city] for city in sequence))
            candidate_sequences = list(all_sequences)
            if policy.top_k_sequences > 0:
                candidate_sequences = candidate_sequences[: policy.top_k_sequences]

            for sequence in candidate_sequences:
                seq_score = self._sequence_score(sequence, row, policy, route_mode)
                if seq_score < best_score:
                    best_score = seq_score
                    best_sequence = list(sequence)

            if best_sequence is None and len(candidate_sequences) < len(all_sequences):
                for sequence in all_sequences[len(candidate_sequences) :]:
                    seq_score = self._sequence_score(sequence, row, policy, route_mode)
                    if seq_score < best_score:
                        best_score = seq_score
                        best_sequence = list(sequence)

            if best_sequence is not None:
                return best_sequence

        raise ValueError(f"Unable to find a valid city sequence for {row['query']}")

    def _city_rank_score(
        self,
        city: str,
        row: Dict[str, Any],
        policy: TravelPlannerPolicy,
        route_mode: str,
    ) -> Optional[float]:
        accommodation = self._best_accommodation(city, row)
        if accommodation is None:
            return None
        restaurants = self.db.restaurants_in_city(city)
        attractions = self.db.attractions_in_city(city)
        if len(restaurants) < 3 or len(attractions) < 1:
            return None
        required_cuisines = set(row["local_constraint"]["cuisine"] or [])
        cuisine_coverage = 0
        if required_cuisines:
            city_cuisine_text = " ".join(restaurants["Cuisines"].astype(str).tolist())
            cuisine_coverage = sum(1 for cuisine in required_cuisines if cuisine in city_cuisine_text)
        transport_probe = self._transport_probe(row["org"], city, row["date"][0], row, policy, route_mode)
        if transport_probe is None and row["visiting_city_number"] == 1:
            return None
        accommodation_cost = accommodation["block_cost"]
        restaurant_cost = float(restaurants["Average Cost"].nsmallest(min(3, len(restaurants))).sum()) * row["people_number"]
        attraction_term = len(attractions)
        restaurant_term = len(restaurants)
        transport_cost = transport_probe.cost if transport_probe is not None else 0.0
        return (
            policy.city_cost_weight * accommodation_cost
            + policy.city_transport_weight * transport_cost
            + policy.city_restaurant_weight * (1.0 / max(restaurant_term, 1))
            + policy.city_attraction_weight * (1.0 / max(attraction_term, 1))
            - policy.city_cuisine_weight * cuisine_coverage
        )

    def _sequence_score(
        self,
        sequence: Sequence[str],
        row: Dict[str, Any],
        policy: TravelPlannerPolicy,
        route_mode: str,
    ) -> float:
        total = 0.0
        required_cuisines = set(row["local_constraint"]["cuisine"] or [])
        covered_cuisines: set[str] = set()
        prev = row["org"]
        travel_day_indexes = [0] + [2 * index for index in range(1, len(sequence))] + [2 * len(sequence)]

        for block_index, city in enumerate(sequence):
            date = row["date"][travel_day_indexes[block_index]]
            transport = self._transport_probe(prev, city, date, row, policy, route_mode)
            accommodation = self._best_accommodation(city, row)
            restaurants = self.db.restaurants_in_city(city)
            if transport is None or accommodation is None or len(restaurants) < 3:
                return float("inf")
            total += transport.cost + accommodation["block_cost"]
            covered_cuisines.update(
                cuisine
                for cuisine in required_cuisines
                if cuisine in " ".join(restaurants["Cuisines"].astype(str).tolist())
            )
            prev = city

        final_transport = self._transport_probe(sequence[-1], row["org"], row["date"][-1], row, policy, route_mode)
        if final_transport is None:
            return float("inf")
        total += final_transport.cost
        if required_cuisines and not required_cuisines.issubset(covered_cuisines):
            total += 10000.0
        return total

    def _build_city_blocks(
        self,
        row: Dict[str, Any],
        sequence: Sequence[str],
        policy: TravelPlannerPolicy,
        route_mode: str,
    ) -> List[CityBlockPlan]:
        required_cuisines = set(row["local_constraint"]["cuisine"] or [])
        remaining_cuisines = set(required_cuisines)
        blocks: List[CityBlockPlan] = []
        prev = row["org"]
        total_blocks = len(sequence)

        for block_index, city in enumerate(sequence):
            date = row["date"][2 * block_index]
            transport = self._choose_transport(prev, city, date, row, policy, route_mode)
            accommodation = self._best_accommodation(city, row)
            if transport is None or accommodation is None:
                raise ValueError(f"Unable to build block for {city}")
            include_arrival_dinner = self._should_include_arrival_dinner(
                remaining_cuisines=remaining_cuisines,
                remaining_blocks_after_current=total_blocks - block_index - 1,
                policy=policy,
            )
            meals, remaining_cuisines = self._choose_city_meals(
                city=city,
                remaining_cuisines=remaining_cuisines,
                people=row["people_number"],
                policy=policy,
                meal_slots=4 if include_arrival_dinner else 3,
            )
            attraction = self._choose_attraction(city, set())
            if include_arrival_dinner:
                arrival_dinner, breakfast, lunch, dinner = meals
            else:
                arrival_dinner = "-"
                breakfast, lunch, dinner = meals
            blocks.append(
                CityBlockPlan(
                    city=city,
                    arrival_transport=transport,
                    accommodation_text=accommodation["text"],
                    accommodation_cost_per_night=accommodation["nightly_cost"],
                    arrival_dinner=arrival_dinner,
                    breakfast=breakfast,
                    lunch=lunch,
                    dinner=dinner,
                    stay_attraction=attraction,
                )
            )
            prev = city

        if remaining_cuisines:
            # Retry with a looser meal assignment across chosen cities.
            blocks = []
            remaining_cuisines = set(required_cuisines)
            prev = row["org"]
            for block_index, city in enumerate(sequence):
                date = row["date"][2 * block_index]
                transport = self._choose_transport(prev, city, date, row, policy, route_mode)
                accommodation = self._best_accommodation(city, row)
                include_arrival_dinner = self._should_include_arrival_dinner(
                    remaining_cuisines=remaining_cuisines,
                    remaining_blocks_after_current=total_blocks - block_index - 1,
                    policy=policy,
                )
                meals, remaining_cuisines = self._choose_city_meals(
                    city=city,
                    remaining_cuisines=remaining_cuisines,
                    people=row["people_number"],
                    policy=policy,
                    force_coverage=True,
                    meal_slots=4 if include_arrival_dinner else 3,
                )
                attraction = self._choose_attraction(city, set())
                if include_arrival_dinner:
                    arrival_dinner, breakfast, lunch, dinner = meals
                else:
                    arrival_dinner = "-"
                    breakfast, lunch, dinner = meals
                blocks.append(
                    CityBlockPlan(
                        city=city,
                        arrival_transport=transport,
                        accommodation_text=accommodation["text"],
                        accommodation_cost_per_night=accommodation["nightly_cost"],
                        arrival_dinner=arrival_dinner,
                        breakfast=breakfast,
                        lunch=lunch,
                        dinner=dinner,
                        stay_attraction=attraction,
                    )
                )
                prev = city
        return blocks

    def _assemble_plan(
        self,
        row: Dict[str, Any],
        sequence: Sequence[str],
        blocks: Sequence[CityBlockPlan],
        policy: TravelPlannerPolicy,
        route_mode: str,
    ) -> List[Dict[str, Any]]:
        plan: List[Dict[str, Any]] = []
        used_attractions: set[str] = set()
        for block_index, block in enumerate(blocks):
            prev_city = row["org"] if block_index == 0 else sequence[block_index - 1]
            arrival_day = {
                "day": 2 * block_index + 1,
                "current_city": f"from {prev_city} to {block.city}",
                "transportation": block.arrival_transport.text,
                "breakfast": "-",
                "attraction": "-",
                "lunch": "-",
                "dinner": block.arrival_dinner,
                "accommodation": block.accommodation_text,
            }
            plan.append(arrival_day)
            stay_attraction = self._choose_attraction(block.city, used_attractions)
            used_attractions.add(stay_attraction)
            stay_day = {
                "day": 2 * block_index + 2,
                "current_city": block.city,
                "transportation": "-",
                "breakfast": block.breakfast,
                "attraction": stay_attraction,
                "lunch": block.lunch,
                "dinner": block.dinner,
                "accommodation": block.accommodation_text,
            }
            plan.append(stay_day)

        final_transport = self._choose_transport(sequence[-1], row["org"], row["date"][-1], row, policy, route_mode)
        final_day = {
            "day": row["days"],
            "current_city": f"from {sequence[-1]} to {row['org']}",
            "transportation": final_transport.text if final_transport is not None else "-",
            "breakfast": "-",
            "attraction": "-",
            "lunch": "-",
            "dinner": "-",
            "accommodation": "-",
        }
        if policy.breakfast_on_final_day:
            final_day["breakfast"] = self._choose_final_breakfast(sequence[-1], self._used_restaurants_from_plan(plan)) or "-"
        plan.append(final_day)
        return plan

    def _used_restaurants_from_plan(self, plan: Sequence[Dict[str, Any]]) -> set[str]:
        used = set()
        for unit in plan:
            for key in ("breakfast", "lunch", "dinner"):
                value = unit.get(key)
                if value and value != "-":
                    used.add(value)
        return used

    def _choose_city_meals(
        self,
        *,
        city: str,
        remaining_cuisines: set[str],
        people: int,
        policy: TravelPlannerPolicy,
        force_coverage: bool = False,
        meal_slots: int = 4,
    ) -> Tuple[List[str], set[str]]:
        frame = self.db.restaurants_in_city(city).copy()
        if len(frame) < meal_slots:
            raise ValueError(f"Not enough restaurants in {city}")

        frame["coverage"] = frame["Cuisines"].apply(
            lambda value: len([cuisine for cuisine in remaining_cuisines if cuisine in str(value)])
        )
        frame["score"] = (
            frame["Average Cost"].astype(float)
            - policy.restaurant_rating_weight * frame["Aggregate Rating"].astype(float)
            - policy.required_cuisine_bonus * frame["coverage"].astype(float)
        )
        chosen_rows = []
        used_names: set[str] = set()
        still_needed = set(remaining_cuisines)

        optimized_combo = self._optimize_meal_combination(frame, still_needed, meal_slots)
        if optimized_combo is not None:
            chosen_rows = list(optimized_combo)
            used_names = {str(row["Name"]) for row in chosen_rows}
            covered = set()
            for row in chosen_rows:
                cuisines = str(row["Cuisines"])
                for cuisine in still_needed:
                    if cuisine in cuisines:
                        covered.add(cuisine)
            still_needed -= covered

        if still_needed and not chosen_rows:
            coverage_sorted = frame.sort_values(
                by=["coverage", "Aggregate Rating", "Average Cost"],
                ascending=[False, False, True],
            )
            for _, row in coverage_sorted.iterrows():
                if row["Name"] in used_names:
                    continue
                cuisines = str(row["Cuisines"])
                if any(cuisine in cuisines for cuisine in still_needed):
                    chosen_rows.append(row)
                    used_names.add(str(row["Name"]))
                    for cuisine in list(still_needed):
                        if cuisine in cuisines:
                            still_needed.discard(cuisine)
                if len(chosen_rows) >= meal_slots or (not still_needed and len(chosen_rows) >= min(2, meal_slots)):
                    break

        fill_sorted = frame.sort_values(by=["score", "Average Cost"], ascending=[True, True])
        for _, row in fill_sorted.iterrows():
            if row["Name"] in used_names:
                continue
            chosen_rows.append(row)
            used_names.add(str(row["Name"]))
            if len(chosen_rows) >= meal_slots:
                break

        if force_coverage and still_needed:
            for cuisine in list(still_needed):
                matching = frame[frame["Cuisines"].astype(str).str.contains(cuisine, regex=False)]
                if len(matching) < 1:
                    continue
                row = matching.sort_values(by=["Average Cost", "Aggregate Rating"], ascending=[True, False]).iloc[0]
                if row["Name"] not in used_names:
                    chosen_rows[-1] = row
                    used_names.add(str(row["Name"]))
                still_needed.discard(cuisine)

        chosen_rows = chosen_rows[:meal_slots]
        if len(chosen_rows) < meal_slots:
            raise ValueError(f"Unable to choose {meal_slots} meals in {city}")
        texts = [f"{row['Name']}, {city}" for row in chosen_rows]
        return texts, still_needed

    def _optimize_meal_combination(
        self,
        frame: pd.DataFrame,
        required_cuisines: set[str],
        meal_slots: int,
    ) -> Optional[Sequence[pd.Series]]:
        if not required_cuisines or meal_slots < len(required_cuisines):
            return None

        candidate_rows: Dict[str, pd.Series] = {}
        cheapest = frame.sort_values(by=["Average Cost", "Aggregate Rating"], ascending=[True, False])

        for cuisine in required_cuisines:
            matching = cheapest[cheapest["Cuisines"].astype(str).str.contains(cuisine, regex=False)].head(6)
            for _, row in matching.iterrows():
                candidate_rows.setdefault(str(row["Name"]), row)

        for _, row in cheapest.head(max(8, meal_slots * 3)).iterrows():
            candidate_rows.setdefault(str(row["Name"]), row)

        candidates = list(candidate_rows.values())
        if len(candidates) < meal_slots:
            return None

        best_combo: Optional[Sequence[pd.Series]] = None
        best_rank: Optional[Tuple[float, float]] = None

        for combo in combinations(candidates, meal_slots):
            covered = set()
            total_cost = 0.0
            total_rating = 0.0
            for row in combo:
                cuisines = str(row["Cuisines"])
                total_cost += float(row["Average Cost"])
                total_rating += float(row["Aggregate Rating"])
                for cuisine in required_cuisines:
                    if cuisine in cuisines:
                        covered.add(cuisine)
            if not required_cuisines.issubset(covered):
                continue
            rank = (total_cost, -total_rating)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_combo = combo
        return best_combo

    def _should_include_arrival_dinner(
        self,
        *,
        remaining_cuisines: set[str],
        remaining_blocks_after_current: int,
        policy: TravelPlannerPolicy,
    ) -> bool:
        if policy.travel_dinner_policy == "always":
            return True
        if policy.travel_dinner_policy == "never":
            return False
        if not remaining_cuisines:
            return False
        remaining_nontravel_slots = 3 * (remaining_blocks_after_current + 1)
        return len(remaining_cuisines) > remaining_nontravel_slots

    def _choose_attraction(self, city: str, used_attractions: set[str]) -> str:
        frame = self.db.attractions_in_city(city)
        for _, row in frame.iterrows():
            text = f"{row['Name']}, {city};"
            if text not in used_attractions:
                return text
        raise ValueError(f"No attraction left in {city}")

    def _choose_final_breakfast(self, city: str, used_restaurants: set[str]) -> Optional[str]:
        frame = self.db.restaurants_in_city(city).sort_values(by=["Average Cost", "Aggregate Rating"], ascending=[True, False])
        for _, row in frame.iterrows():
            text = f"{row['Name']}, {city}"
            if text not in used_restaurants:
                return text
        return None

    def _best_accommodation(self, city: str, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        local = row["local_constraint"]
        return self._cached_best_accommodation(
            city,
            int(row["people_number"]),
            local.get("room type"),
            local.get("house rule"),
        )

    @lru_cache(maxsize=50000)
    def _cached_best_accommodation(
        self,
        city: str,
        people: int,
        room_type: Optional[str],
        house_rule: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        frame = self.db.accommodations_in_city(city)
        if len(frame) < 1:
            return None
        local = {
            "room type": room_type,
            "house rule": house_rule,
        }
        filtered = self._filter_accommodations(frame.copy(), local)
        if len(filtered) < 1:
            return None
        block_nights = 2
        filtered["rooms_needed"] = filtered["maximum occupancy"].apply(
            lambda value: max(1, (people + int(value) - 1) // int(value))
        )
        filtered = filtered[filtered["minimum nights"].astype(float) <= block_nights]
        if len(filtered) < 1:
            return None
        filtered["nightly_total"] = filtered["price"].astype(float) * filtered["rooms_needed"].astype(float)
        filtered["block_cost"] = filtered["nightly_total"] * block_nights
        chosen = filtered.sort_values(by=["block_cost", "review rate number"], ascending=[True, False]).iloc[0]
        return {
            "text": f"{chosen['NAME']}, {city}",
            "nightly_cost": float(chosen["nightly_total"]),
            "block_cost": float(chosen["block_cost"]),
        }

    def _filter_accommodations(self, frame: pd.DataFrame, local: Dict[str, Any]) -> pd.DataFrame:
        room_type = local.get("room type")
        if room_type == "entire room":
            frame = frame[frame["room type"] == "Entire home/apt"]
        elif room_type == "private room":
            frame = frame[frame["room type"] == "Private room"]
        elif room_type == "shared room":
            frame = frame[frame["room type"] == "Shared room"]
        elif room_type == "not shared room":
            frame = frame[frame["room type"] != "Shared room"]

        house_rule = local.get("house rule")
        if house_rule == "smoking":
            frame = frame[~frame["house_rules"].astype(str).str.contains("No smoking", regex=False)]
        elif house_rule == "parties":
            frame = frame[~frame["house_rules"].astype(str).str.contains("No parties", regex=False)]
        elif house_rule == "children under 10":
            frame = frame[~frame["house_rules"].astype(str).str.contains("No children under 10", regex=False)]
        elif house_rule == "visitors":
            frame = frame[~frame["house_rules"].astype(str).str.contains("No visitors", regex=False)]
        elif house_rule == "pets":
            frame = frame[~frame["house_rules"].astype(str).str.contains("No pets", regex=False)]
        return frame

    def _transport_probe(
        self,
        origin: str,
        destination: str,
        date: str,
        row: Dict[str, Any],
        policy: TravelPlannerPolicy,
        route_mode: str,
    ) -> Optional[TransportChoice]:
        return self._choose_transport(origin, destination, date, row, policy, route_mode)

    @lru_cache(maxsize=100000)
    def _cached_transport(
        self,
        origin: str,
        destination: str,
        date: str,
        route_mode: str,
        no_flight: bool,
        prefer_flight_multiplier: float,
    ) -> Optional[TransportChoice]:
        options: List[TransportChoice] = []
        if route_mode == "air_taxi":
            taxi = self.db.taxi_cost(origin, destination)
            if taxi is not None:
                taxi_cost, duration = taxi
                options.append(
                    TransportChoice(
                        mode="taxi",
                        text=f"Taxi, from {origin} to {destination}, Duration: {duration}, Cost: {taxi_cost}",
                        cost=float(taxi_cost),
                    )
                )
        if route_mode == "self_driving_only":
            self_driving = self.db.self_driving_cost(origin, destination)
            if self_driving is not None:
                driving_cost, duration = self_driving
                options.append(
                    TransportChoice(
                        mode="self-driving",
                        text=f"Self-driving, from {origin} to {destination}, Duration: {duration}, Cost: {driving_cost}",
                        cost=float(driving_cost),
                    )
                )
        if route_mode == "air_taxi" and not no_flight:
            flights = self.db.flights_between(origin, destination, date)
            if len(flights) > 0:
                best = flights.sort_values(by=["Price", "DepTime"], ascending=[True, True]).iloc[0]
                options.append(
                    TransportChoice(
                        mode="flight",
                        text=(
                            f"Flight Number: {best['Flight Number']}, from {origin} to {destination}, "
                            f"Departure Time: {best['DepTime']}, Arrival Time: {best['ArrTime']}"
                        ),
                        cost=float(best["Price"]) * prefer_flight_multiplier,
                    )
                )
        if not options:
            return None
        options.sort(key=lambda item: item.cost)
        chosen = options[0]
        if chosen.mode == "flight":
            return TransportChoice(mode="flight", text=chosen.text, cost=chosen.cost / prefer_flight_multiplier)
        return chosen

    def _choose_transport(
        self,
        origin: str,
        destination: str,
        date: str,
        row: Dict[str, Any],
        policy: TravelPlannerPolicy,
        route_mode: str,
    ) -> Optional[TransportChoice]:
        no_flight = row["local_constraint"].get("transportation") == "no flight"
        no_self_driving = row["local_constraint"].get("transportation") == "no self-driving"
        if route_mode == "self_driving_only" and no_self_driving:
            return None
        return self._cached_transport(origin, destination, date, route_mode, no_flight, policy.prefer_flight_multiplier)

    @staticmethod
    def _extract_from_to(text: str) -> Tuple[Optional[str], Optional[str]]:
        if "from " not in text or " to " not in text:
            return None, None
        after_from = text.split("from ", 1)[1]
        left, right = after_from.split(" to ", 1)
        destination = right.split(",", 1)[0].strip()
        return left.strip(), destination


def select_validation_rows(*, limit: int, seed: int = 42) -> List[Dict[str, Any]]:
    rows = TravelMemoryPlanner.load_validation()
    if limit >= len(rows):
        return rows

    rng = random.Random(seed)
    by_days: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_days.setdefault(int(row["days"]), []).append(row)

    selected: List[Dict[str, Any]] = []
    remaining = limit
    day_buckets = sorted(by_days)
    base = limit // len(day_buckets)
    extras = limit % len(day_buckets)

    for index, day in enumerate(day_buckets):
        bucket = list(by_days[day])
        rng.shuffle(bucket)
        take = min(len(bucket), base + (1 if index < extras else 0))
        selected.extend(bucket[:take])
        remaining -= take

    if remaining > 0:
        leftovers = [row for day in day_buckets for row in by_days[day] if row not in selected]
        rng.shuffle(leftovers)
        selected.extend(leftovers[:remaining])
    return selected


def _objective_from_scores(scores: Dict[str, float]) -> float:
    return (
        scores["Final Pass Rate"] * 1.0
        + scores["Hard Constraint Macro Pass Rate"] * 0.3
        + scores["Commonsense Constraint Macro Pass Rate"] * 0.2
        + scores["Hard Constraint Micro Pass Rate"] * 0.1
        + scores["Commonsense Constraint Micro Pass Rate"] * 0.1
    )


def evaluate_rows(
    policy: TravelPlannerPolicy,
    rows: Sequence[Dict[str, Any]],
    *,
    write_jsonl_path: Optional[str | Path] = None,
    include_diagnostics: bool = False,
    planner: Optional[TravelMemoryPlanner] = None,
    evaluator: Optional[TravelPlannerOfficialEvaluator] = None,
) -> TravelEvaluationResult:
    planner = planner or TravelMemoryPlanner()
    plan_results = []
    for index, row in enumerate(rows):
        error = None
        try:
            plan = planner.plan_query(row, policy)
            estimated_cost = planner.estimate_cost(plan, row)
        except Exception as exc:
            plan = []
            estimated_cost = float("inf")
            error = str(exc)
        plan_results.append(
            TravelPlanResult(
                query_index=index + 1,
                query=row,
                plan=plan,
                estimated_cost=estimated_cost,
                policy_id=policy.policy_id,
                error=error,
            )
        )

    if write_jsonl_path is not None:
        path = Path(write_jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for result in plan_results:
                handle.write(
                    json.dumps(
                        {
                            "idx": result.query_index,
                            "query": result.query["query"],
                            "plan": result.plan,
                        }
                    )
                    + "\n"
                )

    evaluator = evaluator or TravelPlannerOfficialEvaluator()
    scores = evaluator.evaluate_plans(
        query_data_list=rows,
        plans=[result.plan for result in plan_results],
    )
    diagnostics: List[Dict[str, Any]] = []
    if include_diagnostics:
        evaluator._ensure_modules()
        commonsense_mod = evaluator._commonsense_module
        hard_mod = evaluator._hard_module
        for result, row in zip(plan_results, rows):
            if not result.plan:
                diagnostics.append(
                    {
                        "query_index": result.query_index,
                        "days": row["days"],
                        "visiting_city_number": row["visiting_city_number"],
                        "budget": row["budget"],
                        "estimated_cost": result.estimated_cost,
                        "error": result.error,
                        "commonsense_failures": ["plan_generation_failed"],
                        "hard_failures": [],
                    }
                )
                continue
            commonsense_info = commonsense_mod.evaluation(row, result.plan)
            hard_info = None
            if commonsense_info["is_not_absent"][0] and commonsense_info["is_valid_information_in_sandbox"][0]:
                hard_info = hard_mod.evaluation(row, result.plan)
            diagnostics.append(
                {
                    "query_index": result.query_index,
                    "days": row["days"],
                    "visiting_city_number": row["visiting_city_number"],
                    "budget": row["budget"],
                    "estimated_cost": result.estimated_cost,
                    "error": result.error,
                    "commonsense_failures": [
                        key for key, value in commonsense_info.items() if value[0] is False
                    ],
                    "hard_failures": [
                        key for key, value in (hard_info or {}).items() if value[0] is False
                    ],
                }
            )
    return TravelEvaluationResult(
        policy=policy,
        scores=scores,
        objective=_objective_from_scores(scores),
        output_path=str(write_jsonl_path) if write_jsonl_path is not None else None,
        sample_plans=plan_results[:3],
        diagnostics=diagnostics,
    )


def evaluate_policy(
    policy: TravelPlannerPolicy,
    *,
    limit: Optional[int] = None,
    write_jsonl_path: Optional[str | Path] = None,
    include_diagnostics: bool = False,
) -> TravelEvaluationResult:
    rows = TravelMemoryPlanner.load_validation(limit=limit)
    return evaluate_rows(
        policy,
        rows,
        write_jsonl_path=write_jsonl_path,
        include_diagnostics=include_diagnostics,
    )
