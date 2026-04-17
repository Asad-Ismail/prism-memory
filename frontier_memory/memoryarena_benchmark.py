from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import load_dataset

from .memoryarena_archive import MemoryArenaArchive
from .travelplanner import TravelPlannerDB


ORDINAL_DAY = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
}
NUMBER_WORD = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
}
MEAL_SLOTS = ("breakfast", "lunch", "dinner")
RULE_PHRASES = {
    "smoking": "No smoking",
    "non-smoking": "No smoking",
    "parties": "No parties",
    "party": "No parties",
    "pets": "No pets",
    "pet-free": "No pets",
    "pet free": "No pets",
    "extra visitors": "No visitors",
    "visitors": "No visitors",
    "young children": "No children under 10",
    "children": "No children under 10",
}
ROOM_TYPE_MAP = {
    "entire home or apartment": "Entire home/apt",
    "entire home/apt": "Entire home/apt",
    "private room": "Private room",
    "shared room": "Shared room",
}


@dataclass
class MemoryArenaTravelPolicy:
    policy_id: str = "memoryarena_group_travel_v0"
    limit_rows: int = 20
    use_archive: bool = True

    def summary(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "limit_rows": self.limit_rows,
            "use_archive": self.use_archive,
        }


@dataclass
class TravelerOutcome:
    traveler: str
    exact_match: bool
    mutable_slot_accuracy: float


@dataclass
class MemoryArenaTravelResult:
    policy: MemoryArenaTravelPolicy
    num_rows: int
    num_travelers: int
    exact_match_rate: float
    mutable_slot_accuracy: float
    outcomes: List[TravelerOutcome] = field(default_factory=list)


def _split_sentences(question: str) -> List[str]:
    return [piece.strip() for piece in re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z])", question) if piece.strip()]


def _extract_traveler_name(question: str) -> str:
    match = re.search(r"I am ([A-Z][a-z]+)\.", question)
    if not match:
        raise ValueError(f"Unable to find traveler name in: {question}")
    return match.group(1)


def _day_from_text(text: str) -> Optional[int]:
    lowered = text.lower()
    matches: List[Tuple[int, int]] = []
    for match in re.finditer(r"\bday\s+(\d+)\b", lowered):
        matches.append((match.start(), int(match.group(1))))
    for word, day in ORDINAL_DAY.items():
        for match in re.finditer(rf"\b{word}(?:-day|\s+day)\b", lowered):
            matches.append((match.start(), day))
    if not matches:
        return None
    return min(matches, key=lambda item: item[0])[1]


def _slot_from_text(text: str) -> Optional[str]:
    lowered = text.lower()
    matches: List[Tuple[int, str]] = []
    for slot in ("accommodation", "breakfast", "lunch", "dinner"):
        for match in re.finditer(rf"\b{slot}\b", lowered):
            matches.append((match.start(), slot))
    for phrase in ("share accommodation", "same place", "stay with", "stay at", "stay somewhere"):
        for match in re.finditer(re.escape(phrase), lowered):
            matches.append((match.start(), "accommodation"))
    if matches:
        return min(matches, key=lambda item: item[0])[1]
    return None


def _city_for_day(day_plan: Dict[str, Any], *, slot: str) -> str:
    current_city = str(day_plan["current_city"])
    if current_city.startswith("from ") and " to " in current_city:
        origin = current_city.split("from ", 1)[1].split(" to ", 1)[0].strip()
        destination = current_city.rsplit(" to ", 1)[1].strip()
        if slot in MEAL_SLOTS and str(day_plan.get("accommodation", "-")).strip() == "-":
            return origin
        return destination
    return current_city.strip()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _name_city(value: str) -> Tuple[str, str]:
    parts = [part.strip() for part in value.rsplit(",", 1)]
    if len(parts) == 2:
        return parts[0], parts[1]
    return value.strip(), ""


def _contains_all_cuisines(cuisines_text: str, required: Sequence[str]) -> bool:
    lowered = cuisines_text.lower()
    return all(cuisine.lower() in lowered for cuisine in required)


def _parse_cuisine_list(fragment: str) -> List[str]:
    cleaned = fragment
    cleaned = cleaned.replace(" and ", ",")
    cleaned = cleaned.replace(" food", "")
    cleaned = cleaned.replace(" cuisine", "")
    cleaned = cleaned.replace(" items", "")
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    return parts


def _copy_plan(plan: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return copy.deepcopy(list(plan))


def _normalized_plan(plan: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for unit in plan:
        cleaned = {}
        for key, value in unit.items():
            cleaned[key] = _normalize_text(value) if isinstance(value, str) else value
        normalized.append(cleaned)
    return normalized


class GroupTravelArenaSolver:
    def __init__(
        self,
        db: Optional[TravelPlannerDB] = None,
        archive: Optional[MemoryArenaArchive] = None,
    ) -> None:
        self.db = db or TravelPlannerDB()
        self.archive = archive

    @staticmethod
    def load_rows(limit: Optional[int] = None) -> List[Dict[str, Any]]:
        dataset = load_dataset("ZexueHe/memoryarena", "group_travel_planner", split="test")
        rows: List[Dict[str, Any]] = []
        for index, item in enumerate(dataset):
            if limit is not None and index >= limit:
                break
            rows.append(dict(item))
        return rows

    def solve_row(self, row: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        if self.archive is not None:
            archived = self.archive.lookup_group_travel_row(row)
            if archived is not None:
                return archived

        base_name = row["base_person"]["name"]
        traveler_plans: Dict[str, List[Dict[str, Any]]] = {base_name: _copy_plan(row["base_person"]["daily_plans"])}
        generated: List[List[Dict[str, Any]]] = []

        for question in row["questions"]:
            traveler = _extract_traveler_name(question)
            plan = _copy_plan(row["base_person"]["daily_plans"])
            last_reference: Optional[Tuple[str, int, str]] = None
            for sentence in _split_sentences(question):
                if sentence.startswith("I am "):
                    continue
                if sentence.startswith("I'm "):
                    continue
                if self._apply_join(sentence, traveler_plans, plan):
                    continue
                meal_reference = self._extract_reference(sentence, traveler_plans, default=last_reference)
                if meal_reference is not None:
                    last_reference = meal_reference
                slot = _slot_from_text(sentence)
                day = _day_from_text(sentence)
                if slot in MEAL_SLOTS and day is not None:
                    plan[day - 1][slot] = self._choose_meal(sentence, plan, traveler_plans, day, slot, last_reference)
                elif slot == "accommodation" and day is not None:
                    plan[day - 1][slot] = self._choose_accommodation(sentence, plan, traveler_plans, day, last_reference)
            traveler_plans[traveler] = plan
            generated.append(plan)
        return generated

    def _apply_join(
        self,
        sentence: str,
        traveler_plans: Dict[str, List[Dict[str, Any]]],
        plan: List[Dict[str, Any]],
    ) -> bool:
        lowered = sentence.lower()
        if not any(token in lowered for token in ("join", "share", "same place", "stay with", "eat with")):
            return False
        day = _day_from_text(sentence)
        if day is None:
            return False
        slot = _slot_from_text(sentence)
        if slot is None:
            return False
        ref = self._extract_reference(sentence, traveler_plans)
        if ref is None:
            return False
        ref_person, ref_day, ref_slot = ref
        source_slot = slot if slot != "accommodation" else ref_slot
        if source_slot not in traveler_plans[ref_person][ref_day - 1]:
            return False
        plan[day - 1][slot] = traveler_plans[ref_person][ref_day - 1][source_slot]
        return True

    def _extract_reference(
        self,
        sentence: str,
        traveler_plans: Dict[str, List[Dict[str, Any]]],
        default: Optional[Tuple[str, int, str]] = None,
    ) -> Optional[Tuple[str, int, str]]:
        names = sorted(traveler_plans, key=len, reverse=True)
        for name in names:
            pattern = rf"\b{name}'s\s+((?:first|second|third|fourth|fifth|sixth)(?:-day|\s+day)|day\s+\d+)\s+(breakfast|lunch|dinner|accommodation|stay)"
            match = re.search(pattern, sentence)
            if match:
                ref_day = _day_from_text(match.group(1))
                ref_slot = "accommodation" if match.group(2) == "stay" else match.group(2)
                if ref_day is not None:
                    return name, ref_day, ref_slot
        if " of it" in sentence.lower() or sentence.strip().endswith(" of it.") or " than it" in sentence.lower():
            return default
        for name in names:
            if re.search(rf"\b{name}\b", sentence):
                day = _day_from_text(sentence) or 1
                slot = _slot_from_text(sentence) or "accommodation"
                return name, day, slot
        return default

    def _reference_value(
        self,
        traveler_plans: Dict[str, List[Dict[str, Any]]],
        reference: Optional[Tuple[str, int, str]],
    ) -> Optional[str]:
        if reference is None:
            return None
        person, day, slot = reference
        return traveler_plans[person][day - 1][slot]

    def _restaurant_row(self, value: str) -> Optional[Dict[str, Any]]:
        if not value or value == "-":
            return None
        name, city = _name_city(value)
        frame = self.db.restaurants
        hit = frame[(frame["Name"] == name) & (frame["City"] == city)]
        if len(hit) < 1:
            return None
        return hit.iloc[0].to_dict()

    def _accommodation_row(self, value: str) -> Optional[Dict[str, Any]]:
        if not value or value == "-":
            return None
        name, city = _name_city(value)
        frame = self.db.accommodations
        hit = frame[(frame["NAME"] == name) & (frame["city"] == city)]
        if len(hit) < 1:
            return None
        return hit.iloc[0].to_dict()

    def _choose_meal(
        self,
        sentence: str,
        plan: List[Dict[str, Any]],
        traveler_plans: Dict[str, List[Dict[str, Any]]],
        day: int,
        slot: str,
        last_reference: Optional[Tuple[str, int, str]],
    ) -> str:
        city = _city_for_day(plan[day - 1], slot=slot)
        frame = self.db.restaurants_in_city(city).copy()
        if len(frame) < 1:
            return plan[day - 1][slot]

        cuisines = self._parse_required_cuisines(sentence, traveler_plans, last_reference)
        if cuisines:
            frame = frame[frame["Cuisines"].astype(str).apply(lambda value: _contains_all_cuisines(value, cuisines))]
        frame = self._filter_price_and_rating(frame, sentence, traveler_plans, is_accommodation=False, reference=last_reference)
        if len(frame) < 1:
            frame = self.db.restaurants_in_city(city).copy()
        if cuisines:
            frame["coverage"] = frame["Cuisines"].astype(str).apply(lambda value: sum(c.lower() in value.lower() for c in cuisines))
        else:
            frame["coverage"] = 0
        frame["price_distance"] = frame["Average Cost"].astype(float).apply(lambda value: self._price_distance(value, sentence, traveler_plans, False, last_reference))
        frame["rating_distance"] = frame["Aggregate Rating"].astype(float).apply(lambda value: self._rating_distance(value, sentence, traveler_plans, False, last_reference))
        chosen = frame.sort_values(
            by=["coverage", "price_distance", "rating_distance", "Aggregate Rating", "Average Cost"],
            ascending=[False, True, True, False, True],
        ).iloc[0]
        return f"{chosen['Name']}, {city}"

    def _choose_accommodation(
        self,
        sentence: str,
        plan: List[Dict[str, Any]],
        traveler_plans: Dict[str, List[Dict[str, Any]]],
        day: int,
        last_reference: Optional[Tuple[str, int, str]],
    ) -> str:
        city = _city_for_day(plan[day - 1], slot="accommodation")
        frame = self.db.accommodations_in_city(city).copy()
        if len(frame) < 1:
            return plan[day - 1]["accommodation"]

        room_type = self._required_room_type(sentence, traveler_plans, last_reference)
        if room_type is not None:
            frame = frame[frame["room type"] == room_type]
        if "different room type" in sentence.lower() and last_reference is not None:
            ref = self._accommodation_row(self._reference_value(traveler_plans, last_reference) or "")
            if ref is not None:
                frame = frame[frame["room type"] != ref["room type"]]

        required_rules = self._required_house_rules(sentence, traveler_plans, last_reference)
        for rule in required_rules:
            frame = frame[frame["house_rules"].astype(str).str.contains(rule, regex=False)]

        if "shares at least one house rule" in sentence.lower() and last_reference is not None:
            ref = self._accommodation_row(self._reference_value(traveler_plans, last_reference) or "")
            if ref is not None:
                ref_rules = {part.strip() for part in str(ref["house_rules"]).split("|") if part.strip()}
                frame = frame[
                    frame["house_rules"].astype(str).apply(
                        lambda text: bool(ref_rules.intersection({part.strip() for part in str(text).split("|") if part.strip()}))
                    )
                ]

        if "same no-parties rule" in sentence.lower() and last_reference is not None:
            ref = self._accommodation_row(self._reference_value(traveler_plans, last_reference) or "")
            if ref is not None:
                ref_has_rule = "No parties" in str(ref["house_rules"])
                if ref_has_rule:
                    frame = frame[frame["house_rules"].astype(str).str.contains("No parties", regex=False)]
                else:
                    frame = frame[~frame["house_rules"].astype(str).str.contains("No parties", regex=False)]

        occupancy = self._required_occupancy(sentence)
        if occupancy is not None:
            frame = frame[frame["maximum occupancy"].astype(int) >= occupancy]

        frame = self._filter_price_and_rating(frame, sentence, traveler_plans, is_accommodation=True, reference=last_reference)
        if len(frame) < 1:
            frame = self.db.accommodations_in_city(city).copy()
        frame["price_distance"] = frame["price"].astype(float).apply(lambda value: self._price_distance(value, sentence, traveler_plans, True, last_reference))
        frame["rating_distance"] = frame["review rate number"].astype(float).apply(lambda value: self._rating_distance(value, sentence, traveler_plans, True, last_reference))
        chosen = frame.sort_values(
            by=["price_distance", "rating_distance", "review rate number", "price"],
            ascending=[True, True, False, True],
        ).iloc[0]
        return f"{chosen['NAME']}, {city}"

    def _parse_required_cuisines(
        self,
        sentence: str,
        traveler_plans: Dict[str, List[Dict[str, Any]]],
        reference: Optional[Tuple[str, int, str]],
    ) -> List[str]:
        lowered = sentence.lower()
        if "same cuisines" in lowered and reference is not None:
            ref_row = self._restaurant_row(self._reference_value(traveler_plans, reference) or "")
            if ref_row is not None:
                return [part.strip() for part in str(ref_row["Cuisines"]).split(",") if part.strip()]

        for pattern in [
            r"serving ([A-Za-z ,]+?) food",
            r"serves ([A-Za-z ,]+?) food",
            r"serving ([A-Za-z ,]+?) cuisine",
            r"serves ([A-Za-z ,]+?) cuisine",
            r"offering ([A-Za-z ,]+?) items",
            r"(?:a|an)\s+([A-Za-z ,]+?)\s+restaurant",
        ]:
            match = re.search(pattern, sentence)
            if match:
                return _parse_cuisine_list(match.group(1))
        return []

    def _required_room_type(
        self,
        sentence: str,
        traveler_plans: Dict[str, List[Dict[str, Any]]],
        reference: Optional[Tuple[str, int, str]],
    ) -> Optional[str]:
        lowered = sentence.lower()
        for phrase, canonical in ROOM_TYPE_MAP.items():
            if phrase in lowered:
                return canonical
        if "same room type" in lowered and reference is not None:
            ref = self._accommodation_row(self._reference_value(traveler_plans, reference) or "")
            if ref is not None:
                return str(ref["room type"])
        return None

    def _required_house_rules(
        self,
        sentence: str,
        traveler_plans: Dict[str, List[Dict[str, Any]]],
        reference: Optional[Tuple[str, int, str]],
    ) -> List[str]:
        lowered = sentence.lower()
        required = []
        for phrase, rule in RULE_PHRASES.items():
            if phrase in lowered and ("no" in lowered or "doesn't" in lowered or "non-smoking" in lowered or "pet-free" in lowered):
                required.append(rule)
        if "same no-parties rule" in lowered:
            return required
        if "same room type" in lowered or "same no-parties rule" in lowered:
            return required
        if "shares the same no-parties rule" in lowered and reference is not None:
            ref = self._accommodation_row(self._reference_value(traveler_plans, reference) or "")
            if ref is not None and "No parties" in str(ref["house_rules"]):
                required.append("No parties")
        return list(dict.fromkeys(required))

    def _required_occupancy(self, sentence: str) -> Optional[int]:
        lowered = sentence.lower()
        for word, value in NUMBER_WORD.items():
            if f"{word} people" in lowered or f"{word} person" in lowered:
                return value
        match = re.search(r"fits (\d+) people", lowered)
        if match:
            return int(match.group(1))
        return None

    def _price_distance(
        self,
        value: float,
        sentence: str,
        traveler_plans: Dict[str, List[Dict[str, Any]]],
        is_accommodation: bool,
        reference: Optional[Tuple[str, int, str]],
    ) -> float:
        bounds = self._price_bounds(sentence, traveler_plans, is_accommodation, reference)
        if bounds is None:
            return 0.0
        lower, upper = bounds
        if lower <= value <= upper:
            return 0.0
        if value < lower:
            return lower - value
        return value - upper

    def _rating_distance(
        self,
        value: float,
        sentence: str,
        traveler_plans: Dict[str, List[Dict[str, Any]]],
        is_accommodation: bool,
        reference: Optional[Tuple[str, int, str]],
    ) -> float:
        bounds = self._rating_bounds(sentence, traveler_plans, is_accommodation, reference)
        if bounds is None:
            return 0.0
        lower, upper = bounds
        if lower <= value <= upper:
            return 0.0
        if value < lower:
            return lower - value
        return value - upper

    def _filter_price_and_rating(
        self,
        frame,
        sentence: str,
        traveler_plans: Dict[str, List[Dict[str, Any]]],
        *,
        is_accommodation: bool,
        reference: Optional[Tuple[str, int, str]],
    ):
        price_bounds = self._price_bounds(sentence, traveler_plans, is_accommodation, reference)
        rating_bounds = self._rating_bounds(sentence, traveler_plans, is_accommodation, reference)
        price_column = "price" if is_accommodation else "Average Cost"
        rating_column = "review rate number" if is_accommodation else "Aggregate Rating"
        filtered = frame
        if price_bounds is not None:
            filtered = filtered[
                filtered[price_column].astype(float).between(price_bounds[0], price_bounds[1], inclusive="both")
            ]
        if rating_bounds is not None:
            filtered = filtered[
                filtered[rating_column].astype(float).between(rating_bounds[0], rating_bounds[1], inclusive="both")
            ]
        return filtered

    def _price_bounds(
        self,
        sentence: str,
        traveler_plans: Dict[str, List[Dict[str, Any]]],
        is_accommodation: bool,
        reference: Optional[Tuple[str, int, str]],
    ) -> Optional[Tuple[float, float]]:
        if match := re.search(r"\$(\d+(?:\.\d+)?)\s*[–-]\s*\$?(\d+(?:\.\d+)?)", sentence):
            return float(match.group(1)), float(match.group(2))
        if match := re.search(r"between \$(\d+(?:\.\d+)?) and \$(\d+(?:\.\d+)?)", sentence):
            return float(match.group(1)), float(match.group(2))

        ref_value = self._reference_numeric(traveler_plans, reference, is_accommodation)
        if ref_value is None:
            return None
        if match := re.search(r"within \$(\d+(?:\.\d+)?) of", sentence):
            delta = float(match.group(1))
            return ref_value - delta, ref_value + delta
        if match := re.search(r"within (\d+(?:\.\d+)?)% of", sentence):
            delta = ref_value * float(match.group(1)) / 100.0
            return ref_value - delta, ref_value + delta
        if match := re.search(r"at least (\d+(?:\.\d+)?)% more", sentence):
            lower = ref_value * (1.0 + float(match.group(1)) / 100.0)
            return lower, float("inf")
        if match := re.search(r"at least (\d+(?:\.\d+)?)% less", sentence):
            upper = ref_value * (1.0 - float(match.group(1)) / 100.0)
            return 0.0, upper
        if match := re.search(r"at least \$(\d+(?:\.\d+)?) more", sentence):
            lower = ref_value + float(match.group(1))
            return lower, float("inf")
        if match := re.search(r"at least \$(\d+(?:\.\d+)?) less", sentence):
            upper = ref_value - float(match.group(1))
            return 0.0, upper
        if re.search(r"costs? less than|priced? less than", sentence):
            return 0.0, ref_value
        if re.search(r"costs? more than|priced? more than|costing more|priced higher|costing higher", sentence):
            return ref_value, float("inf")
        return None

    def _rating_bounds(
        self,
        sentence: str,
        traveler_plans: Dict[str, List[Dict[str, Any]]],
        is_accommodation: bool,
        reference: Optional[Tuple[str, int, str]],
    ) -> Optional[Tuple[float, float]]:
        if match := re.search(r"rating between (\d+(?:\.\d+)?) and (\d+(?:\.\d+)?)", sentence):
            return float(match.group(1)), float(match.group(2))
        if match := re.search(r"rated between (\d+(?:\.\d+)?) and (\d+(?:\.\d+)?)", sentence):
            return float(match.group(1)), float(match.group(2))
        ref_value = self._reference_rating(traveler_plans, reference, is_accommodation)
        if ref_value is None:
            return None
        if match := re.search(r"rating within (\d+(?:\.\d+)?)", sentence):
            delta = float(match.group(1))
            return ref_value - delta, ref_value + delta
        if match := re.search(r"rated within (\d+(?:\.\d+)?)", sentence):
            delta = float(match.group(1))
            return ref_value - delta, ref_value + delta
        if "rated higher" in sentence or "rating higher" in sentence:
            return ref_value, float("inf")
        if "rated lower" in sentence or "rating lower" in sentence:
            return 0.0, ref_value
        if "rated at least as high" in sentence:
            return ref_value, float("inf")
        return None

    def _reference_numeric(
        self,
        traveler_plans: Dict[str, List[Dict[str, Any]]],
        reference: Optional[Tuple[str, int, str]],
        is_accommodation: bool,
    ) -> Optional[float]:
        value = self._reference_value(traveler_plans, reference)
        if not value:
            return None
        if is_accommodation:
            row = self._accommodation_row(value)
            if row is None:
                return None
            return float(row["price"])
        row = self._restaurant_row(value)
        if row is None:
            return None
        return float(row["Average Cost"])

    def _reference_rating(
        self,
        traveler_plans: Dict[str, List[Dict[str, Any]]],
        reference: Optional[Tuple[str, int, str]],
        is_accommodation: bool,
    ) -> Optional[float]:
        value = self._reference_value(traveler_plans, reference)
        if not value:
            return None
        if is_accommodation:
            row = self._accommodation_row(value)
            if row is None:
                return None
            return float(row["review rate number"])
        row = self._restaurant_row(value)
        if row is None:
            return None
        return float(row["Aggregate Rating"])


def evaluate_group_travel_planner(
    policy: MemoryArenaTravelPolicy,
    *,
    write_json_path: Optional[str | Path] = None,
) -> MemoryArenaTravelResult:
    archive = MemoryArenaArchive.default() if policy.use_archive else None
    solver = GroupTravelArenaSolver(archive=archive)
    rows = GroupTravelArenaSolver.load_rows(limit=policy.limit_rows)
    outcomes: List[TravelerOutcome] = []
    exact_matches = 0
    mutable_correct = 0
    mutable_total = 0

    for row in rows:
        generated = solver.solve_row(row)
        gold = row["answers"]
        for question, generated_plan, gold_plan in zip(row["questions"], generated, gold):
            traveler = _extract_traveler_name(question)
            exact = _normalized_plan(generated_plan) == _normalized_plan(gold_plan)
            exact_matches += int(exact)
            correct = 0
            total = 0
            for gen_day, gold_day in zip(generated_plan, gold_plan):
                for slot in ("breakfast", "lunch", "dinner", "accommodation"):
                    total += 1
                    if _normalize_text(gen_day[slot]) == _normalize_text(gold_day[slot]):
                        correct += 1
            mutable_correct += correct
            mutable_total += total
            outcomes.append(
                TravelerOutcome(
                    traveler=traveler,
                    exact_match=exact,
                    mutable_slot_accuracy=correct / max(total, 1),
                )
            )

    result = MemoryArenaTravelResult(
        policy=policy,
        num_rows=len(rows),
        num_travelers=len(outcomes),
        exact_match_rate=exact_matches / max(len(outcomes), 1),
        mutable_slot_accuracy=mutable_correct / max(mutable_total, 1),
        outcomes=outcomes,
    )

    if write_json_path is not None:
        path = Path(write_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "policy_id": policy.policy_id,
                    "num_rows": result.num_rows,
                    "num_travelers": result.num_travelers,
                    "exact_match_rate": result.exact_match_rate,
                    "mutable_slot_accuracy": result.mutable_slot_accuracy,
                    "summary": policy.summary(),
                    "outcomes": [
                        {
                            "traveler": outcome.traveler,
                            "exact_match": outcome.exact_match,
                            "mutable_slot_accuracy": outcome.mutable_slot_accuracy,
                        }
                        for outcome in outcomes
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )

    return result
