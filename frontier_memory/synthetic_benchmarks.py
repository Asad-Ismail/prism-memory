from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ConversationTurn:
    speaker: str
    text: str
    turn_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QAPair:
    question: str
    answer: str
    question_type: str
    required_turns: List[int]
    hard_negatives: List[str] = field(default_factory=list)


@dataclass
class BenchmarkExample:
    id: str
    conversation: List[ConversationTurn]
    qa_pairs: List[QAPair]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SyntheticBenchmark:
    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)


ATTRIBUTE_TEMPLATES = {
    "job": {
        "establish": "{entity} works as a {value} at {company}.",
        "update": "{entity} just started a new position as {value} at {company}.",
        "current_q": "What does {entity} do for work?",
        "historical_q": "What was {entity}'s job before they changed careers?",
        "transition_q": "When did {entity} switch jobs?",
    },
    "city": {
        "establish": "{entity} lives in {value}.",
        "update": "{entity} recently moved to {value}.",
        "current_q": "Where does {entity} live now?",
        "historical_q": "Where did {entity} use to live before moving?",
        "transition_q": "When did {entity} move to a new city?",
    },
    "relationship": {
        "establish": "{entity} is in a relationship with {value}.",
        "update": "{entity} and {value} recently got together.",
        "current_q": "Who is {entity} currently dating?",
        "historical_q": "Who was {entity} with before their current partner?",
        "transition_q": "When did {entity}'s relationship status change?",
    },
    "hobby": {
        "establish": "{entity} loves {value} and does it every weekend.",
        "update": "{entity} has really gotten into {value} lately.",
        "current_q": "What is {entity}'s main hobby these days?",
        "historical_q": "What hobby did {entity} have before their current one?",
        "transition_q": "When did {entity} pick up their new hobby?",
    },
}

FILLER_TOPICS = [
    "talked about the weather and upcoming holidays",
    "mentioned a book they had been reading",
    "discussed a movie they watched recently",
    "brought up a friend's birthday party",
    "chatted about weekend plans",
    "mentioned something funny that happened at the grocery store",
    "talked about a recipe they tried",
    "discussed local news",
]

ENTITY_NAMES = [
    "Alice",
    "Bob",
    "Carol",
    "David",
    "Eve",
    "Frank",
    "Grace",
    "Henry",
    "Isabella",
    "Jack",
    "Karen",
    "Liam",
    "Maya",
    "Noah",
    "Olivia",
]

COMPANIES = [
    "Google",
    "Microsoft",
    "Stripe",
    "Airbnb",
    "a startup",
    "a hospital",
    "the university",
    "a law firm",
    "a design studio",
]
CITIES = [
    "Seattle",
    "Austin",
    "Boston",
    "Denver",
    "Portland",
    "Nashville",
    "Chicago",
    "Miami",
    "Phoenix",
    "San Diego",
]
PARTNERS = ["Jamie", "Sam", "Riley", "Jordan", "Casey", "Morgan", "Quinn"]
HOBBIES_OLD = ["cycling", "painting", "cooking", "chess", "gardening", "photography"]
HOBBIES_NEW = ["rock climbing", "pottery", "surfing", "running", "yoga", "woodworking"]
JOBS_OLD = ["software engineer", "accountant", "teacher", "nurse", "designer", "analyst"]
JOBS_NEW = ["product manager", "data scientist", "entrepreneur", "consultant", "writer", "therapist"]


class TemporalDriftBenchmark(SyntheticBenchmark):
    def __init__(
        self,
        seed: int = 42,
        *,
        update_turn_options: Optional[List[int]] = None,
        total_turns: int = 80,
        attributes: Optional[List[str]] = None,
    ) -> None:
        super().__init__(seed=seed)
        self.update_turn_options = update_turn_options or [15, 30, 50]
        self.total_turns = total_turns
        self.attributes = attributes or list(ATTRIBUTE_TEMPLATES.keys())

    def generate_example(
        self,
        *,
        attribute: Optional[str] = None,
        update_turn: Optional[int] = None,
    ) -> BenchmarkExample:
        attr = attribute or self.rng.choice(self.attributes)
        update_at = update_turn or self.rng.choice(self.update_turn_options)
        entity = self.rng.choice(ENTITY_NAMES)
        template = ATTRIBUTE_TEMPLATES[attr]
        old_value, new_value, context = self._sample_values(attr)

        turns: List[ConversationTurn] = []
        turns.append(
            ConversationTurn(
                speaker="User",
                text=template["establish"].format(entity=entity, value=old_value, company=context.get("company0", "")),
                turn_index=1,
                metadata={
                    "phase": "establish",
                    "entity": entity,
                    "attribute": attr,
                },
            )
        )
        turns.append(
            ConversationTurn(
                speaker="AI",
                text=f"I'll remember that about {entity}.",
                turn_index=2,
            )
        )
        occupied_turns = {1, 2}
        for turn_index in range(3, self.total_turns + 1):
            if turn_index in occupied_turns:
                continue
            if turn_index == update_at:
                turns.append(
                    ConversationTurn(
                        speaker="User",
                        text=template["update"].format(entity=entity, value=new_value, company=context.get("company1", "")),
                        turn_index=turn_index,
                        metadata={
                            "phase": "update",
                            "entity": entity,
                            "attribute": attr,
                        },
                    )
                )
                occupied_turns.add(turn_index)
                if turn_index + 1 <= self.total_turns:
                    turns.append(
                        ConversationTurn(
                            speaker="AI",
                            text="That is a big update.",
                            turn_index=turn_index + 1,
                        )
                    )
                    occupied_turns.add(turn_index + 1)
                continue
            filler = self.rng.choice(FILLER_TOPICS)
            turns.append(
                ConversationTurn(
                    speaker="User" if turn_index % 2 else "AI",
                    text=f"They {filler}.",
                    turn_index=turn_index,
                )
            )

        qa_pairs = [
            QAPair(
                question=template["current_q"].format(entity=entity),
                answer=self._value_str(attr, new_value, context.get("company1", "")),
                question_type="current_state",
                required_turns=[update_at],
            ),
            QAPair(
                question=template["historical_q"].format(entity=entity),
                answer=self._value_str(attr, old_value, context.get("company0", "")),
                question_type="historical",
                required_turns=[1],
            ),
            QAPair(
                question=template["transition_q"].format(entity=entity),
                answer=f"Around turn {update_at} of the conversation / after turn {max(update_at - 5, 1)}",
                question_type="transition_timing",
                required_turns=[update_at],
            ),
        ]
        return BenchmarkExample(
            id=str(uuid.uuid4()),
            conversation=sorted(turns, key=lambda item: item.turn_index),
            qa_pairs=qa_pairs,
            metadata={"entity": entity, "attribute": attr, "update_turn": update_at},
        )

    def _sample_values(self, attr: str) -> Tuple[str, str, Dict[str, str]]:
        if attr == "job":
            old_value = self.rng.choice(JOBS_OLD)
            new_value = self.rng.choice([job for job in JOBS_NEW if job != old_value])
            return old_value, new_value, {"company0": self.rng.choice(COMPANIES), "company1": self.rng.choice(COMPANIES)}
        if attr == "city":
            old_value = self.rng.choice(CITIES)
            new_value = self.rng.choice([city for city in CITIES if city != old_value])
            return old_value, new_value, {}
        if attr == "relationship":
            old_value = self.rng.choice(PARTNERS)
            new_value = self.rng.choice([partner for partner in PARTNERS if partner != old_value])
            return old_value, new_value, {}
        old_value = self.rng.choice(HOBBIES_OLD)
        new_value = self.rng.choice(HOBBIES_NEW)
        return old_value, new_value, {}

    def _value_str(self, attr: str, value: str, company: str) -> str:
        if attr == "job" and company:
            return f"{value} at {company}"
        return value


CONTRADICTION_PAIRS: List[Tuple[str, str, str, str, str, str, str]] = [
    (
        "Alice",
        "job",
        "software engineer at Google",
        "product manager at a startup",
        "What does Alice do for work now?",
        "What was Alice's job before her career change?",
        "Has Alice ever worked as a software engineer?",
    ),
    (
        "Bob",
        "city",
        "living in Seattle",
        "living in Austin after relocating",
        "Where does Bob live now?",
        "Where did Bob used to live before moving?",
        "Has Bob ever lived in Seattle?",
    ),
    (
        "Carol",
        "relationship status",
        "single and not dating anyone",
        "in a new relationship with someone she met at work",
        "Is Carol currently in a relationship?",
        "What was Carol's relationship status before?",
        "Was Carol ever single?",
    ),
    (
        "David",
        "dietary preference",
        "a big meat-eater who loves BBQ",
        "fully plant-based after a health scare",
        "What does David eat now?",
        "What were David's eating habits before his change?",
        "Did David ever eat meat?",
    ),
    (
        "Eve",
        "commute",
        "driving 45 minutes to the office every day",
        "fully remote works from home since her company changed policy",
        "How does Eve get to work now?",
        "How did Eve commute before going remote?",
        "Did Eve ever commute by car?",
    ),
]

STABLE_FACTS: List[Tuple[str, str, str, str]] = [
    ("Alice", "loves hiking on weekends", "What does Alice like to do on weekends?", "hiking"),
    ("Bob", "has a golden retriever named Max", "What is Bob's dog's name?", "Max"),
    ("Carol", "grew up in Portland", "Where did Carol grow up?", "Portland"),
    ("David", "has a PhD in chemistry", "What degree does David have?", "PhD in chemistry"),
    ("Eve", "plays classical piano", "What instrument does Eve play?", "piano"),
]


class ContraFactBenchmark(SyntheticBenchmark):
    def __init__(self, seed: int = 42, *, update_turn: int = 40, total_turns: int = 100) -> None:
        super().__init__(seed=seed)
        self.update_turn = update_turn
        self.total_turns = total_turns

    def generate_example(self, *, pair_idx: Optional[int] = None) -> BenchmarkExample:
        pair = CONTRADICTION_PAIRS[pair_idx if pair_idx is not None else self.rng.randrange(len(CONTRADICTION_PAIRS))]
        entity, attr, old_fact, new_fact, current_q, historical_q, ever_q = pair
        stable = [fact for fact in STABLE_FACTS if fact[0] == entity][:1] + self.rng.sample(
            [fact for fact in STABLE_FACTS if fact[0] != entity],
            2,
        )

        turns = [
            ConversationTurn(
                speaker="User",
                text=f"I should mention that {entity} is {old_fact}.",
                turn_index=1,
                metadata={"phase": "establish", "entity": entity, "attribute": attr, "fact": old_fact},
            ),
            ConversationTurn(speaker="AI", text=f"Got it, I will remember that about {entity}.", turn_index=2),
        ]
        next_turn = 5
        for stable_entity, stable_fact, _, _ in stable:
            turns.append(
                ConversationTurn(
                    speaker="User",
                    text=f"Also, {stable_entity} {stable_fact}.",
                    turn_index=next_turn,
                    metadata={"phase": "stable", "entity": stable_entity, "fact": stable_fact},
                )
            )
            turns.append(ConversationTurn(speaker="AI", text="Noted.", turn_index=next_turn + 1))
            next_turn += 4
        for filler_turn in range(next_turn, self.update_turn, 2):
            turns.append(ConversationTurn(speaker="User", text="Just catching up on life.", turn_index=filler_turn))
            turns.append(ConversationTurn(speaker="AI", text="Sounds good.", turn_index=filler_turn + 1))
        turns.append(
            ConversationTurn(
                speaker="User",
                text=f"Oh by the way, big update about {entity}: they are now {new_fact}.",
                turn_index=self.update_turn,
                metadata={"phase": "update", "entity": entity, "attribute": attr, "fact": new_fact, "contradicts": old_fact},
            )
        )
        turns.append(ConversationTurn(speaker="AI", text="That is quite a change.", turn_index=self.update_turn + 1))

        qa_pairs = [
            QAPair(current_q, new_fact, "current_state_post_contradiction", [self.update_turn]),
            QAPair(historical_q, old_fact, "historical_pre_contradiction", [1], hard_negatives=[new_fact]),
            QAPair(ever_q, "yes", "ever_true", [1]),
        ]
        for _, _, question, answer in stable:
            qa_pairs.append(QAPair(question, answer, "stable_fact", []))
        return BenchmarkExample(
            id=str(uuid.uuid4()),
            conversation=sorted(turns, key=lambda item: item.turn_index),
            qa_pairs=qa_pairs,
            metadata={"entity": entity, "attribute": attr, "update_turn": self.update_turn},
        )


SPEAKER_NAMES = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry"]

TRAVEL_FACTS: Dict[str, Tuple[str, str, str]] = {
    "Alice": ("Paris for a conference", "Where did Alice travel recently?", "Paris"),
    "Bob": ("Tokyo on a holiday", "Where did Bob travel recently?", "Tokyo"),
    "Carol": ("Rome for a wedding", "Where did Carol travel recently?", "Rome"),
    "David": ("Sydney for a work trip", "Where did David travel recently?", "Sydney"),
    "Eve": ("Nairobi for volunteer work", "Where did Eve travel recently?", "Nairobi"),
    "Frank": ("Montreal for a music festival", "Where did Frank travel recently?", "Montreal"),
    "Grace": ("Seoul for a friend's birthday", "Where did Grace travel recently?", "Seoul"),
    "Henry": ("Buenos Aires on sabbatical", "Where did Henry travel recently?", "Buenos Aires"),
}

FOOD_FACTS: Dict[str, Tuple[str, str, str]] = {
    "Alice": ("she's been eating a lot of sushi", "What food has Alice been enjoying?", "sushi"),
    "Bob": ("he started making sourdough bread", "What has Bob been making?", "sourdough bread"),
    "Carol": ("she discovered a great taco place", "What has Carol discovered?", "a taco place"),
    "David": ("he went vegan last month", "What dietary change has David made?", "went vegan"),
    "Eve": ("she's obsessed with Ethiopian food", "What cuisine is Eve into?", "Ethiopian food"),
    "Frank": ("he makes homemade ramen every weekend", "What does Frank make on weekends?", "ramen"),
    "Grace": ("she tried fermented foods for health", "What has Grace been exploring?", "fermented foods"),
    "Henry": ("he takes a cooking class on Tuesdays", "What does Henry do on Tuesdays?", "cooking class"),
}

WORK_FACTS: Dict[str, Tuple[str, str, str]] = {
    "Alice": ("got a promotion to senior engineer", "What professional news does Alice have?", "promotion to senior engineer"),
    "Bob": ("is applying to PhD programs", "What is Bob planning?", "applying to PhD programs"),
    "Carol": ("started freelancing part-time", "What work change did Carol make?", "started freelancing"),
    "David": ("is mentoring interns this summer", "What is David doing this summer?", "mentoring interns"),
    "Eve": ("gave a TEDx talk last month", "What did Eve do last month?", "gave a TEDx talk"),
    "Frank": ("is building a side project startup", "What is Frank working on outside work?", "side project startup"),
    "Grace": ("moved to a fully remote position", "What change did Grace make at work?", "moved to remote"),
    "Henry": ("is writing a technical book", "What is Henry writing?", "a technical book"),
}

TOPIC_POOLS = {"travel": TRAVEL_FACTS, "food": FOOD_FACTS, "work": WORK_FACTS}


class AttributionStressBenchmark(SyntheticBenchmark):
    def __init__(self, seed: int = 42, *, n_speakers: int = 4, topics: Optional[List[str]] = None) -> None:
        super().__init__(seed=seed)
        self.n_speakers = n_speakers
        self.topics = topics or ["travel", "food", "work"]

    def generate_example(self) -> BenchmarkExample:
        speakers = self.rng.sample(SPEAKER_NAMES, self.n_speakers)
        events = []
        for topic in self.topics:
            pool = TOPIC_POOLS[topic]
            for speaker in speakers:
                fact_text, question, answer = pool[speaker]
                events.append((speaker, topic, fact_text, question, answer))
        self.rng.shuffle(events)

        turns: List[ConversationTurn] = []
        qa_pairs: List[QAPair] = []
        turn_index = 1
        for idx, (speaker, topic, fact_text, question, answer) in enumerate(events):
            if idx and idx % 3 == 0:
                turns.append(
                    ConversationTurn(
                        speaker="User",
                        text="Thanks for the updates everyone. Anything else going on?",
                        turn_index=turn_index,
                    )
                )
                turn_index += 1
            turns.append(
                ConversationTurn(
                    speaker=speaker,
                    text=f"By the way, {fact_text}.",
                    turn_index=turn_index,
                    metadata={"speaker": speaker, "topic": topic, "fact": fact_text},
                )
            )
            turn_index += 1
            reactor = self.rng.choice([name for name in speakers if name != speaker])
            turns.append(
                ConversationTurn(
                    speaker=reactor,
                    text=self.rng.choice(
                        [
                            f"Oh wow, {speaker}!",
                            "That's great to hear!",
                            f"Good for you, {speaker}.",
                            "Interesting!",
                        ]
                    ),
                    turn_index=turn_index,
                )
            )
            turn_index += 1
            hard_negatives = [
                TOPIC_POOLS[topic][other][2]
                for other in speakers
                if other != speaker
            ][:2]
            qa_pairs.append(
                QAPair(
                    question=question,
                    answer=answer,
                    question_type=f"attribution_{self.n_speakers}speakers",
                    required_turns=[turn_index - 2],
                    hard_negatives=hard_negatives,
                )
            )
        return BenchmarkExample(
            id=str(uuid.uuid4()),
            conversation=turns,
            qa_pairs=qa_pairs,
            metadata={"speakers": speakers, "topics": self.topics},
        )


RELATION_TEMPLATES = {
    "colleague": {
        "statement": "{A} works with {B} at the office.",
        "question": "Who does {A} work with?",
    },
    "neighbour": {
        "statement": "{A} lives next door to {B}.",
        "question": "Who lives next door to {A}?",
    },
    "friend": {
        "statement": "{A} and {B} are close friends.",
        "question": "Who is {A}'s close friend?",
    },
    "cousin": {
        "statement": "{A}'s cousin is {B}.",
        "question": "Who is {A}'s cousin?",
    },
    "classmate": {
        "statement": "{A} and {B} were classmates in college.",
        "question": "Who was {A}'s college classmate?",
    },
}

TERMINAL_FACTS = {
    "city": {
        "statement": "{entity} lives in {value}.",
        "q_suffix": "live",
        "values": ["Seattle", "Boston", "Austin", "Portland", "Denver"],
    },
    "job": {
        "statement": "{entity} works as a {value}.",
        "q_suffix": "do for work",
        "values": ["teacher", "nurse", "chef", "architect", "pilot"],
    },
    "hobby": {
        "statement": "{entity} loves {value}.",
        "q_suffix": "enjoy as a hobby",
        "values": ["hiking", "painting", "cooking", "chess", "gardening"],
    },
}

CHAIN_NAMES = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry", "Iris", "Jake", "Laura", "Mike", "Nora", "Oscar"]
CHAIN_RELATIONS = list(RELATION_TEMPLATES.keys())


class EntityChainBenchmark(SyntheticBenchmark):
    def __init__(self, seed: int = 42, *, depths: Optional[List[int]] = None) -> None:
        super().__init__(seed=seed)
        self.depths = depths or [2, 3, 4]

    def generate_example(self, *, depth: Optional[int] = None) -> BenchmarkExample:
        chain_depth = depth or self.rng.choice(self.depths)
        names = self.rng.sample(CHAIN_NAMES, chain_depth + 1)
        relations = self.rng.sample(CHAIN_RELATIONS, chain_depth)
        terminal_type = self.rng.choice(list(TERMINAL_FACTS.keys()))
        terminal_value = self.rng.choice(TERMINAL_FACTS[terminal_type]["values"])

        positions = sorted(self.rng.sample(range(1, 40), chain_depth + 1))
        turns: Dict[int, ConversationTurn] = {}
        for index, relation in enumerate(relations):
            turns[positions[index]] = ConversationTurn(
                speaker="User",
                text=f"Oh I should tell you, {RELATION_TEMPLATES[relation]['statement'].format(A=names[index], B=names[index + 1])}",
                turn_index=positions[index],
                metadata={"type": "relation", "A": names[index], "B": names[index + 1], "rel": relation},
            )
        terminal_statement = TERMINAL_FACTS[terminal_type]["statement"].format(entity=names[-1], value=terminal_value)
        turns[positions[-1]] = ConversationTurn(
            speaker="User",
            text=f"Also, {terminal_statement}",
            turn_index=positions[-1],
            metadata={"type": "terminal", "entity": names[-1], "fact_type": terminal_type, "value": terminal_value},
        )
        for turn_index in range(1, 45):
            turns.setdefault(
                turn_index,
                ConversationTurn(
                    speaker="User" if turn_index % 2 else "AI",
                    text=self.rng.choice(
                        [
                            "How has your week been?",
                            "The weather has been nice.",
                            "I tried a new restaurant.",
                            "Work has been busy.",
                            "I went for a walk this morning.",
                        ]
                    ),
                    turn_index=turn_index,
                ),
            )

        chain_question = f"What does {names[0]}'s " + "'s ".join(relations) + f" {TERMINAL_FACTS[terminal_type]['q_suffix']}?"
        qa_pairs = [
            QAPair(
                question=chain_question,
                answer=terminal_value,
                question_type=f"chain_{chain_depth}hop",
                required_turns=positions,
            ),
            QAPair(
                question=RELATION_TEMPLATES[relations[0]]["question"].format(A=names[0]),
                answer=names[1],
                question_type="chain_1hop",
                required_turns=[positions[0]],
            ),
        ]
        return BenchmarkExample(
            id=str(uuid.uuid4()),
            conversation=[turns[index] for index in sorted(turns)],
            qa_pairs=qa_pairs,
            metadata={"chain": names, "relations": relations, "terminal_value": terminal_value},
        )


LOW_FREQ_FACTS: List[Tuple[str, str, str, str]] = [
    ("Alice", "has a twin sister named Ava who lives in New Zealand", "Does Alice have a twin?", "yes, her twin sister Ava lives in New Zealand"),
    ("Bob", "holds a world record for the fastest time assembling a 1000-piece puzzle", "What world record does Bob hold?", "fastest 1000-piece puzzle assembly"),
    ("Carol", "was a child actress in a TV commercial at age 4", "What did Carol do as a young child?", "was in a TV commercial"),
    ("David", "is afraid of butterflies", "What is David afraid of?", "butterflies"),
    ("Eve", "can solve a Rubik's cube blindfolded", "What impressive thing can Eve do?", "solve a Rubik's cube blindfolded"),
    ("Frank", "once accidentally replied-all to a 10,000-person company email", "What funny email mishap did Frank have?", "replied-all to 10000 people"),
]

HIGH_FREQ_FACTS: List[Tuple[str, str, str, str]] = [
    ("Alice", "loves hiking on weekends", "What does Alice like to do on weekends?", "hiking"),
    ("Bob", "works as a software engineer", "What does Bob do for work?", "software engineer"),
    ("Carol", "is training for a half-marathon", "What is Carol training for?", "half-marathon"),
    ("David", "is really into cooking Italian food", "What cuisine does David enjoy cooking?", "Italian food"),
    ("Eve", "drives a red car", "What colour is Eve's car?", "red"),
    ("Frank", "loves watching football on Sundays", "What does Frank watch on Sundays?", "football"),
]

FILLER_PHRASES = [
    "How's your day going?",
    "Did you do anything fun this weekend?",
    "It's been a hectic week.",
    "Have you tried the new coffee shop?",
    "I saw a really good movie last night.",
    "Work has been pretty busy lately.",
    "I've been trying to sleep earlier.",
    "The weather has been wild.",
]


class LowFrequencyBenchmark(SyntheticBenchmark):
    def __init__(
        self,
        seed: int = 42,
        *,
        high_freq_mentions: int = 10,
        total_turns: int = 100,
        low_freq_insert_turn: int = 5,
    ) -> None:
        super().__init__(seed=seed)
        self.high_freq_mentions = high_freq_mentions
        self.total_turns = total_turns
        self.low_freq_insert_turn = low_freq_insert_turn

    def generate_example(self, *, pair_idx: Optional[int] = None) -> BenchmarkExample:
        index = pair_idx if pair_idx is not None else self.rng.randrange(len(LOW_FREQ_FACTS))
        entity, low_fact, low_question, low_answer = LOW_FREQ_FACTS[index]
        _, high_fact, high_question, high_answer = HIGH_FREQ_FACTS[index]

        high_turns = sorted(
            self.rng.sample(
                range(self.low_freq_insert_turn + 5, self.total_turns),
                min(self.high_freq_mentions, self.total_turns - self.low_freq_insert_turn - 5),
            )
        )

        turns: Dict[int, ConversationTurn] = {
            self.low_freq_insert_turn: ConversationTurn(
                speaker="User",
                text=f"Oh, random thing, {entity} {low_fact}.",
                turn_index=self.low_freq_insert_turn,
                metadata={"type": "low_freq", "entity": entity, "fact": low_fact},
            )
        }
        paraphrases = [
            f"{entity} mentioned they {high_fact}.",
            f"As {entity} said, they {high_fact}.",
            f"{entity} always talks about how they {high_fact}.",
            f"Apparently {entity} really {high_fact}.",
            f"{entity} brought up again that they {high_fact}.",
            f"You know {entity}, they {high_fact} every week.",
        ]
        for index, turn_index in enumerate(high_turns):
            turns[turn_index] = ConversationTurn(
                speaker="User",
                text=paraphrases[index % len(paraphrases)],
                turn_index=turn_index,
                metadata={"type": "high_freq", "entity": entity, "fact": high_fact, "mention": index},
            )
        for turn_index in range(1, self.total_turns + 1):
            turns.setdefault(
                turn_index,
                ConversationTurn(
                    speaker="User" if turn_index % 2 else "AI",
                    text=self.rng.choice(FILLER_PHRASES),
                    turn_index=turn_index,
                ),
            )
        return BenchmarkExample(
            id=str(uuid.uuid4()),
            conversation=[turns[index] for index in sorted(turns)],
            qa_pairs=[
                QAPair(low_question, low_answer, "low_frequency", [self.low_freq_insert_turn]),
                QAPair(high_question, high_answer, "high_frequency", high_turns),
            ],
            metadata={"entity": entity},
        )


MUNDANE_FACTS = [
    ("Alice", "went to the dentist last Tuesday", "When did Alice go to the dentist?", "last Tuesday"),
    ("Bob", "bought groceries on Saturday", "When did Bob buy groceries?", "Saturday"),
    ("Carol", "watched TV in the evening", "What did Carol do in the evening?", "watched TV"),
    ("Alice", "took the bus to work", "How did Alice get to work?", "by bus"),
    ("Bob", "ordered pizza for dinner", "What did Bob have for dinner?", "pizza"),
    ("Carol", "watered her plants in the morning", "What did Carol do in the morning?", "watered plants"),
    ("Alice", "checked her email before bed", "What did Alice do before bed?", "checked email"),
    ("Bob", "met a colleague for coffee", "Who did Bob meet for coffee?", "a colleague"),
    ("Carol", "cleaned the kitchen on Sunday", "When did Carol clean the kitchen?", "Sunday"),
    ("Alice", "went for a short walk after lunch", "What did Alice do after lunch?", "went for a walk"),
    ("Bob", "called his mom on the weekend", "Who did Bob call on the weekend?", "his mom"),
    ("Carol", "read a book before sleeping", "What did Carol do before sleeping?", "read a book"),
    ("Alice", "had oatmeal for breakfast", "What did Alice have for breakfast?", "oatmeal"),
    ("Bob", "took a nap in the afternoon", "What did Bob do in the afternoon?", "took a nap"),
    ("Carol", "forgot her umbrella at the office", "What did Carol forget at the office?", "her umbrella"),
]

SURPRISING_FACTS = [
    ("Alice", "won a national chess championship at age 9", "What championship did Alice win as a child?", "national chess championship"),
    ("Bob", "once survived a lightning strike while hiking", "What unusual thing happened to Bob while hiking?", "was struck by lightning"),
    ("Carol", "speaks seven languages fluently", "How many languages does Carol speak?", "seven"),
    ("Alice", "used to be a professional circus acrobat", "What was Alice's unusual former profession?", "circus acrobat"),
    ("Bob", "discovered a previously unknown insect species", "What did Bob discover?", "a new insect species"),
    ("Carol", "holds a patent for an underwater breathing device", "What does Carol hold a patent for?", "underwater breathing device"),
    ("Alice", "was briefly mistaken for a famous celebrity at an airport", "What funny thing happened to Alice at an airport?", "was mistaken for a celebrity"),
    ("Bob", "once cooked dinner for a sitting president", "Who did Bob cook dinner for?", "a president"),
    ("Carol", "completed an Ironman triathlon without any prior training", "What athletic feat did Carol accomplish without training?", "Ironman triathlon"),
    ("Alice", "accidentally adopted three rabbits she found in a parking lot", "How did Alice end up with rabbits?", "found them in a parking lot"),
]


class SurpriseRecallBenchmark(SyntheticBenchmark):
    def __init__(self, seed: int = 42, *, n_mundane: int = 8, n_surprising: int = 3, interleave: bool = True) -> None:
        super().__init__(seed=seed)
        self.n_mundane = n_mundane
        self.n_surprising = n_surprising
        self.interleave = interleave

    def generate_example(self) -> BenchmarkExample:
        mundane = self.rng.sample(MUNDANE_FACTS, min(self.n_mundane, len(MUNDANE_FACTS)))
        surprising = self.rng.sample(SURPRISING_FACTS, min(self.n_surprising, len(SURPRISING_FACTS)))
        facts = [("mundane",) + item for item in mundane] + [("surprising",) + item for item in surprising]
        if self.interleave:
            self.rng.shuffle(facts)

        turns: List[ConversationTurn] = []
        qa_pairs: List[QAPair] = []
        for index, (surprise_level, entity, fact, question, answer) in enumerate(facts):
            turn_index = index * 3 + 1
            turns.append(
                ConversationTurn(
                    speaker="User",
                    text=f"Did I mention that {entity} {fact}?",
                    turn_index=turn_index,
                    metadata={"surprise_level": surprise_level, "entity": entity, "fact": fact},
                )
            )
            turns.append(
                ConversationTurn(
                    speaker="AI",
                    text="Interesting, I will keep that in mind.",
                    turn_index=turn_index + 1,
                )
            )
            qa_pairs.append(
                QAPair(
                    question=question,
                    answer=answer,
                    question_type=surprise_level,
                    required_turns=[turn_index],
                )
            )
        return BenchmarkExample(
            id=str(uuid.uuid4()),
            conversation=turns,
            qa_pairs=qa_pairs,
            metadata={"n_mundane": len(mundane), "n_surprising": len(surprising)},
        )
