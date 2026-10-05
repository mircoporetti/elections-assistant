import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

QUIZ_PATH = os.path.join(os.path.dirname(__file__), "quiz_data.json")

ANSWERS = {"agree": 1, "neutral": 0, "disagree": -1}
SKIP = "skip"
NOT_ADDRESSED = "not_addressed"
MAX_POINTS_PER_STATEMENT = 2
DOUBLE_WEIGHT = 2


class InvalidAnswersError(ValueError):
    pass


@dataclass
class PartyAffinity:
    party: str
    percentage: Optional[float]
    points: int
    max_points: int
    statements_compared: int


@lru_cache(maxsize=1)
def load_quiz(path: str = QUIZ_PATH):
    with open(path, encoding="utf-8") as data:
        return json.load(data)


def points_for(user_answer: str, party_position: str) -> int:
    return MAX_POINTS_PER_STATEMENT - abs(ANSWERS[user_answer] - ANSWERS[party_position])


def validate(quiz, answers: Dict[str, str], weighted: Iterable[str]):
    statement_ids = {statement["id"] for statement in quiz["statements"]}
    unknown = (set(answers) | set(weighted)) - statement_ids
    if unknown:
        raise InvalidAnswersError(f"Unknown statement ids: {', '.join(sorted(unknown))}")
    invalid = {sid: answer for sid, answer in answers.items() if answer not in ANSWERS and answer != SKIP}
    if invalid:
        raise InvalidAnswersError(f"Invalid answers: {invalid}. Use agree, neutral, disagree or skip")


def score(answers: Dict[str, str], weighted: Iterable[str] = (), quiz=None) -> List[PartyAffinity]:
    """Affinity per party, highest first.

    Skipped statements count for nobody. A statement the party's manifesto does not address is
    left out of that party's maximum, so parties are not rewarded or punished for silence;
    statements_compared says how many statements each percentage rests on.
    """
    quiz = quiz or load_quiz()
    weighted = set(weighted)
    validate(quiz, answers, weighted)

    results = []
    for party in quiz["parties"]:
        points = max_points = compared = 0
        for statement in quiz["statements"]:
            user_answer = answers.get(statement["id"], SKIP)
            party_position = statement["positions"][party]["position"]
            if user_answer == SKIP or party_position == NOT_ADDRESSED:
                continue
            weight = DOUBLE_WEIGHT if statement["id"] in weighted else 1
            points += weight * points_for(user_answer, party_position)
            max_points += weight * MAX_POINTS_PER_STATEMENT
            compared += 1
        percentage = round(100 * points / max_points, 1) if max_points else None
        results.append(PartyAffinity(party, percentage, points, max_points, compared))

    return sorted(results, key=lambda result: (result.percentage is None, -(result.percentage or 0)))
