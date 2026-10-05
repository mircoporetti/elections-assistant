import pytest

from quiz.scoring import InvalidAnswersError, load_quiz, points_for, score


def position(value):
    return {"position": value, "source": None, "quote": None}


QUIZ = {
    "parties": ["A", "B", "C"],
    "statements": [
        {"id": "s1", "positions": {"A": position("agree"), "B": position("disagree"), "C": position("neutral")}},
        {"id": "s2", "positions": {"A": position("agree"), "B": position("agree"), "C": position("not_addressed")}},
        {"id": "s3", "positions": {"A": position("disagree"), "B": position("agree"), "C": position("not_addressed")}},
    ],
}


def by_party(results):
    return {result.party: result for result in results}


@pytest.mark.parametrize("user, party, expected", [
    ("agree", "agree", 2), ("disagree", "disagree", 2), ("neutral", "neutral", 2),
    ("agree", "neutral", 1), ("neutral", "disagree", 1),
    ("agree", "disagree", 0), ("disagree", "agree", 0),
])
def tests_points_follow_wahl_o_mat_rules(user, party, expected):
    assert points_for(user, party) == expected


def tests_identical_answers_give_full_affinity():
    results = by_party(score({"s1": "agree", "s2": "agree", "s3": "disagree"}, quiz=QUIZ))

    assert results["A"].percentage == 100.0
    assert results["B"].percentage == pytest.approx(33.3)


def tests_results_are_sorted_highest_first():
    results = score({"s1": "disagree", "s2": "agree", "s3": "agree"}, quiz=QUIZ)

    assert [result.party for result in results] == ["B", "C", "A"]


def tests_skipped_statements_count_for_nobody():
    results = by_party(score({"s1": "agree", "s2": "skip"}, quiz=QUIZ))

    assert results["A"].statements_compared == 1
    assert results["A"].max_points == 2
    assert results["A"].percentage == 100.0


def tests_unanswered_statements_are_treated_as_skipped():
    assert score({"s1": "agree"}, quiz=QUIZ) == score({"s1": "agree", "s2": "skip", "s3": "skip"}, quiz=QUIZ)


def tests_not_addressed_is_left_out_of_the_party_maximum():
    results = by_party(score({"s1": "neutral", "s2": "agree", "s3": "agree"}, quiz=QUIZ))

    assert results["C"].statements_compared == 1
    assert results["C"].max_points == 2
    assert results["C"].percentage == 100.0


def tests_double_weight_counts_twice():
    answers = {"s1": "agree", "s2": "disagree"}

    plain = by_party(score(answers, quiz=QUIZ))["A"]
    weighted = by_party(score(answers, weighted=["s1"], quiz=QUIZ))["A"]

    assert (plain.points, plain.max_points) == (2, 4)
    assert (weighted.points, weighted.max_points) == (4, 6)
    assert weighted.percentage == pytest.approx(66.7)


def tests_party_with_nothing_to_compare_has_no_percentage_and_sorts_last():
    results = score({"s2": "agree", "s3": "agree"}, quiz=QUIZ)

    assert results[-1].party == "C"
    assert results[-1].percentage is None
    assert results[-1].statements_compared == 0


def tests_all_skipped_gives_no_percentages():
    assert all(result.percentage is None for result in score({}, quiz=QUIZ))


def tests_unknown_statement_is_rejected():
    with pytest.raises(InvalidAnswersError, match="nope"):
        score({"nope": "agree"}, quiz=QUIZ)


def tests_unknown_weighted_statement_is_rejected():
    with pytest.raises(InvalidAnswersError, match="nope"):
        score({"s1": "agree"}, weighted=["nope"], quiz=QUIZ)


def tests_invalid_answer_is_rejected():
    with pytest.raises(InvalidAnswersError, match="maybe"):
        score({"s1": "maybe"}, quiz=QUIZ)


def tests_scores_the_real_quiz_for_every_party():
    quiz = load_quiz()
    answers = {statement["id"]: "agree" for statement in quiz["statements"]}

    results = score(answers)

    assert {result.party for result in results} == set(quiz["parties"])
    assert all(0 <= result.percentage <= 100 for result in results)
