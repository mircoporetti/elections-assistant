import json
import os

import pytest

QUIZ_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "src", "quiz", "quiz_data.json")
POSITIONS = {"agree", "neutral", "disagree", "not_addressed"}


@pytest.fixture(scope="module")
def quiz():
    with open(QUIZ_PATH, encoding="utf-8") as data:
        return json.load(data)


def tests_quiz_has_thirty_unique_statements(quiz):
    ids = [statement["id"] for statement in quiz["statements"]]
    assert len(ids) == 30
    assert len(set(ids)) == len(ids), "statement ids must be unique"


def tests_every_statement_is_bilingual(quiz):
    for statement in quiz["statements"]:
        assert statement["text"]["de"].strip(), f"{statement['id']} has no German text"
        assert statement["text"]["en"].strip(), f"{statement['id']} has no English text"


def tests_every_party_has_a_position_on_every_statement(quiz):
    for statement in quiz["statements"]:
        assert set(statement["positions"]) == set(quiz["parties"]), statement["id"]
        for party, position in statement["positions"].items():
            assert position["position"] in POSITIONS, f"{statement['id']}/{party}"


def tests_every_stated_position_is_cited_from_its_own_manifesto(quiz):
    for statement in quiz["statements"]:
        for party, position in statement["positions"].items():
            if position["position"] == "not_addressed":
                assert position["source"] is None and position["quote"] is None, f"{statement['id']}/{party}"
            else:
                assert position["quote"], f"{statement['id']}/{party} has no quote"
                assert position["source"].startswith(f"{party} p."), f"{statement['id']}/{party}"
