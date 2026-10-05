import base64
import os

from fastapi.testclient import TestClient

from src.webapp.main import app

client = TestClient(app)

username = os.getenv("API_USERNAME")
password = os.getenv("API_PASSWORD")
basic_auth = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("utf-8")
headers = {"Authorization": f"Basic {basic_auth}"}

PARTIES = {"SPD", "CDU", "AFD", "FDP", "DL", "DG", "BSW"}


def tests_quiz_requires_authentication():
    assert client.get("/api/quiz").status_code == 401
    assert client.post("/api/quiz/score", json={"answers": {}}).status_code == 401


def tests_quiz_returns_german_statements_by_default():
    response = client.get("/api/quiz", headers=headers)

    assert response.status_code == 200
    quiz = response.json()
    assert set(quiz["parties"]) == PARTIES
    assert len(quiz["statements"]) == 30
    conscription = next(s for s in quiz["statements"] if s["id"] == "conscription")
    assert "Wehrpflicht" in conscription["text"]


def tests_quiz_returns_english_statements():
    response = client.get("/api/quiz", params={"lang": "en"}, headers=headers)

    assert response.status_code == 200
    conscription = next(s for s in response.json()["statements"] if s["id"] == "conscription")
    assert "military service" in conscription["text"]


def tests_quiz_does_not_reveal_party_positions_before_answering():
    statement = client.get("/api/quiz", headers=headers).json()["statements"][0]

    assert set(statement) == {"id", "area", "text"}


def tests_quiz_rejects_unsupported_language():
    assert client.get("/api/quiz", params={"lang": "fr"}, headers=headers).status_code == 422


def tests_score_ranks_the_party_whose_positions_were_given_first():
    statements = client.get("/api/quiz", headers=headers).json()["statements"]
    answers = {s["id"]: "skip" for s in statements}
    answers.update({"debt-brake": "disagree", "wealth-tax": "disagree", "nuclear-power": "agree",
                    "speed-limit": "disagree", "self-determination": "agree", "cannabis": "agree"})

    response = client.post("/api/quiz/score", json={"answers": answers, "weighted": ["nuclear-power"]},
                           headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert {r["party"] for r in body["results"]} == PARTIES
    assert body["results"][0]["party"] == "CDU"
    assert body["results"][0]["percentage"] == 100.0
    assert body["results"][0]["statements_compared"] == 6


def tests_score_returns_cited_positions_for_answered_statements_only():
    response = client.post("/api/quiz/score", json={"answers": {"conscription": "agree", "cannabis": "skip"},
                                                    "lang": "en"}, headers=headers)

    statements = response.json()["statements"]
    assert [s["id"] for s in statements] == ["conscription"]
    assert statements[0]["answer"] == "agree"
    assert "military service" in statements[0]["text"]
    fdp = statements[0]["positions"]["FDP"]
    assert fdp["position"] == "disagree"
    assert fdp["source"] == "FDP p.44"
    assert "conscription" in fdp["quote"]


def tests_score_rejects_unknown_statement():
    response = client.post("/api/quiz/score", json={"answers": {"nope": "agree"}}, headers=headers)

    assert response.status_code == 400
    assert "nope" in response.json()["detail"]


def tests_score_rejects_invalid_answer():
    response = client.post("/api/quiz/score", json={"answers": {"conscription": "maybe"}}, headers=headers)

    assert response.status_code == 422
