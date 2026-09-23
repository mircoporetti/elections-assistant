import base64
import os

from fastapi.testclient import TestClient

from src.webapp.main import app
from src.webapp.properties import Properties
import pytest

from store import vector_store

client = TestClient(app)

username = os.getenv("API_USERNAME")
password = os.getenv("API_PASSWORD")
basic_auth = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("utf-8")


@pytest.fixture(scope="session", autouse=True)
def init_vector_store():
    vector_store.init()
    print("Vector store initialized")


def assert_mentions_immigration(answer):
    assert 'immigrant' in answer.lower() or 'immigration' in answer.lower(), \
        f"expected the answer to mention immigration, got: {answer!r}"


def tests_chat_completion():
    response = client.post("/api/chat/completion", json={
        "history": [
            {
                "role": "You",
                "content": "What CDU wants to do for immigrants?"
            }
        ],
        "question": "What CDU wants to do for immigrants?"
    },
                           headers={"Authorization": f"Basic {basic_auth}"})

    assert response.status_code == 200

    response_json = response.json()
    assert 'answer' in response_json
    assert 'CDU' in response_json['answer']
    assert_mentions_immigration(response_json['answer'])


def tests_chat_party_inferred_from_history():
    response = client.post("/api/chat/completion", json={
        "history": [
            {
                "role": "You",
                "content": "What CDU wants to do for economy?"
            },
            {
                "role": "AI",
                "content": "CDU wants to actuate the Plan X"
            },
            {
                "role": "You",
                "content": "What do they wants to do for immigrants?"
            }
        ],
        "question": "What CDU wants to do for immigrants?"
    },
                           headers={"Authorization": f"Basic {basic_auth}"})

    assert response.status_code == 200

    response_json = response.json()
    assert 'answer' in response_json
    assert 'CDU' in response_json['answer']
    assert_mentions_immigration(response_json['answer'])


def tests_chat_doesnt_support_party():
    question = "What do my favorite party wants to do for economy?"
    response = client.post("/api/chat/completion", json={
        "history": [
            {
                "role": "You",
                "content": question
            }
        ],
        "question": question
    },
                           headers={"Authorization": f"Basic {basic_auth}"})

    assert response.status_code == 404

    response_json = response.json()

    error_detail = response_json['detail']
    print(error_detail)
    assert error_detail == ("Oops! I did not understand which party your question is referring to."
                            " Please include one of the following: SPD, CDU, AFD, FDP, DL, DG, BSW")


def tests_chatbot_understand_old_questions():
    question = "What did I ask you in the previous question?"
    response = client.post("/api/chat/completion", json={
        "history": [
            {
                "role": "You",
                "content": "What CDU wants to do for economy?"
            },
            {
                "role": "AI",
                "content": "CDU wants to actuate the Plan X"
            },
            {
                "role": "You",
                "content": question
            }
        ],
        "question": question
    },
                           headers={"Authorization": f"Basic {basic_auth}"})

    assert response.status_code == 200

    response_json = response.json()
    answer = response_json['answer'].lower()
    assert 'economy' in answer or 'cdu' in answer, \
        f"expected the answer to recall the previous question, got: {response_json['answer']!r}"


@pytest.fixture
def no_calls_allowed():
    original_limit = Properties.call_limit
    Properties.call_limit = 0
    yield
    Properties.call_limit = original_limit


def tests_chatbot_received_too_many_daily_requests(no_calls_allowed):
    question = "What did I ask you in the previous question?"
    response = client.post("/api/chat/completion", json={
        "history": [
            {
                "role": "You",
                "content": question
            }
        ],
        "question": question
    },
                           headers={"Authorization": f"Basic {basic_auth}"})

    assert response.status_code == 429


def tests_chat_returns_most_pertinent_chunks():
    response = client.post("/api/chat/retrieve", json={
        "question": "What CDU wants to do for illegal immigration?"
    },
                           headers={"Authorization": f"Basic {basic_auth}"})

    assert response.status_code == 200

    response_json = response.json()
    assert 'chunks' in response_json
    assert len(response_json['chunks']) == 4


def tests_chat_returns_only_cdu_chunks():
    response = client.post("/api/chat/retrieve", json={
        "question": "What CDU wants to do for economy?"},
                           headers={"Authorization": f"Basic {basic_auth}"})

    assert response.status_code == 200

    response_json = response.json()

    chunks = response_json['chunks']
    assert len(chunks) > 0
    assert all('CDU' in chunk['text']['metadata']['source'] for chunk in chunks)
