import base64
import os

from fastapi.testclient import TestClient

from src.webapp.main import app
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


def tests_store_cleanup_requires_confirmation():
    response = client.post("/api/store/clean", headers={"Authorization": f"Basic {basic_auth}"})

    assert response.status_code == 400
    assert os.path.exists("faiss/index.faiss"), "the index must survive an unconfirmed request"


def tests_store_cleanup():
    response = client.post("/api/store/clean?confirm=delete-the-index",
                           headers={"Authorization": f"Basic {basic_auth}"})

    assert response.status_code == 200

    remaining = [file for file in os.listdir("faiss") if file != "init.lock"]
    assert remaining == []


