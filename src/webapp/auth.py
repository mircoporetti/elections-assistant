import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

username = os.getenv("API_USERNAME")
password = os.getenv("API_PASSWORD")
if not username or not password:
    raise ValueError("API_USERNAME or API_PASSWORD environment variable is not set.")


def matches(provided: str, expected: str):
    return secrets.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def unauthorized():
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Basic"},
    )


def basic_auth(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = matches(credentials.username, username)
    correct_password = matches(credentials.password, password)
    if correct_username and correct_password:
        return True
    raise unauthorized()


