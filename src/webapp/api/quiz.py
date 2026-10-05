from dataclasses import asdict
from typing import Dict, List, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from quiz.scoring import SKIP, InvalidAnswersError, load_quiz, score
from ..auth import basic_auth

router = APIRouter(prefix="/api/quiz", tags=["quiz"])

Lang = Literal["de", "en"]
Answer = Literal["agree", "neutral", "disagree", "skip"]


class ScoreRequest(BaseModel):
    answers: Dict[str, Answer]
    weighted: List[str] = []
    lang: Lang = "de"


@router.get("")
async def get_quiz(lang: Lang = "de", credentials=Depends(basic_auth)):
    quiz = load_quiz()
    return {
        "parties": quiz["parties"],
        "statements": [
            {"id": statement["id"], "area": statement["area"], "text": statement["text"][lang]}
            for statement in quiz["statements"]
        ],
    }


@router.post("/score")
async def score_answers(request: ScoreRequest, credentials=Depends(basic_auth)):
    try:
        results = score(request.answers, request.weighted)
    except InvalidAnswersError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return {"results": [asdict(result) for result in results], "statements": comparison(request)}


def comparison(request: ScoreRequest):
    """Each answered statement next to every party's position and the manifesto quote behind it."""
    weighted = set(request.weighted)
    return [
        {
            "id": statement["id"],
            "text": statement["text"][request.lang],
            "answer": request.answers[statement["id"]],
            "weighted": statement["id"] in weighted,
            "positions": statement["positions"],
        }
        for statement in load_quiz()["statements"]
        if request.answers.get(statement["id"], SKIP) != SKIP
    ]
