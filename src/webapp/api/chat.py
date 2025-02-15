from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from chat import ai_assistant
from .middleware import track_daily_calls
from ..auth import basic_auth

from src.webapp.properties import Properties
from lingua import Language, LanguageDetectorBuilder

router = APIRouter(prefix="/api/chat", tags=["chat"])

languages = [Language.ENGLISH, Language.GERMAN]
detector = LanguageDetectorBuilder.from_languages(*languages).build()


class CompletionRequest(BaseModel):
    history: List
    question: str


class ChunksRetrievalRequest(BaseModel):
    question: str


@router.post("/completion")
async def answer_question(request: CompletionRequest, credentials=Depends(basic_auth), _=Depends(track_daily_calls)):
    language = detector.detect_language_of(request.question)
    Properties.user_lang = language

    result = ai_assistant.answer(request.question, request.history, language)
    return {"answer": result}


@router.post("/retrieve")
async def retrieve_most_pertinent_chunks(request: ChunksRetrievalRequest, credentials=Depends(basic_auth)):
    result = ai_assistant.answer_with_most_pertinent_chunks(request.question)
    return {"chunks": result}
