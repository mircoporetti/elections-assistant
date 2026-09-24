import os
from typing import List, Dict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from lingua.lingua import Language

from chat.party import Party, extract_party_from
from chat.prompt import system_english_prompt, system_german_prompt
from store import vector_store
from store.vector_store import similarity_search_for

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI API KEY environment variable is not set.")

PREVIOUS_EXCHANGES = 6


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2
)


def answer(question: str, history: List[Dict[str, str]], user_language: Language):
    system_prompt = system_english_prompt if user_language == Language.ENGLISH else system_german_prompt

    party = Party.get_from_history(history, user_language)
    context_docs = vector_store.most_relevant_for(party, question)
    context = format_context(context_docs)

    conversation = [SystemMessage(content=system_prompt.format(context=context, party=party.name))]
    for message in previous_exchanges(history, question):
        if message["role"].lower() != "ai":
            conversation.append(HumanMessage(content=message["content"]))
        else:
            conversation.append(AIMessage(content=message["content"]))
    conversation.append(HumanMessage(content=question))

    return llm.invoke(conversation).content, sources_of(context_docs)


def format_context(context_docs):
    passages = []
    for doc in context_docs:
        passages.append(f"[{citation_for(doc)}]\n{doc.page_content}")
    return "\n\n".join(passages)


def citation_for(doc):
    party = os.path.basename(doc.metadata.get("source", "")).replace(".pdf", "")
    page = doc.metadata.get("page")
    if page is None:
        return party
    return f"{party} p.{page + 1}"


def sources_of(context_docs):
    seen = []
    for doc in context_docs:
        citation = citation_for(doc)
        if citation not in seen:
            seen.append(citation)
    return seen


def previous_exchanges(history: List[Dict[str, str]], question: str):
    exchanges = list(history)
    while exchanges and is_same_question(exchanges[-1], question):
        exchanges.pop()
    return exchanges[-PREVIOUS_EXCHANGES:]


def is_same_question(message: Dict[str, str], question: str):
    return message["role"].lower() != "ai" and message["content"].strip() == question.strip()


def answer_with_most_pertinent_chunks(question: str, user_language: Language = Language.ENGLISH):
    return similarity_search_for(extract_party_from(question, user_language), question)
