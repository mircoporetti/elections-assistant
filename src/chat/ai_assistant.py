import os
from typing import List, Dict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from lingua.lingua import Language

from chat.party import Party
from chat.prompt import system_english_prompt, system_german_prompt
from store import vector_store
from store.vector_store import similarity_search_for

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI API KEY environment variable is not set.")


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2
)


def answer(question: str, history: List[Dict[str, str]], user_language: Language):
    system_prompt = system_english_prompt if user_language == Language.ENGLISH else system_german_prompt

    party = Party.get_from_history(history)
    retriever = vector_store.get_store_as_retriever_for(party)
    context_docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in context_docs])

    conversation = [SystemMessage(content=system_prompt.format(context=context, party=party.name))]
    for message in history[-2:]:
        if message["role"].lower() != "ai":
            conversation.append(HumanMessage(content=message["content"]))
        else:
            conversation.append(AIMessage(content=message["content"]))

    return llm.invoke(conversation).content


def answer_with_most_pertinent_chunks(question: str):
    best_chunks = similarity_search_for(Party.get_from_message(question), question)

    return best_chunks
