import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_openai import ChatOpenAI

from chat.party import Party
from store import vector_store

FOLLOWUP_SET_PATH = os.path.join(os.path.dirname(__file__), "followup_set.json")

REWRITE_PROMPT = (
    "Rewrite the user's latest question as a standalone search query.\n"
    "Resolve pronouns and implicit references using the conversation, and keep the "
    "original language.\n"
    "Keep every topical term from the question. Do not answer it, do not add "
    "information that is not there, and output only the rewritten query.\n\n"
    "Conversation:\n{conversation}\n\n"
    "Latest question: {question}\n\n"
    "Standalone query:"
)

rewriter_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=2)


def rewrite_as_standalone(question, history):
    previous = [message for message in history if message["content"].strip() != question.strip()]
    if not previous:
        return question

    conversation = "\n".join(
        f"{'Assistant' if message['role'].lower() == 'ai' else 'User'}: {message['content']}"
        for message in previous[-4:]
    )
    return rewriter_llm.invoke(
        REWRITE_PROMPT.format(conversation=conversation, question=question)
    ).content.strip()


def chunk_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def retrieve(query, party, k):
    party_filter = vector_store.party_filter_for(party)
    return vector_store.vector_store.similarity_search(
        query, k=k, filter=party_filter, fetch_k=vector_store.fetch_k_for(party_filter)
    )


def evaluate(label, query_for, followup_set, k):
    hits = 0
    reciprocal_ranks = 0.0
    party_correct = 0

    for item in followup_set:
        history = item["history"] + [{"role": "You", "content": item["followup_question"]}]
        party = Party.get_from_history(history)
        if party.name == item["expected_party"]:
            party_correct += 1

        docs = retrieve(query_for(item, history), party, k)
        for position, doc in enumerate(docs, start=1):
            if chunk_id(doc.page_content) == item["expected_chunk_id"]:
                hits += 1
                reciprocal_ranks += 1.0 / position
                break

    total = len(followup_set)
    print(f"{label}")
    print(f"  recall@{k} : {hits}/{total} = {100 * hits / total:.1f}%")
    print(f"  MRR@{k}    : {reciprocal_ranks / total:.3f}")
    print(f"  party routing (from history): {party_correct}/{total}")
    print()
    return hits / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    with open(FOLLOWUP_SET_PATH, encoding="utf-8") as source:
        followup_set = json.load(source)

    vector_store.init()
    print(f"follow-up set: {len(followup_set)} pronoun-only questions, k={args.k}")
    print()

    raw = evaluate(
        "A. raw follow-up as query (current)",
        lambda item, history: item["followup_question"],
        followup_set, args.k)

    rewritten = evaluate(
        "B. LLM-rewritten standalone query",
        lambda item, history: rewrite_as_standalone(item["followup_question"], history),
        followup_set, args.k)

    ceiling = evaluate(
        "C. original self-contained question (ceiling)",
        lambda item, history: item["question"],
        followup_set, args.k)

    print(f"rewriting gain: {100 * (rewritten - raw):+.1f} pts "
          f"(ceiling is {100 * (ceiling - raw):+.1f} pts)")


if __name__ == "__main__":
    main()