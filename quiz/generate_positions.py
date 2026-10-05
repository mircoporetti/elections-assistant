import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from chat.party import Party
from store import reranker, vector_store
from generate_statements import PARTY_LABELS

STATEMENTS_PATH = os.path.join(os.path.dirname(__file__), "statements.json")
POSITIONS_PATH = os.path.join(os.path.dirname(__file__), "positions.json")
PASSAGES = 5
CANDIDATES_PER_QUERY = 12
MODEL = "gpt-4o-mini"

Position = Literal["agree", "neutral", "disagree", "not_addressed"]

LABEL_PROMPT = """You are labelling party positions for a voting advice application (like the German Wahl-O-Mat).
Be strictly neutral and base your answer ONLY on the manifesto passages below, never on outside
knowledge of the party.

Party: {party}
Statement: "{statement}"

Manifesto passages:
{passages}

Decide the party's position on the statement:
- agree: the passages clearly support the statement.
- disagree: the passages clearly oppose it, or commit to something incompatible with it.
- neutral: the passages address the topic but take a mixed, conditional or in-between position.
- not_addressed: the passages do not say enough to tell. Prefer this over guessing.

Then give a short VERBATIM quote (one or two sentences, copied exactly, max 300 characters) from
the passage that best supports your decision, and that passage's number. For not_addressed,
leave the quote empty and use passage 0."""

VERIFY_PROMPT = """You are checking someone else's work for a voting advice application. Be sceptical.

Party: {party}
Statement: "{statement}"
Claimed position: {position}
Evidence quote from the party's manifesto: "{quote}"

Surrounding passage:
{passage}

Based ONLY on this passage, what is the party's position on the statement
(agree / neutral / disagree / not_addressed)? Answer independently of the claimed position,
then say whether the claimed position is supported by this evidence."""


class Label(BaseModel):
    position: Position
    passage: int = Field(description="Number of the passage the quote comes from, 0 if none")
    quote: str = Field(description="Verbatim quote from that passage, empty if not_addressed")
    reasoning: str = Field(description="One sentence explaining the decision")


class Verification(BaseModel):
    position: Position
    claimed_is_supported: bool
    reasoning: str = Field(description="One sentence")


def normalise(text):
    return re.sub(r"\s+", " ", text.replace("­", "")).strip().lower()


def citation_for(doc):
    return f"{doc.metadata['source'].split('/')[-1].replace('.pdf', '')} p.{doc.metadata['page'] + 1}"


def passages_for(party, statement):
    seen, pool = set(), []
    for query in (statement["statement_de"], statement["query"]):
        for doc in vector_store.candidates_for(party, query, CANDIDATES_PER_QUERY):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                pool.append(doc)
    return reranker.rerank(statement["statement_de"], pool, PASSAGES)


def label(labeller, verifier, statement, party, docs):
    passages = "\n\n".join(f"[{i}] ({citation_for(d)})\n{d.page_content}" for i, d in enumerate(docs, 1))
    party_label = PARTY_LABELS[party.name]

    result = labeller.invoke(LABEL_PROMPT.format(party=party_label, statement=statement["statement_de"],
                                                 passages=passages))
    flags = []
    entry = {"statement_id": statement["id"], "party": party.name, "position": result.position,
             "quote": result.quote, "source": None, "reasoning": result.reasoning}

    if result.position == "not_addressed":
        flags.append("not_addressed")
        return entry | {"flags": flags}

    doc = docs[result.passage - 1] if 1 <= result.passage <= len(docs) else None
    if doc is None:
        flags.append("invalid_passage_number")
    else:
        entry["source"] = citation_for(doc)
        if normalise(result.quote) not in normalise(doc.page_content):
            flags.append("quote_not_verbatim")

        check = verifier.invoke(VERIFY_PROMPT.format(party=party_label, statement=statement["statement_de"],
                                                     position=result.position, quote=result.quote,
                                                     passage=doc.page_content))
        entry["verifier_position"] = check.position
        entry["verifier_reasoning"] = check.reasoning
        if check.position != result.position or not check.claimed_is_supported:
            flags.append("verifier_disagrees")

    return entry | {"flags": flags}


def main():
    vector_store.init()
    reranker.load()
    labeller = ChatOpenAI(model=MODEL, temperature=0, max_retries=8).with_structured_output(Label)
    verifier = ChatOpenAI(model=MODEL, temperature=0, max_retries=8).with_structured_output(Verification)
    statements = json.load(open(STATEMENTS_PATH, encoding="utf-8"))

    # Retrieval runs sequentially up front (CPU-bound, shared models); only LLM calls run in parallel.
    jobs = [(s, p, passages_for(p, s)) for s in statements for p in Party]
    print(f"retrieved passages for {len(jobs)} statement/party pairs, labelling...")
    with ThreadPoolExecutor(max_workers=3) as pool:
        positions = list(pool.map(lambda job: label(labeller, verifier, *job), jobs))

    with open(POSITIONS_PATH, "w", encoding="utf-8") as out:
        json.dump(positions, out, ensure_ascii=False, indent=2)

    flagged = [p for p in positions if p["flags"]]
    print(f"wrote {len(positions)} positions to {POSITIONS_PATH}, {len(flagged)} flagged")


if __name__ == "__main__":
    main()
