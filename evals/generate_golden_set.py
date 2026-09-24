import hashlib
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_openai import ChatOpenAI

from store.vector_store import chunk_manifests_pdfs

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.json")
QUESTIONS_PER_PARTY = 8
MIN_CHUNK_CHARS = 900
SEED = 20260923

PARTY_BY_FILENAME = {
    "SPD": "SPD", "CDU": "CDU", "AFD": "AfD", "FDP": "FDP",
    "DL": "Die Linke", "DG": "Die Grünen", "BSW": "BSW",
}

GENERATION_PROMPT = """Here is an excerpt from the {party_label} German federal election manifesto.

Write ONE question that a voter would plausibly ask, which this excerpt answers.

Rules:
- The question MUST contain the party name "{party_label}".
- Write it in {language}.
- Ask about the substance (a policy, a position, a promise) — do not refer to
  "the excerpt" or "the text".
- It must be answerable from this excerpt alone, and specific enough that a
  different party's manifesto would not answer it.
- Output ONLY the question, nothing else.

Excerpt:
{excerpt}"""


def party_from_source(source):
    return os.path.basename(source).replace(".pdf", "")


def chunk_id(chunk):
    return hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:16]


def main():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_retries=3)
    random.seed(SEED)

    by_party = defaultdict(list)
    for chunk in chunk_manifests_pdfs():
        if len(chunk.page_content) >= MIN_CHUNK_CHARS:
            by_party[party_from_source(chunk.metadata["source"])].append(chunk)

    golden_set = []
    for party, chunks in sorted(by_party.items()):
        sampled = random.sample(chunks, min(QUESTIONS_PER_PARTY, len(chunks)))
        for index, chunk in enumerate(sampled):
            language = "German" if index % 4 else "English"
            prompt = GENERATION_PROMPT.format(
                party_label=PARTY_BY_FILENAME[party],
                language=language,
                excerpt=chunk.page_content[:2000],
            )
            question = llm.invoke(prompt).content.strip().strip('"')
            golden_set.append({
                "question": question,
                "language": language,
                "expected_party": party,
                "expected_chunk_id": chunk_id(chunk),
                "expected_source": chunk.metadata["source"],
                "expected_page": chunk.metadata.get("page"),
            })
            print(f"[{party}] {question}")

    with open(GOLDEN_SET_PATH, "w", encoding="utf-8") as out:
        json.dump(golden_set, out, ensure_ascii=False, indent=2)
    print(f"\nwrote {len(golden_set)} items to {GOLDEN_SET_PATH}")


if __name__ == "__main__":
    main()