"""Merge statements, generated positions and reviewed overrides into the quiz data file the app serves.

Every quote must appear verbatim on the cited manifesto page. Generated quotes that the LLM paraphrased
are snapped to the closest real passage on that page; if nothing close enough exists the build fails.
"""
import json
import os
import re
import sys
from difflib import SequenceMatcher
from functools import lru_cache

from pypdf import PdfReader

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
STATEMENTS_PATH = os.path.join(HERE, "statements.json")
POSITIONS_PATH = os.path.join(HERE, "positions.json")
OVERRIDES_PATH = os.path.join(HERE, "position_overrides.json")
QUIZ_PATH = os.path.join(ROOT, "src", "quiz", "quiz_data.json")
MANIFESTS_PATH = os.path.join(ROOT, "resources", "manifests")

PARTIES = ["SPD", "CDU", "AFD", "FDP", "DL", "DG", "BSW"]
POSITIONS = {"agree", "neutral", "disagree", "not_addressed"}
MIN_SNAP_SIMILARITY = 0.75
MAX_SENTENCES = 3


def normalise(text):
    return re.sub(r"\s+", " ", text.replace("­", "")).strip()


@lru_cache(None)
def page_text(party, page):
    reader = PdfReader(os.path.join(MANIFESTS_PATH, f"{party}.pdf"))
    if not 1 <= page <= len(reader.pages):
        return ""
    text = reader.pages[page - 1].extract_text() or ""
    # Quotes may run onto the next page, as the Greens' unanimity sentence does.
    if page < len(reader.pages):
        text += " " + (reader.pages[page].extract_text() or "")
    return normalise(text)


def parse_source(source):
    party, page = source.split(" p.")
    return party, int(page)


def contains(haystack, needle):
    return normalise(needle).lower() in haystack.lower()


def snap_to_page(quote, text):
    """Return the run of 1-3 consecutive sentences on the page that best matches the quote."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    target = normalise(quote).lower()
    best, best_score = None, 0.0
    for start in range(len(sentences)):
        for length in range(1, MAX_SENTENCES + 1):
            candidate = " ".join(sentences[start:start + length])
            score = SequenceMatcher(None, target, candidate.lower()).ratio()
            if score > best_score:
                best, best_score = candidate, score
    return best, best_score


def build():
    statements = json.load(open(STATEMENTS_PATH, encoding="utf-8"))
    generated = {(p["statement_id"], p["party"]): p for p in json.load(open(POSITIONS_PATH, encoding="utf-8"))}
    overrides = {(o["statement_id"], o["party"]): o for o in json.load(open(OVERRIDES_PATH, encoding="utf-8"))}

    errors, snapped = [], []
    quiz_statements = []
    for statement in statements:
        positions = {}
        for party in PARTIES:
            key = (statement["id"], party)
            entry = overrides.get(key) or generated.get(key)
            if entry is None:
                errors.append(f"{key}: no position")
                continue
            position, source, quote = entry["position"], entry.get("source"), entry.get("quote", "")
            if position not in POSITIONS:
                errors.append(f"{key}: unknown position {position!r}")
                continue

            if position == "not_addressed":
                positions[party] = {"position": position, "source": None, "quote": None}
                continue
            if not source or not quote:
                errors.append(f"{key}: {position} needs a source and a quote")
                continue

            source_party, page = parse_source(source)
            if source_party != party:
                errors.append(f"{key}: cites another party's manifesto ({source})")
                continue
            text = page_text(party, page)
            if not contains(text, quote):
                if key in overrides:
                    errors.append(f"{key}: reviewed quote not found on {source}")
                    continue
                replacement, score = snap_to_page(quote, text)
                if score < MIN_SNAP_SIMILARITY:
                    errors.append(f"{key}: quote not on {source} (closest match {score:.2f})")
                    continue
                snapped.append((key, score, quote, replacement))
                quote = replacement

            positions[party] = {"position": position, "source": source, "quote": normalise(quote).lstrip("•-– ")}

        quiz_statements.append({
            "id": statement["id"],
            "area": statement["area"],
            "text": {"de": statement["statement_de"], "en": statement["statement_en"]},
            "positions": positions,
        })

    return quiz_statements, snapped, errors


def main():
    quiz_statements, snapped, errors = build()

    for key, score, original, replacement in snapped:
        print(f"snapped {key[0]}/{key[1]} ({score:.2f})\n   was: {original[:150]}\n   now: {replacement[:150]}")
    if errors:
        print("\nBUILD FAILED:\n  " + "\n  ".join(errors))
        sys.exit(1)

    with open(QUIZ_PATH, "w", encoding="utf-8") as out:
        json.dump({"parties": PARTIES, "statements": quiz_statements}, out, ensure_ascii=False, indent=2)
    print(f"\nwrote {len(quiz_statements)} statements to {QUIZ_PATH}, {len(snapped)} quotes snapped")


if __name__ == "__main__":
    main()
