import re
from enum import Enum

from lingua.lingua import Language

PARTY_ALIASES = {
    "SPD": ["SPD", "Sozialdemokraten", "Sozialdemokratische Partei"],
    "CDU": ["CDU", "CSU", "Christdemokraten", "Christlich Demokratische Union"],
    "AFD": ["AfD", "AFD", "Alternative für Deutschland", "Alternative fuer Deutschland"],
    "FDP": ["FDP", "Freie Demokraten", "Freie Demokratische Partei"],
    "DL": ["DL", "Die Linke", "Linkspartei"],
    "DG": ["DG", "Die Grünen", "Die Gruenen", "Bündnis 90", "Buendnis 90"],
    "BSW": ["BSW", "Bündnis Sahra Wagenknecht", "Buendnis Sahra Wagenknecht"],
}

ADJECTIVAL_ALIASES = {"die linke", "die grünen", "die gruenen"}
NOT_FOLLOWED_BY_NOUN = r"(?!\s+(?-i:[A-ZÄÖÜ]))"


def _alias_regex(alias):
    pattern = re.escape(alias) + r"\b"
    if alias.lower() in ADJECTIVAL_ALIASES:
        pattern += NOT_FOLLOWED_BY_NOUN
    return pattern


def _alias_pattern(aliases):
    ordered = sorted(aliases, key=len, reverse=True)
    return re.compile(
        r"\b(?:" + "|".join(_alias_regex(alias) for alias in ordered) + r")",
        re.IGNORECASE,
    )


_PARTY_PATTERNS = {name: _alias_pattern(aliases) for name, aliases in PARTY_ALIASES.items()}


class Party(Enum):
    SPD = "SPD"
    CDU = "CDU"
    AFD = "AFD"
    FDP = "FDP"
    DL = "DL"
    DG = "DG"
    BSW = "BSW"

    @staticmethod
    def get_from_history(history, language=Language.ENGLISH):
        for message in reversed(history):
            if message["role"] != "AI":
                party = Party.get_from_message(message["content"])
                if party:
                    return party
        raise PartyNotFoundError("No party found in chat history", language)

    @staticmethod
    def get_from_message(message):
        if not message:
            return None
        best_party, best_position = None, len(message)
        for party in Party:
            match = _PARTY_PATTERNS[party.name].search(message)
            if match and match.start() < best_position:
                best_party, best_position = party, match.start()
        return best_party


class PartyNotFoundError(Exception):
    def __init__(self, question, language=Language.ENGLISH):
        supported_parties = ", ".join([party.name for party in Party])
        self.question = question
        self.language = language
        if language == Language.GERMAN:
            message = f"Ups! Ich habe nicht verstanden, auf welche Partei sich Ihre Frage bezieht. " \
                    f"Bitte geben Sie eine der folgenden Parteien an: {supported_parties}"
        else:
            message = f"Oops! I did not understand which party your question is referring to. " \
                      f"Please include one of the following: {supported_parties}"
        super().__init__(message)


def extract_party_from(question, language=Language.ENGLISH):
    party = Party.get_from_message(question)
    if not party:
        raise PartyNotFoundError(question, language)
    return party
