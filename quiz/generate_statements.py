import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from chat.party import Party
from store import reranker, vector_store

STATEMENTS_PATH = os.path.join(os.path.dirname(__file__), "statements.json")
PASSAGES_PER_PARTY = 3
PASSAGE_CHARS = 1500

# (id, area, retrieval query). Queries are German because the manifestos are.
TOPICS = [
    ("border-rejections", "Migration", "Zurückweisung von Asylsuchenden an den deutschen Grenzen"),
    ("family-reunification", "Migration", "Familiennachzug für subsidiär Schutzberechtigte"),
    ("citizenship", "Migration", "Einbürgerung, doppelte Staatsbürgerschaft, Staatsangehörigkeitsrecht"),
    ("debt-brake", "Economy & Finance", "Schuldenbremse im Grundgesetz"),
    ("wealth-tax", "Economy & Finance", "Vermögensteuer auf große Vermögen"),
    ("solidarity-surcharge", "Economy & Finance", "Solidaritätszuschlag abschaffen"),
    ("corporate-tax", "Economy & Finance", "Unternehmenssteuern senken"),
    ("minimum-wage", "Labour & Social", "gesetzlicher Mindestlohn 15 Euro"),
    ("citizens-income", "Labour & Social", "Bürgergeld, Sanktionen, Grundsicherung"),
    ("retirement-age", "Labour & Social", "Renteneintrittsalter, Rente mit 67, längeres Arbeiten"),
    ("health-insurance", "Labour & Social", "Bürgerversicherung statt privater Krankenversicherung"),
    ("rent-cap", "Housing", "Mietpreisbremse, Mietendeckel, Mieten begrenzen"),
    ("ukraine-weapons", "Foreign & Defence", "Waffenlieferungen an die Ukraine"),
    ("russia-sanctions", "Foreign & Defence", "Sanktionen gegen Russland, russisches Gas"),
    ("conscription", "Foreign & Defence", "Wehrpflicht, Wehrdienst"),
    ("defence-spending", "Foreign & Defence", "Verteidigungsausgaben, NATO-Ziel, Bundeswehr Ausrüstung"),
    ("eu-integration", "Europe", "Europäische Union vertiefen, Mehrheitsentscheidungen statt Einstimmigkeit"),
    ("nuclear-power", "Energy & Climate", "Kernkraft, Atomkraftwerke wieder in Betrieb nehmen"),
    ("heating-law", "Energy & Climate", "Gebäudeenergiegesetz, Heizungsgesetz, Wärmepumpe"),
    ("combustion-ban", "Energy & Climate", "Verbrenner-Aus 2035, Verbrennungsmotor"),
    ("climate-target", "Energy & Climate", "Klimaneutralität 2045, Klimaschutzziele"),
    ("co2-price", "Energy & Climate", "CO2-Preis, Emissionshandel, Klimageld"),
    ("speed-limit", "Transport", "Tempolimit auf Autobahnen"),
    ("deutschlandticket", "Transport", "Deutschlandticket, öffentlicher Nahverkehr"),
    ("cannabis", "Society", "Cannabis-Legalisierung"),
    ("abortion", "Society", "Schwangerschaftsabbruch, Paragraph 218"),
    ("self-determination", "Society", "Selbstbestimmungsgesetz, Geschlechtseintrag"),
    ("voting-age", "Democracy", "Wahlalter 16 bei Bundestagswahlen"),
    ("referendums", "Democracy", "bundesweite Volksentscheide, direkte Demokratie"),
    ("data-retention", "Security & Digital", "Vorratsdatenspeicherung, IP-Adressen speichern"),
]

PARTY_LABELS = {
    "SPD": "SPD", "CDU": "CDU/CSU", "AFD": "AfD", "FDP": "FDP",
    "DL": "Die Linke", "DG": "Die Grünen", "BSW": "BSW",
}

PROMPT = """You are writing statements for a voting advice application (like the German Wahl-O-Mat)
for the German federal election. Users answer each statement with agree / neutral / disagree,
and are then matched against the parties' manifestos.

Topic: {topic}

Below are the most relevant manifesto passages for this topic, per party.

{passages}

Write ONE statement about this topic.

Rules:
- It must be a single, concrete policy claim a citizen can agree or disagree with
  (e.g. "Germany should reintroduce compulsory military service."). One idea only, no "and".
- It must DIVIDE the parties according to the passages above: at least two parties should be on
  each side. If the passages show broad agreement on the obvious question, pick the aspect of
  the topic where they actually disagree.
- Neutral wording: no loaded or party-specific terms, no slogans, no double negatives.
- Understandable for an ordinary voter, at most 20 words.
- Provide the German and English versions with the same meaning. The German version is
  authoritative and must sound natural.
- In expected_agree / expected_disagree list party codes ({codes}) based ONLY on the passages.
  Leave a party out of both lists if the passages do not make its position clear."""


class Statement(BaseModel):
    statement_de: str = Field(description="The statement in German")
    statement_en: str = Field(description="The same statement in English")
    expected_agree: list[str] = Field(description="Party codes that the passages suggest agree")
    expected_disagree: list[str] = Field(description="Party codes that the passages suggest disagree")
    rationale: str = Field(description="One sentence: why this statement divides the parties")


def passages_for(query):
    blocks = []
    for party in Party:
        docs = vector_store.most_relevant_for(party, query, k=PASSAGES_PER_PARTY)
        texts = "\n---\n".join(doc.page_content[:PASSAGE_CHARS] for doc in docs)
        blocks.append(f"### {party.name} ({PARTY_LABELS[party.name]})\n{texts}")
    return "\n\n".join(blocks)


def main():
    vector_store.init()
    reranker.load()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_retries=3).with_structured_output(Statement)
    codes = ", ".join(party.name for party in Party)

    statements = []
    for topic_id, area, query in TOPICS:
        result = llm.invoke(PROMPT.format(topic=query, passages=passages_for(query), codes=codes))
        statements.append({"id": topic_id, "area": area, "query": query, **result.model_dump()})
        print(f"[{topic_id}] {result.statement_de}\n    {result.statement_en}\n"
              f"    agree={result.expected_agree} disagree={result.expected_disagree}")

    with open(STATEMENTS_PATH, "w", encoding="utf-8") as out:
        json.dump(statements, out, ensure_ascii=False, indent=2)
    print(f"\nwrote {len(statements)} statements to {STATEMENTS_PATH}")


if __name__ == "__main__":
    main()
