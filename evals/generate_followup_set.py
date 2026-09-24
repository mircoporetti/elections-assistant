import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_openai import ChatOpenAI

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.json")
FOLLOWUP_SET_PATH = os.path.join(os.path.dirname(__file__), "followup_set.json")

REWRITE_AS_FOLLOWUP_PROMPT = """Rewrite this question as a conversational follow-up, as if the
user had already been talking about the same party in a previous message.

Original question: {question}

Rules:
- Remove the party name entirely. Refer to the party only as "they"/"sie" or by
  dropping the subject, the way someone would in a real conversation.
- Keep the topic identical and keep it in {language}.
- Output ONLY the rewritten question, nothing else."""


def main():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_retries=3)

    with open(GOLDEN_SET_PATH, encoding="utf-8") as source:
        golden_set = json.load(source)

    followups = []
    for item in golden_set:
        prompt = REWRITE_AS_FOLLOWUP_PROMPT.format(
            question=item["question"], language=item["language"]
        )
        followup = llm.invoke(prompt).content.strip().strip('"')

        opening = ("Tell me about the {party} programme."
                   if item["language"] == "English"
                   else "Erzähl mir vom Programm der {party}.")
        followups.append({
            **item,
            "followup_question": followup,
            "history": [
                {"role": "You", "content": opening.format(party=item["expected_party"])},
                {"role": "AI", "content": "Sure, what would you like to know?"},
            ],
        })
        print(f"[{item['expected_party']}] {followup}")

    with open(FOLLOWUP_SET_PATH, "w", encoding="utf-8") as out:
        json.dump(followups, out, ensure_ascii=False, indent=2)
    print(f"\nwrote {len(followups)} items to {FOLLOWUP_SET_PATH}")


if __name__ == "__main__":
    main()