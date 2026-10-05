# Party match quiz: data pipeline

The quiz (`GET /api/quiz`, `POST /api/quiz/score`) serves `src/quiz/quiz_data.json`. That file is
built offline from the manifestos in `resources/manifests/` by the scripts in this folder, then
reviewed by hand. Nothing here runs at request time.

```
generate_statements.py ──► statements.json ──┐
                                             ├─► (human review) ─► position_overrides.json
generate_positions.py  ──► positions.json ───┤
                                             └─► build_quiz.py ──► src/quiz/quiz_data.json
```

Run everything from the repo root. The two `generate_*` scripts call OpenAI and need the key from
`.env`; `build_quiz.py` does not.

## Files

| File | What it is | Regenerable? |
|---|---|---|
| `statements.json` | The 30 statements (DE + EN), one per topic in `TOPICS` | Yes, but **overwrites manual edits** |
| `positions.json` | Raw LLM labels: one position + quote per statement × party, with `flags` | Yes, but costs money and the labels change |
| `position_overrides.json` | Reviewed corrections, each with a `note` saying why | **No, this is the human review** |
| `src/quiz/quiz_data.json` | What the API serves | Yes, `build_quiz.py` rebuilds it |

Commit all four. `quiz_data.json` is not built in the Docker image.

## Steps

### 1. Statements

```bash
env $(cat .env | xargs) poetry run python quiz/generate_statements.py
```

Topics and their (German) retrieval queries are the `TOPICS` list in `generate_statements.py`. For each
topic the script retrieves the top passages per party and asks the LLM for one neutral, divisive
statement.

Review the output before going on:
- Each statement makes one claim, no "and".
- It's current: no deadlines already passed, no prices that already changed.
- The wording is balanced. Make "agree" the left-leaning answer for some statements and the
  right-leaning answer for others.
- The German wording is the authoritative one and must read naturally.

Edit `statements.json` directly. Mark edited entries with `"revised_in_review": true` and clear
`expected_agree`/`expected_disagree`, which are only rough hints. **Rerunning the script overwrites
these edits.**

### 2. Party positions

```bash
env $(cat .env | xargs) TOKENIZERS_PARALLELISM=false poetry run python quiz/generate_positions.py
```

For every statement × party it retrieves 5 passages, labels the position (`agree`, `neutral`,
`disagree`, `not_addressed`) with a quote, then a second LLM call re-checks the label against the
quoted passage. About 420 calls, a few minutes, well under a dollar with `gpt-4o-mini`.

- `MODEL` is set to `gpt-4o-mini` because that's the only model the OpenAI project has access to.
  Use a stronger model if one becomes available.
- `max_workers=3` keeps the run under the 200k tokens/minute limit. More workers hit 429s.

Each position carries `flags`:
- `not_addressed`: no position found. Often retrieval just missed the passage.
- `verifier_disagrees`: the second check gave a different label.
- `quote_not_verbatim`: the quote isn't in the passage word for word.

### 3. Review

Go through every flagged position, and spot-check the unflagged ones. Searching the PDF text by keyword
finds what retrieval misses:

```bash
poetry run python -c "
import re; from pypdf import PdfReader
for i, p in enumerate(PdfReader('resources/manifests/FDP.pdf').pages):
    t = re.sub(r'\s+', ' ', p.extract_text() or '')
    for m in re.finditer(r'conscription', t, re.I):
        print(f'p.{i+1}:', t[max(0, m.start()-200):m.end()+200])
"
```

Mistakes we found the first time, worth checking for again:
- Clear positions labelled `not_addressed` because retrieval missed them.
- The right position backed by a quote on a different topic.
- Inverted direction on negatively worded statements ("take *longer* than 2045").
- "At least 2%" labelled as agreeing with "well over 2%".

Put every correction in `position_overrides.json`, which wins over `positions.json`:

```json
{"statement_id": "conscription", "party": "FDP", "position": "disagree", "source": "FDP p.44",
 "quote": "We reject the reinstatement of general conscription.", "note": "Labeller missed this passage."}
```

- `source` is `<PARTY> p.<1-based page>`.
- `quote` must be copied exactly from that page. It may run onto the next page.
- For `not_addressed`, use `"source": null, "quote": ""`.
- Add `"needs_human_review": true` while a decision is still open.

### 4. Build and check

```bash
poetry run python quiz/build_quiz.py
poetry run pytest tests/quiz
```

`build_quiz.py` merges the three files and checks every quote against the cited PDF page:
- Generated quotes that are close but not exact (similarity ≥ 0.75) are replaced with the matching
  sentences from the page. The script prints each replacement.
- The build fails on any reviewed quote that isn't on its page, and on any generated quote with no
  close match. Fix those with an override.

Read the replacements it prints. If a page footer was spliced into a sentence, it shows up there.

Before shipping, check that each party ranks first when you answer exactly like it, and look at
which statements split the parties weakly (fewer than two parties on each side).

## Adding another election (e.g. Italy)

The pipeline is general. These parts are currently specific to Germany 2025:

- `generate_statements.py`: `TOPICS` (with German queries), `PARTY_LABELS`, and the prompt's
  mention of the German federal election.
- `build_quiz.py`: `PARTIES`, the hard-coded file paths and `MANIFESTS_PATH`.
- `src/chat/party.py`: the `Party` enum and aliases, which retrieval uses for per-party filtering.
- `src/quiz/scoring.py`: the single `QUIZ_PATH`.
- The frontend (`elections` repo): `components/parties.ts` (colours, names) and the `en`/`de` locales.

Give each election its own data folder and quiz file instead of overwriting these, then run steps 1–4
for it. Write the topics for the new country from scratch: the most debated issues differ.
