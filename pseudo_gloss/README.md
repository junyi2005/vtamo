# Pseudo Gloss English

Convert English sentences into pseudo glosses for sign language research.

> **Which one does VTaMo use?** The paper's pseudo-gloss is **`pseudo_gloss_en.py`** —
> a frozen spaCy part-of-speech filter that keeps sign-relevant content tokens
> (`NOUN`, `VERB`, `ADJ`, `ADV`, `NUM`, `PRON`, `PROPN`) and drops the rest
> (`DET`, `ADP`, `AUX`, `PART`, `PUNCT`, `CCONJ`). Nothing else. This is what
> `scripts/preprocess/run_external_pseudo_gloss.py` runs by default (`--mode pos_only`),
> and it is what both the alignment targets and the decoder targets are built from.
>
> The other two entry points below are **ChatSign product extensions, not the paper's
> method**. `combined_pipeline.py` additionally merges multi-word ASL phrases and
> **reorders the tokens into ASL grammar order**, which materially changes the decoder
> targets. Use it only if you specifically want that behaviour (`--mode combined`).

Three approaches are provided:

1. **POS-based filtering** (`pseudo_gloss_en.py`) — the paper's pseudo-gloss
2. **ASL vocabulary matching** (`asl_gloss_seprate/`) — matches against a 25K-entry ASL vocabulary
3. **Combined pipeline** (`combined_pipeline.py`) — ASL phrase merging + POS filter + ASL reordering

## Combined Pipeline (ChatSign extension — not the paper's pseudo-gloss)

The combined pipeline (`combined_pipeline.py`) chains both approaches:

1. **Contraction expansion** (`didn't` → `did not`)
2. **ASL vocabulary phrase merging** — multi-word signs like `how are you`, `good morning` become single tokens (`HOW_ARE_YOU`, `GOOD_MORNING`)
3. **Contextual POS filtering** — remaining single words are POS-tagged in sentence context, keeping only content words (`NOUN`, `VERB`, `ADJ`, `ADV`, `NUM`, `PRON`, `PROPN`)
4. **ASL grammar reordering** — the surviving tokens are reordered towards ASL word order, so the output is *not* in English order

**Example:**

```
Original:     "And I'm a psychologist in Manhattan, New York City, where I have my own private practice."
Pseudo Gloss: PSYCHOLOGIST MANHATTAN PRACTICE I_AM NEW_YORK CITY PRIVATE WHERE
              (16 words → 8 tokens: multi-word "New York" merged, function words
               removed, remainder reordered into ASL order)
```

**Usage:**

```python
from combined_pipeline import process_sentences

results = process_sentences(["Good morning, how are you today?"])
# {"Good morning, how are you today?": ["GOOD_MORNING", "TODAY", "HOW_ARE_YOU"]}
```

> Multi-word signs are emitted with `_` (e.g. `NEW_YORK`) so that the T5 tokenizer
> keeps them atomic. Building the vocabulary lemmatizes all 25K entries and takes
> ~1–2 minutes on first call.

**CLI (runs 20 How2Sign examples by default):**

```bash
python combined_pipeline.py
```

## Environment Setup

### Requirements

- Python >= 3.8
- SpaCy
- pandas
- tqdm

### Installation

```bash
pip install spacy pandas tqdm
python -m spacy download en_core_web_sm
```

## Scripts

### 1. `combined_pipeline.py` — Combined Pipeline (ChatSign extension, not the paper)

Chains ASL vocabulary phrase merging with POS-based filtering. See above for details.

### 2. `pseudo_gloss_en.py` — POS-only (How2Sign)

Processes the How2Sign dataset TSV file. Reads the `sentence` column.

**Before running**, modify the paths at the top of the script:

```python
tsv_path = "data/how2sign/data.tsv"
output_path = "data/how2sign/processed_words.pkl"
```

```bash
python pseudo_gloss_en.py
```

### 3. `pseudo_gloss_en_openasl.py` — POS-only (OpenASL)

Processes the OpenASL dataset TSV file. Reads the `raw-text` column.

```bash
python pseudo_gloss_en_openasl.py
```

### 4. `asl_gloss_seprate/asl_gloss_extract.py` — ASL Vocabulary Matching

Vocabulary-based extraction with phrase awareness. See `asl_gloss_seprate/README.md`.

```bash
python asl_gloss_seprate/asl_gloss_extract.py "Good morning, how are you today?"
```


> `asl_gloss_seprate/data/gloss.csv` is **committed to this repo** — no fetch step is
> needed. It is a 27,080-row ASL vocabulary index (`word` / `gloss` / `alternate_words`);
> only the `word` column is read for phrase merging. The corresponding sign videos are
> not part of this release and are not needed to reproduce the pseudo-gloss.
## Input Format

- **How2Sign**: TSV with `sentence` column (or `SENTENCE` in the raw CSV)
- **OpenASL**: TSV with `raw-text` column

## Output Format

### POS scripts (`pseudo_gloss_en*.py`)

Pickle file (`.pkl`) with:

| Key | Type | Description |
|-----|------|-------------|
| `dict_sentence` | `dict[str, list[str]]` | Original sentence → pseudo gloss word list |
| `dict_lem_to_id` | `dict[str, int]` | Word → integer ID |
| `dict_lem_counter` | `dict[str, int]` | Word frequency counts |

### Combined pipeline (`combined_pipeline.py`)

Returns a `dict[str, list[str]]` mapping sentences to gloss token lists.

### ASL extraction (`asl_gloss_extract.py`)

JSON with `glosses`, `descriptions`, `vocab`, `match_details`, `unmatched`.

## Model

- **SpaCy model**: `en_core_web_sm` (English, ~15 MB)
- No GPU required
