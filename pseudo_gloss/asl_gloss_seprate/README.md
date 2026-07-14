# ASL Gloss Extraction

Vocabulary-based English → ASL gloss segmentation. Matches words against a 25K-entry sign language video corpus (`gloss.csv`), ensuring every extracted gloss has a corresponding video.

## How It Works

1. Expand contractions (`didn't` → `did not`)
2. Tokenize with phrase awareness (`good morning`, `how are you` → single tokens)
3. Filter stop words
4. Multi-level vocabulary lookup:
   - **Exact match** (confidence 0.95) — word found directly in corpus
   - **Lemma match** (confidence 0.90) — `sitting` → `sit`
   - **Double-lemma match** (confidence 0.85) — both input and vocab lemmatized

## Quick Start

```bash
pip install spacy
python -m spacy download en_core_web_sm

# Single sentence
python asl_gloss_extract.py "Good morning, how are you today?"

# From file
python asl_gloss_extract.py --input sentences.txt --output results.json

# From TSV (How2Sign / OpenASL format)
python asl_gloss_extract.py --tsv data.tsv --column sentence --output results.json
```

## Example

```
Input:  "Good morning, how are you today?"
Output: ["GOOD MORNING", "HOW ARE YOU", "TODAY"]
```

Compared to the original spaCy POS-filter approach (`pseudo_gloss_en.py`):
- Original: `["GOOD", "MORNING", "YOU", "TODAY"]` (4 separate words)
- This tool: `["GOOD MORNING", "HOW ARE YOU", "TODAY"]` (phrase-aware, corpus-matched)

## Files

```
asl_gloss_seprate/
├── asl_gloss_extract.py   # Main script
├── data/
│   └── gloss.csv          # ASL vocabulary, 27,080 rows (committed to git)
└── README.md
```

### `gloss.csv` — 27K ASL vocabulary

`data/gloss.csv` is **committed to this repo** — no fetch step is needed.
It is the vocabulary index used for phrase-aware matching; the sign videos it
was derived from are not part of this release and are not required, since only
the `word` column drives tokenization and lookup.

| Column | Description |
|--------|-------------|
| word | Vocabulary entry (the only column used for matching) |
| gloss | Gloss label, when the entry has one |
| alternate_words | Comma-separated synonyms |
