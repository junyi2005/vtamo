#!/usr/bin/env python3
"""ASL Gloss Extraction — vocabulary-based sentence segmentation.

Splits English sentences into ASL glosses by matching words against a
sign language video corpus (gloss.csv). Each extracted gloss is guaranteed
to have a corresponding video in the corpus.

Matching strategy (3 levels, best-first):
  1. Exact match          (confidence 0.95)
  2. Lemma match          (confidence 0.90)  — input lemmatized
  3. Double-lemma match   (confidence 0.85)  — both sides lemmatized

Usage:
    # Single sentence
    python asl_gloss_extract.py "Good morning, how are you today?"

    # From file (one sentence per line)
    python asl_gloss_extract.py --input sentences.txt --output results.json

    # Batch from TSV (like How2Sign / OpenASL format)
    python asl_gloss_extract.py --tsv data.tsv --column sentence --output results.json
"""

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Default path to gloss.csv
# ---------------------------------------------------------------------------
_DEFAULT_GLOSS_CSV = Path(__file__).resolve().parent / "data" / "gloss.csv"

# ---------------------------------------------------------------------------
# Contraction expansion
# ---------------------------------------------------------------------------
_CONTRACTIONS = {
    "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
    "'d": " would", "'m": " am", "'s": " is",
}
_CONTRACTION_RE = re.compile(
    "(" + "|".join(re.escape(k) for k in sorted(_CONTRACTIONS, key=len, reverse=True)) + ")",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Stop words — rarely have sign language equivalents
# ---------------------------------------------------------------------------
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "am", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "shall", "may", "might", "can", "must",
    "i", "me", "my", "mine", "we", "us", "our", "ours",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "theirs",
    "this", "that", "these", "those",
    "and", "but", "or", "nor", "so", "yet", "for",
    "in", "on", "at", "to", "of", "by", "with", "from", "up", "as",
    "into", "about", "between", "through", "after", "before",
    "not", "no", "if", "then", "than", "when", "while",
    "who", "whom", "which", "what", "where", "how",
    "all", "each", "every", "both", "few", "more", "most",
    "some", "any", "such", "only", "own", "same", "too", "very",
    "just", "also", "now", "here", "there",
}


def _normalize_apostrophes(text: str) -> str:
    return text.replace("\u2019", "'").replace("\u2018", "'")


def expand_contractions(text: str) -> str:
    """Expand English contractions: didn't → did not, I've → I have."""
    text = _normalize_apostrophes(text)
    return _CONTRACTION_RE.sub(lambda m: _CONTRACTIONS[m.group(0).lower()], text)


# ---------------------------------------------------------------------------
# GlossVocab — loads gloss.csv and provides multi-level lookup
# ---------------------------------------------------------------------------
class GlossVocab:
    """Sign language vocabulary loaded from gloss.csv."""

    def __init__(self, csv_path: str | Path):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {csv_path}")

        self.word_to_entries: dict[str, list[dict]] = {}
        self.phrases: list[str] = []
        self._lemma_to_words: dict[str, set[str]] = {}
        self._nlp = None

        self._load_csv(csv_path)
        self._build_lemma_index()

    def _load_csv(self, csv_path: Path):
        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = (row.get("word") or "").strip()
                if not word:
                    continue
                word_lower = word.lower()
                entry = {
                    "ref": row.get("ref", ""),
                    "word": word,
                    "sourceid": row.get("sourceid", ""),
                    "synset_id": row.get("synset_id", ""),
                    "gloss": row.get("gloss", ""),
                    "alternate_words": row.get("alternate_words", ""),
                }
                self.word_to_entries.setdefault(word_lower, []).append(entry)

        self.phrases = sorted(
            [w for w in self.word_to_entries if " " in w],
            key=len, reverse=True,
        )
        print(f"[GlossVocab] Loaded {len(self.word_to_entries)} entries, "
              f"{len(self.phrases)} multi-word phrases")

    def _get_nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def _lemmatize(self, word: str) -> str:
        nlp = self._get_nlp()
        doc = nlp(word)
        return doc[0].lemma_.lower() if doc else word.lower()

    def _build_lemma_index(self):
        nlp = self._get_nlp()
        for word_lower in list(self.word_to_entries.keys()):
            if " " in word_lower:
                continue
            doc = nlp(word_lower)
            if doc:
                lemma = doc[0].lemma_.lower()
                if lemma != word_lower:
                    self._lemma_to_words.setdefault(lemma, set()).add(word_lower)

    def tokenize_with_phrases(self, text: str) -> list[str]:
        """Tokenize text, recognizing multi-word phrases from the vocabulary."""
        text_lower = text.lower()
        used = [False] * len(text_lower)
        tokens = []

        for phrase in self.phrases:
            start = 0
            while True:
                idx = text_lower.find(phrase, start)
                if idx == -1:
                    break
                before_ok = (idx == 0 or not text_lower[idx - 1].isalnum())
                end_idx = idx + len(phrase)
                after_ok = (end_idx >= len(text_lower) or not text_lower[end_idx].isalnum())
                if before_ok and after_ok and not any(used[idx:end_idx]):
                    tokens.append(text[idx:end_idx])
                    for i in range(idx, end_idx):
                        used[i] = True
                start = idx + 1

        remaining = []
        current = []
        for i, ch in enumerate(text):
            if used[i]:
                if current:
                    remaining.append("".join(current))
                    current = []
            elif ch.isspace():
                if current:
                    remaining.append("".join(current))
                    current = []
            else:
                current.append(ch)
        if current:
            remaining.append("".join(current))

        tokens.extend(remaining)

        cleaned = []
        for t in tokens:
            t = t.strip(".,!?;:\"'()[]{}—–-")
            if t:
                cleaned.append(t)
        return cleaned

    def lookup(self, word: str) -> dict | None:
        """Multi-level lookup: exact → lemma → double-lemma."""
        word_lower = word.lower().strip()
        if not word_lower:
            return None

        # Level 1: Exact match
        entries = self.word_to_entries.get(word_lower)
        if entries:
            e = entries[0]
            return {
                "ref": e["ref"], "word": word_lower,
                "gloss": e["gloss"], "alternate_words": e["alternate_words"],
                "match_type": "exact", "confidence": 0.95,
                "matched_to": e["word"],
            }

        # Level 2: Lemma match
        input_lemma = self._lemmatize(word_lower)
        if input_lemma != word_lower:
            entries = self.word_to_entries.get(input_lemma)
            if entries:
                e = entries[0]
                return {
                    "ref": e["ref"], "word": word_lower,
                    "gloss": e["gloss"], "alternate_words": e["alternate_words"],
                    "match_type": "lemma", "confidence": 0.90,
                    "matched_to": e["word"],
                }

        # Level 3: Double-lemma match
        vocab_words = self._lemma_to_words.get(input_lemma, set())
        for vw in vocab_words:
            entries = self.word_to_entries.get(vw)
            if entries:
                e = entries[0]
                return {
                    "ref": e["ref"], "word": word_lower,
                    "gloss": e["gloss"], "alternate_words": e["alternate_words"],
                    "match_type": "lemma_lemma", "confidence": 0.85,
                    "matched_to": e["word"],
                }

        return None


# ---------------------------------------------------------------------------
# Extract glosses from sentences
# ---------------------------------------------------------------------------
def extract_glosses(
    sentences: list[str],
    gloss_csv: str | Path | None = None,
) -> dict:
    """Extract glosses from sentences by matching against gloss.csv vocabulary.

    Args:
        sentences: List of English sentences
        gloss_csv: Path to gloss.csv (default: data/gloss.csv)

    Returns:
        dict with keys: glosses, descriptions, vocab, match_details, unmatched
    """
    csv_path = Path(gloss_csv) if gloss_csv else _DEFAULT_GLOSS_CSV
    vocab_db = GlossVocab(csv_path)

    glosses = {}
    descriptions = {}
    vocab_counter = collections.Counter()
    match_details = []
    unmatched_tokens = set()

    for sent in sentences:
        expanded = expand_contractions(sent)
        tokens = vocab_db.tokenize_with_phrases(expanded)

        sent_glosses = []
        for token in tokens:
            token_lower = token.lower().strip()
            if not token_lower or token_lower in STOP_WORDS:
                continue
            if re.fullmatch(r'\d+', token_lower):
                continue

            result = vocab_db.lookup(token)
            if result:
                gloss_word = result["matched_to"].upper()
                sent_glosses.append(gloss_word)

                if gloss_word not in descriptions and result.get("gloss"):
                    descriptions[gloss_word] = result["gloss"]

                match_details.append({
                    "input": token_lower,
                    "matched_to": result["matched_to"],
                    "ref": result["ref"],
                    "match_type": result["match_type"],
                    "confidence": result["confidence"],
                })
            else:
                unmatched_tokens.add(token_lower)

        glosses[sent] = sent_glosses
        vocab_counter.update(sent_glosses)

    vocab = {
        "size": len(vocab_counter),
        "total_tokens": sum(vocab_counter.values()),
        "frequency": dict(vocab_counter.most_common()),
    }

    return {
        "glosses": glosses,
        "descriptions": descriptions,
        "vocab": vocab,
        "match_details": match_details,
        "unmatched": sorted(unmatched_tokens),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract ASL glosses from English sentences using vocabulary matching.",
    )
    parser.add_argument("sentence", nargs="*", help="Sentence(s) to process")
    parser.add_argument("--input", "-i", type=str, help="Input file (one sentence per line)")
    parser.add_argument("--tsv", type=str, help="Input TSV file (How2Sign / OpenASL format)")
    parser.add_argument("--column", type=str, default="sentence", help="TSV column name (default: sentence)")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file (default: stdout)")
    parser.add_argument("--gloss-csv", type=str, default=None, help="Path to gloss.csv")
    args = parser.parse_args()

    # Collect sentences
    sentences = []
    if args.sentence:
        sentences = args.sentence
    elif args.input:
        with open(args.input, encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
    elif args.tsv:
        import pandas as pd
        df = pd.read_csv(args.tsv, sep='\t')
        sentences = [str(s) for s in df[args.column].dropna().unique()]
    else:
        parser.print_help()
        sys.exit(1)

    print(f"Processing {len(sentences)} sentence(s)...\n")

    result = extract_glosses(sentences, gloss_csv=args.gloss_csv)

    # Print summary
    print(f"\n{'='*50}")
    for sent, gloss_list in result["glosses"].items():
        print(f"  {sent}")
        print(f"    → {gloss_list}")
    print(f"{'='*50}")
    print(f"  Vocab size:  {result['vocab']['size']}")
    print(f"  Total tokens: {result['vocab']['total_tokens']}")
    if result["unmatched"]:
        print(f"  Unmatched:   {result['unmatched']}")
    print(f"{'='*50}\n")

    # Save output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
