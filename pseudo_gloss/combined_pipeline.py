#!/usr/bin/env python3
"""Combined Pipeline: ASL Vocabulary Phrase Merging + Pseudo-Gloss Filtering
+ ASL Grammar Reordering.

Full parity with chatsign-auto's
`backend/core/sign_video_generator.py::_extract_sentence_glosses_asl`:

  1. Expand contractions.
  2. Phrase-aware tokenization using ASL vocabulary (multi-word → single token).
  3. Per token: drop ``token_lower in STOP_WORDS and token_lower not in ASL_KEEP``.
     ``ASL_KEEP`` rescues WH-words and negation from the English stopword list
     (they are real ASL signs).
  4. Per surviving token: try `vocab.lookup` (three-tier exact / lemma /
     double-lemma). If it matches, keep the canonical ASL gloss verbatim.
     Unmatched tokens fall through to a full-sentence SpaCy POS filter
     (keep NOUN/NUM/ADV/PRON/PROPN/ADJ/VERB).
  5. Reorder the emitted gloss list into ASL grammar order:
     TIME + TOPIC + SUBJECT + VERB + OTHER + NEG + WH.

This bridges `asl_gloss_extract.py` (phrase merging + vocab lookup) with
pseudo-gloss POS filtering and the ASL grammar reorder.
"""

import sys
import re
from pathlib import Path

# Add parent path so we can import from asl_gloss_seprate
sys.path.insert(0, str(Path(__file__).resolve().parent / "asl_gloss_seprate"))

from asl_gloss_extract import GlossVocab, expand_contractions, STOP_WORDS, _DEFAULT_GLOSS_CSV
import spacy


# POS tags to keep (same as pseudo_gloss_en.py)
SELECTED_POS = {"NOUN", "NUM", "ADV", "PRON", "PROPN", "ADJ", "VERB"}

# Words English treats as stopwords but ASL retains (WH-words + negation).
# Mirrors chatsign-auto/backend/core/sign_video_generator.py::_ASL_KEEP.
ASL_KEEP = {
    "what", "who", "where", "when", "why", "how", "which",  # WH-words
    "not", "no",                                            # negation
}

# Constants used by the ASL grammar reorder step — verbatim copy from
# chatsign-auto/backend/core/sign_video_generator.py (same module, same names).
_WH_WORDS = {"WHAT", "WHO", "WHERE", "WHEN", "WHY", "HOW", "WHICH"}
_NEG_WORDS = {"NOT", "NEVER", "NOTHING", "NOBODY", "NEITHER", "NONE", "NO"}
_TIME_WORDS = {
    "YESTERDAY", "TODAY", "TOMORROW", "NOW", "LATER", "BEFORE", "AFTER",
    "MORNING", "NIGHT", "EVENING", "AFTERNOON", "ALWAYS", "NEVER",
    "ALREADY", "RECENTLY", "SOON", "OFTEN", "SOMETIMES", "USUALLY",
    "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY",
    "WEEK", "MONTH", "YEAR", "AGO", "LAST", "NEXT", "EVERY",
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
    "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
}
_TIME_DEPS = {"npadvmod", "advmod"}


def reorder_glosses_asl(glosses: list[str], sentence: str, nlp=None) -> list[str]:
    """Public wrapper: reorder an English-ordered gloss list into ASL grammar order."""
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    return _reorder_sentence_asl(glosses, sentence, nlp)


def _reorder_sentence_asl(glosses: list[str], sentence: str, nlp) -> list[str]:
    """Reorder glosses from English word order to ASL grammar order.

    ASL rules applied:
      1. Time expressions first
      2. Topic (object) before subject-verb when applicable
      3. Negation after verb
      4. WH-words at end
    """
    if len(glosses) <= 1:
        return glosses

    doc = nlp(sentence)

    # Build a lookup: (dep, pos) per gloss — first unused SpaCy token match
    token_lookup: list[tuple[str | None, str | None]] = []
    used: set[int] = set()
    for gloss in glosses:
        matched_tok = None
        for tok in doc:
            if tok.i in used:
                continue
            if tok.lemma_.upper() == gloss or tok.text.upper() == gloss:
                matched_tok = tok
                break
            # Multi-word glosses like "I_AM" match on the first component
            if "_" in gloss and tok.lemma_.upper() == gloss.split("_")[0]:
                matched_tok = tok
                break
        if matched_tok is not None:
            used.add(matched_tok.i)
            token_lookup.append((matched_tok.dep_, matched_tok.pos_))
        else:
            token_lookup.append((None, None))

    def _is_time(gloss: str) -> bool:
        parts = gloss.replace("_", " ").split()
        return gloss in _TIME_WORDS or any(p in _TIME_WORDS for p in parts)

    def _is_wh(gloss: str) -> bool:
        parts = gloss.replace("_", " ").split()
        return gloss in _WH_WORDS or any(p in _WH_WORDS for p in parts)

    time_group: list[str] = []
    topic_group: list[str] = []
    subject_group: list[str] = []
    verb_group: list[str] = []
    other_group: list[str] = []
    neg_group: list[str] = []
    wh_group: list[str] = []

    for i, gloss in enumerate(glosses):
        dep, pos = token_lookup[i]

        if _is_wh(gloss):
            wh_group.append(gloss)
        elif gloss in _NEG_WORDS:
            neg_group.append(gloss)
        elif _is_time(gloss) or dep in _TIME_DEPS:
            time_group.append(gloss)
        elif dep in ("dobj", "attr", "pobj", "oprd"):
            topic_group.append(gloss)
        elif dep in ("nsubj", "nsubjpass"):
            subject_group.append(gloss)
        elif dep == "ROOT" or pos == "VERB":
            verb_group.append(gloss)
        else:
            other_group.append(gloss)

    return (
        time_group
        + topic_group
        + subject_group
        + verb_group
        + other_group
        + neg_group
        + wh_group
    )


def combined_gloss_pipeline(sentence: str, vocab: GlossVocab, nlp) -> list[str]:
    """Process a single sentence through the combined pipeline.

    Strategy:
      - Drop `STOP_WORDS` tokens early (keep `ASL_KEEP` exceptions).
      - Prefer the ASL vocabulary's three-tier matcher for both single words
        and multi-word phrases — canonical gloss form wins.
      - Unmatched single words fall through to a SpaCy POS filter run over
        the full reconstructed sentence (context matters; a bare word often
        gets the wrong POS tag in isolation).

    Returns:
        List of gloss tokens in original (English) word order.
    """
    # Step 1: Expand contractions
    expanded = expand_contractions(sentence)

    # Step 2: Phrase-aware tokenization (multi-word phrases become single tokens)
    tokens = vocab.tokenize_with_phrases(expanded)

    # Step 3 & 4: Per-token stopword drop + vocab lookup / POS fallback.
    # Each entry: (type, value)
    #   type="vocab" → already a canonical ASL gloss
    #   type="word"  → deferred to POS filter; value = index into single_words
    token_plan: list[tuple[str, object]] = []
    single_words: list[str] = []

    for token in tokens:
        token_clean = token.strip(".,!?;:\"'()[]{}—–-")
        if not token_clean:
            continue
        token_lower = token_clean.lower()
        if not token_lower:
            continue
        # Stopword drop with ASL_KEEP rescue
        if token_lower in STOP_WORDS and token_lower not in ASL_KEEP:
            continue
        # Drop bare digit runs (years like "2025" are typically not signed)
        if re.fullmatch(r"\d+", token_lower):
            continue

        # ASL_KEEP words go straight to POS (ensures the lemma upper, e.g. "NOT")
        if token_lower in ASL_KEEP:
            idx = len(single_words)
            single_words.append(token_clean)
            token_plan.append(("word", idx))
            continue

        # Vocab-first: canonical ASL gloss if the token is in the dictionary.
        # Multi-word phrases ("at least", "abu dhabi") are emitted with '_' so
        # the inner space is not later mistaken for a token boundary by the
        # T5 tokenizer in phase 8 training. Phase 2 _safe_gloss already
        # collapses both space and underscore to '_' for word_lib lookup.
        result = vocab.lookup(token_clean)
        if result:
            token_plan.append(("vocab", result["matched_to"].upper().replace(" ", "_")))
        else:
            idx = len(single_words)
            single_words.append(token_clean)
            token_plan.append(("word", idx))

    # Step 4b: Run SpaCy on all single words joined as a sentence (preserves context)
    pos_results: dict[int, tuple[str, str]] = {}
    if single_words:
        doc = nlp(" ".join(single_words))
        for i, tok in enumerate(doc):
            if i < len(single_words):
                pos_results[i] = (tok.lemma_.upper(), tok.pos_)

    # Step 5: Assemble final gloss list
    sent_glosses: list[str] = []
    for entry_type, value in token_plan:
        if entry_type == "vocab":
            sent_glosses.append(value)  # type: ignore[arg-type]
        else:
            idx = value  # type: ignore[assignment]
            if idx in pos_results:
                lemma, pos = pos_results[idx]
                if pos in SELECTED_POS or lemma.lower() in ASL_KEEP:
                    sent_glosses.append(lemma)

    # Step 6: Reorder into ASL grammar order
    return _reorder_sentence_asl(sent_glosses, sentence, nlp)


def process_sentences(sentences: list[str], gloss_csv=None):
    """Process a list of sentences through the combined pipeline."""
    csv_path = Path(gloss_csv) if gloss_csv else _DEFAULT_GLOSS_CSV
    vocab = GlossVocab(csv_path)
    nlp = spacy.load("en_core_web_sm")

    results = {}
    for sent in sentences:
        glosses = combined_gloss_pipeline(sent, vocab, nlp)
        results[sent] = glosses

    return results


def main():
    import csv as csv_mod

    # Load 20 example sentences from How2Sign training data
    train_csv = "./assets/how2sign/how2sign_train.csv"

    print("Loading training sentences...")
    sentences = []
    with open(train_csv, encoding="utf-8") as f:
        reader = csv_mod.DictReader(f, delimiter='\t')
        for row in reader:
            sent = row.get("SENTENCE", "").strip()
            if sent and len(sent) > 10:  # Skip very short ones like "Hi."
                sentences.append(sent)
                if len(sentences) >= 20:
                    break

    print(f"Selected {len(sentences)} sentences.\n")

    # Process
    results = process_sentences(sentences)

    # Display results
    print("=" * 80)
    print("COMBINED PIPELINE RESULTS: ASL Vocab Merge → Pseudo-Gloss POS Filter")
    print("=" * 80)
    for i, (sent, glosses) in enumerate(results.items(), 1):
        print(f"\n[{i}] Original:    {sent}")
        print(f"    Pseudo-Gloss: {' '.join(glosses)}")
        print(f"    Token count:  {len(sent.split())} → {len(glosses)}")

    # Summary stats
    total_orig = sum(len(s.split()) for s in results)
    total_gloss = sum(len(g) for g in results.values())
    print(f"\n{'=' * 80}")
    print(f"Summary: {total_orig} original words → {total_gloss} gloss tokens "
          f"(compression: {total_gloss/total_orig:.1%})")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
