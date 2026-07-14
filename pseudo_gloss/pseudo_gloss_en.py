"""
How2Sign Pseudo Gloss Generator (English)
Extracts keywords from English sentences using SpaCy for pseudo-labeling.
"""
from tqdm import tqdm
import pandas as pd
import spacy
import pickle
from collections import Counter
import os

# English SpaCy model
language_source = "en_core_web_sm"
tsv_path = "data/how2sign/data.tsv"
output_path = "data/how2sign/processed_words.pkl"


def get_parts_of_speech(sentence, nlp):
    """Extract parts of speech from a sentence."""
    doc = nlp(sentence)
    normalized_sentence = [token.lemma_ for token in doc]
    pos_tags = [(token.text, token.pos_) for token in doc]
    return [(n.lower(), p[0], p[1]) for n, p in zip(normalized_sentence, pos_tags)]


# POS tags we want to keep for pseudo glosses
selected_vocab = ["NOUN", "NUM", "ADV", "PRON", "PROPN", "ADJ", "VERB"]

print("Loading SpaCy model...")
nlp = spacy.load(language_source)

print(f"Reading TSV from {tsv_path}...")
df = pd.read_csv(tsv_path, sep='\t')

# Get all unique sentences
sentences = df["sentence"].unique()
print(f"Processing {len(sentences)} unique sentences...")

dict_sentence = {}
all_lems = []

for sentence in tqdm(sentences, desc="Extracting keywords"):
    if pd.isna(sentence):
        continue

    try:
        pos = get_parts_of_speech(str(sentence), nlp)
        lems = []
        for lem, word, part in pos:
            if part in selected_vocab:
                lems.append(word)

        all_lems.extend(lems)
        dict_sentence[sentence] = lems
    except Exception as e:
        print(f"Error processing sentence: {sentence[:50]}... - {e}")
        dict_sentence[sentence] = []

# Create vocabulary mappings
dict_lem_to_id = {lem: i for i, lem in enumerate(sorted(set(all_lems)))}
dict_lem_counter = dict(Counter(all_lems))

print(f"\nVocabulary size: {len(dict_lem_to_id)}")
print(f"Total tokens: {len(all_lems)}")

# Save processed data
dict_processed_words = {
    "dict_lem_counter": dict_lem_counter,
    "dict_sentence": dict_sentence,
    "dict_lem_to_id": dict_lem_to_id,
}

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "wb") as f:
    pickle.dump(dict_processed_words, f)

print(f"\nSaved to: {output_path}")

# Print some statistics
print("\nTop 20 most common words:")
for word, count in Counter(all_lems).most_common(20):
    print(f"  {word}: {count}")
