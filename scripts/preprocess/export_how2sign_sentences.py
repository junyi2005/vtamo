#!/usr/bin/env python3
"""Export unique How2Sign sentences to a flat text file.

This is the first step of the pseudo-gloss workflow: it produces the list of
English sentences that the external pseudo-gloss repo
(``pseudo_gloss/``) consumes. The pseudo-gloss step then outputs a
JSON mapping {sentence: pseudo_gloss}, which is fed back into
``convert_how2sign_annotations.py`` via ``--pseudo_gloss_json``.

See README Step 0 for the end-to-end workflow.
"""
import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export unique How2Sign sentences for pseudo-gloss generation."
    )
    parser.add_argument(
        "--source_dir",
        type=Path,
        required=True,
        help="Directory containing how2sign_{train,val,test}.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output text file: one unique sentence per line (UTF-8).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    ordered_sentences = []
    per_split_totals = {}

    for split in ["train", "val", "test"]:
        src = args.source_dir / f"how2sign_{split}.csv"
        if not src.exists():
            print(f"[warn] skip missing split: {src}")
            continue
        count = 0
        with src.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                count += 1
                sent = (row.get("SENTENCE") or "").strip()
                if sent and sent not in seen:
                    seen.add(sent)
                    ordered_sentences.append(sent)
        per_split_totals[split] = count

    with args.output.open("w", encoding="utf-8") as f:
        for sent in ordered_sentences:
            f.write(sent + "\n")

    print(f"[ok] wrote {len(ordered_sentences)} unique sentences to {args.output}")
    for split, total in per_split_totals.items():
        print(f"       {split}: {total} rows scanned")


if __name__ == "__main__":
    main()
