#!/usr/bin/env python3
"""Export unique OpenASL sentences to a flat text file.

First step of the OpenASL pseudo-gloss workflow: produces the list of English
sentences that the bundled pseudo-gloss step (``pseudo_gloss/``)
consumes. The pseudo-gloss step then outputs a JSON mapping
{sentence: pseudo_gloss}, which is fed back into
``generate_openasl_labels.py`` via ``--pseudo_gloss_json``.

See README Step 0 for the end-to-end workflow.
"""
import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export unique OpenASL sentences for pseudo-gloss generation."
    )
    parser.add_argument(
        "--label_path",
        type=Path,
        required=True,
        help="Path to the master OpenASL TSV (openasl-v1.0.tsv)",
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

    df = pd.read_csv(args.label_path, sep="\t", low_memory=False)
    if "raw-text" not in df.columns:
        raise KeyError(
            f"{args.label_path} is missing the 'raw-text' column — cannot "
            f"export sentences."
        )

    seen = set()
    ordered = []
    for sent in df["raw-text"].fillna("").astype(str):
        sent = sent.strip()
        if sent and sent not in seen:
            seen.add(sent)
            ordered.append(sent)

    with args.output.open("w", encoding="utf-8") as f:
        for sent in ordered:
            f.write(sent + "\n")

    print(f"[ok] wrote {len(ordered)} unique sentences to {args.output}")
    print(f"       scanned {len(df)} rows from {args.label_path}")


if __name__ == "__main__":
    main()
