#!/usr/bin/env python3
"""Generate OpenASL label files.

Filters the master OpenASL TSV to samples with extracted features, optionally
injects a ``tokenized-text`` pseudo-gloss column from an external JSON mapping,
then splits into train/val/test (90%/5%/5%).

When ``--pseudo_gloss_json`` is provided, the output TSV is guaranteed to
contain a ``tokenized-text`` column holding pseudo-gloss. This column is what
``dataset/openasl.py`` reads at training time — see README Step 0 for the
end-to-end workflow with the external pseudo-gloss repo.
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd


def load_pseudo_gloss_map(path):
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Pseudo-gloss file {path} must be a JSON object "
            f"{{sentence: gloss}}, got {type(raw).__name__}"
        )
    return {k.strip(): (v or "").strip() for k, v in raw.items()}


def inject_pseudo_gloss(df, pg_map, allow_missing):
    """Overwrite (or create) the ``tokenized-text`` column with pseudo-gloss.

    Looks up each row's ``raw-text`` in ``pg_map``. Rows whose sentence is not
    present in the map either abort (strict mode) or fall back to ``raw-text``.
    """
    if "raw-text" not in df.columns:
        raise KeyError(
            "Master OpenASL TSV is missing the 'raw-text' column — cannot "
            "inject pseudo-gloss."
        )

    raw_texts = df["raw-text"].fillna("").astype(str).str.strip()
    tokenized = []
    missing = 0
    for sent in raw_texts:
        gloss = pg_map.get(sent)
        if gloss is None:
            missing += 1
            if not allow_missing:
                raise KeyError(
                    f"No pseudo-gloss found for raw-text: {sent[:100]!r}\n"
                    f"  → Re-run the external pseudo-gloss repo so it covers "
                    f"this sentence, or pass --allow_missing_pseudo_gloss."
                )
            tokenized.append(sent)
        else:
            tokenized.append(gloss)

    df = df.copy()
    df["tokenized-text"] = tokenized
    return df, missing


def main():
    parser = argparse.ArgumentParser(
        description="Generate OpenASL label splits from master TSV + extracted features."
    )
    parser.add_argument(
        "--label_path",
        default="assets/openasl/label/openasl-v1.0.tsv",
        help="Path to original master TSV label file",
    )
    parser.add_argument(
        "--feature_dir",
        default="assets/openasl/clip-vit-large-patch14_openasl",
        help="Directory containing extracted *_s2wrapping.npy feature files",
    )
    parser.add_argument(
        "--output_dir",
        default="assets/openasl/label",
        help="Directory to write split TSV files",
    )
    parser.add_argument(
        "--pseudo_gloss_json",
        type=Path,
        required=True,
        help=(
            "JSON mapping {raw_text: pseudo_gloss_string} produced by the "
            "external pseudo-gloss repo (see README Step 0). The output TSV's "
            "'tokenized-text' column is (over)written with pseudo-gloss from "
            "this file. This flag is required: the OpenASL pipeline trains "
            "against pseudo-gloss, so the split TSVs must carry it."
        ),
    )
    parser.add_argument(
        "--allow_missing_pseudo_gloss",
        action="store_true",
        help=(
            "If set, sentences not found in the pseudo-gloss JSON fall back to "
            "raw-text (with a warning) instead of aborting. Default: strict."
        ),
    )
    args = parser.parse_args()

    print(f"Reading master label file: {args.label_path}")
    df = pd.read_csv(args.label_path, sep="\t", low_memory=False)
    print(f"Total labels: {len(df)}")

    print(f"\nScanning feature directory: {args.feature_dir}")
    feature_files = set()
    for f in os.listdir(args.feature_dir):
        if f.endswith("_s2wrapping.npy"):
            vid = f.replace("_s2wrapping.npy", "")
            feature_files.add(vid)
    print(f"Found {len(feature_files)} feature files")

    df_filtered = df[df["vid"].isin(feature_files)].copy()
    print(f"Filtered samples: {len(df_filtered)}")

    # Inject pseudo-gloss BEFORE splitting so every split gets it.
    # --pseudo_gloss_json is required by argparse, so pg_map is always non-None.
    pg_map = load_pseudo_gloss_map(args.pseudo_gloss_json)
    print(f"\n[pseudo-gloss] loaded {len(pg_map)} entries from {args.pseudo_gloss_json}")
    df_filtered, missing = inject_pseudo_gloss(
        df_filtered, pg_map, args.allow_missing_pseudo_gloss
    )
    matched = len(df_filtered) - missing
    print(f"[pseudo-gloss] matched {matched}/{len(df_filtered)} filtered rows")
    if missing:
        print(
            f"[warn] {missing} sentence(s) had no pseudo-gloss entry and "
            f"fell back to raw-text"
        )

    # Split: train 90%, val 5%, test 5%
    print("\nSplitting dataset...")
    df_shuffled = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)

    n_total = len(df_shuffled)
    n_train = int(n_total * 0.9)
    n_val = int(n_total * 0.05)

    train_df = df_shuffled[:n_train].copy()
    val_df = df_shuffled[n_train : n_train + n_val].copy()
    test_df = df_shuffled[n_train + n_val :].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Preserve the (possibly extended) column order from df_filtered so the
    # tokenized-text column survives the concat when it was newly added.
    col_order = list(df_filtered.columns) + ["split"]
    df_final = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df_final = df_final[col_order]

    print("\nSplit results:")
    print(f"Train: {len(train_df)} ({len(train_df)/len(df_final)*100:.1f}%)")
    print(f"Val:   {len(val_df)} ({len(val_df)/len(df_final)*100:.1f}%)")
    print(f"Test:  {len(test_df)} ({len(test_df)/len(df_final)*100:.1f}%)")
    print(f"Total: {len(df_final)}")

    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(args.output_dir, "openasl-v1.0-filtered.tsv")
    df_final.to_csv(output_path, sep="\t", index=False)
    print(f"\nFiltered label file saved to: {output_path}")

    train_output = os.path.join(args.output_dir, "openasl-v1.0-train.tsv")
    val_output = os.path.join(args.output_dir, "openasl-v1.0-val.tsv")
    test_output = os.path.join(args.output_dir, "openasl-v1.0-test.tsv")

    train_df[col_order].to_csv(train_output, sep="\t", index=False)
    val_df[col_order].to_csv(val_output, sep="\t", index=False)
    test_df[col_order].to_csv(test_output, sep="\t", index=False)

    print("\nSplit files saved:")
    print(f"  Train: {train_output}")
    print(f"  Val:   {val_output}")
    print(f"  Test:  {test_output}")


if __name__ == "__main__":
    main()
