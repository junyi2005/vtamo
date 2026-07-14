#!/usr/bin/env python3
"""Convert How2Sign CSV annotations to ``*_info_ml.npy`` format.

If a pseudo-gloss JSON file is provided via ``--pseudo_gloss_json``, the
``text`` field of each entry will be replaced by the pseudo-gloss string, and
the original sentence will be preserved under ``original_text``. That JSON file
is produced by the bundled pseudo-gloss step (``pseudo_gloss/``);
see README Step 0 for the end-to-end workflow.

Without ``--pseudo_gloss_json`` the script falls back to writing the raw
English ``SENTENCE`` column, which matches the legacy behaviour.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert How2Sign CSV annotations to *_info_ml.npy format."
    )
    parser.add_argument(
        "--source_dir",
        type=Path,
        required=True,
        help="Directory containing how2sign_{train,val,test}.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write *_info_ml.npy files.",
    )
    parser.add_argument(
        "--pseudo_gloss_json",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping {original_sentence: pseudo_gloss_string}. "
            "When provided, the 'text' field is set to pseudo-gloss and "
            "'original_text' keeps the raw English sentence. This file must be "
            "produced by the external pseudo-gloss repo — see README Step 0."
        ),
    )
    parser.add_argument(
        "--allow_missing_pseudo_gloss",
        action="store_true",
        help=(
            "If set, sentences not found in the pseudo-gloss JSON fall back to "
            "the raw English sentence (with a warning) instead of aborting. "
            "Default: abort on any miss (strict)."
        ),
    )
    return parser.parse_args()


def load_pseudo_gloss_map(path: Path) -> dict:
    """Load a JSON mapping of ``{original_sentence: pseudo_gloss_string}``.

    Keys are whitespace-stripped so that minor formatting differences between
    the source handed to the pseudo-gloss repo and the CSV ``SENTENCE`` column do
    not cause spurious misses.
    """
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Pseudo-gloss file {path} must be a JSON object "
            f"{{sentence: gloss}}, got {type(raw).__name__}"
        )
    out = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(
                f"Pseudo-gloss file {path} contains a non-string entry "
                f"({type(k).__name__} → {type(v).__name__})"
            )
        out[k.strip()] = v.strip()
    return out


def load_split(split_path: Path, pg_map: dict, allow_missing: bool):
    records = []
    missing_samples = []
    total = 0
    with split_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total += 1
            fileid = row["SENTENCE_NAME"].strip()
            # SENTENCE is the raw English string from the How2Sign CSV
            original_sentence = (row.get("SENTENCE") or "").strip()

            if pg_map:
                pseudo_gloss = pg_map.get(original_sentence)
                if pseudo_gloss is None:
                    missing_samples.append((fileid, original_sentence))
                    if allow_missing:
                        text = original_sentence
                    else:
                        raise KeyError(
                            f"No pseudo-gloss found for sentence in "
                            f"{split_path.name} (fileid={fileid}): "
                            f"{original_sentence[:100]!r}\n"
                            f"  → Re-run the external pseudo-gloss repo so it "
                            f"covers this sentence, or pass "
                            f"--allow_missing_pseudo_gloss to fall back to the "
                            f"raw English text for missing entries."
                        )
                else:
                    text = pseudo_gloss
            else:
                text = original_sentence

            record = {
                "fileid": fileid,
                # Use sentence-level name for folder so it matches trimmed mp4 filenames
                "folder": fileid,
                "text": text,
                "original_text": original_sentence,
                "gloss": "",
                "start": float(row["START"]) if row["START"] else None,
                "end": float(row["END"]) if row["END"] else None,
                "video_name": row["VIDEO_NAME"].strip(),
            }
            records.append(record)
    return records, missing_samples, total


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pg_map = load_pseudo_gloss_map(args.pseudo_gloss_json)
    if args.pseudo_gloss_json:
        print(f"[pseudo-gloss] loaded {len(pg_map)} entries from {args.pseudo_gloss_json}")
    else:
        print(
            "[pseudo-gloss] no mapping provided — 'text' field will contain "
            "raw English sentences (pseudo-gloss disabled)"
        )

    grand_missing = 0
    for split in ["train", "val", "test"]:
        src = args.source_dir / f"how2sign_{split}.csv"
        if not src.exists():
            print(f"[warn] skip missing split: {src}")
            continue
        records, missing_samples, total = load_split(
            src, pg_map, args.allow_missing_pseudo_gloss
        )
        out_path = args.output_dir / f"{split}_info_ml.npy"
        np.save(out_path, np.array(records, dtype=object))
        msg = f"[ok] wrote {out_path} ({total} samples)"
        if pg_map:
            matched = total - len(missing_samples)
            msg += f" — pseudo-gloss matched: {matched}/{total}"
            if missing_samples:
                msg += f" (missing: {len(missing_samples)}, fell back to raw sentence)"
                grand_missing += len(missing_samples)
        print(msg)

    if pg_map and grand_missing and args.allow_missing_pseudo_gloss:
        print(
            f"\n[warn] {grand_missing} sentence(s) across all splits had no "
            f"pseudo-gloss entry and used the raw English sentence as a "
            f"fallback. Re-run the external pseudo-gloss repo to cover them."
        )


if __name__ == "__main__":
    main()
