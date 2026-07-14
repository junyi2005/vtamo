#!/usr/bin/env python3
"""Bridge from the bundled ``pseudo_gloss/`` module to this repo's pseudo-gloss JSON.

That module exposes three entry points, none of which emit the flat
``{sentence: "space-joined gloss"}`` JSON that this repo consumes:

  * ``combined_pipeline.py``     — ChatSign extension (phrase merge + ASL reorder),
    NOT the paper's pseudo-gloss. Library function
    ``process_sentences(list[str]) → dict[str, list[str]]``. Its CLI only
    prints demo examples, so we call the function directly.
  * ``pseudo_gloss_en.py``       — POS-only; this is the paper's pseudo-gloss and the
    default. Script-style; imports the helper
    ``get_parts_of_speech`` from inside the module.
  * ``asl_gloss_seprate/asl_gloss_extract.py`` — vocabulary matching only,
    different JSON schema. Not used here.

This bridge runs the ``pos_only`` (default, the paper's pseudo-gloss) or
``combined`` pipeline on a list
of sentences read from ``--input``, then writes the flat JSON at ``--output``.
The output is the single file contract consumed by
``convert_how2sign_annotations.py`` / ``generate_openasl_labels.py`` — see
README Step 0.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name!r} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_combined(repo_root: Path, sentences: list[str]) -> dict[str, list[str]]:
    """Invoke combined_pipeline.process_sentences from the bundled module."""
    combined_path = repo_root / "combined_pipeline.py"
    asl_dir = repo_root / "asl_gloss_seprate"
    if not combined_path.exists():
        raise FileNotFoundError(
            f"combined_pipeline.py not found under {repo_root}. "
            f"Is --pseudo_gloss_repo pointing at the bundled pseudo_gloss/ directory?"
        )
    if not asl_dir.exists():
        raise FileNotFoundError(
            f"asl_gloss_seprate/ not found under {repo_root}; combined_pipeline "
            f"depends on it."
        )

    # combined_pipeline.py does `sys.path.insert(0, .../asl_gloss_seprate)` at
    # import time. Do the same here so the relative import lands.
    for p in (str(repo_root), str(asl_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    mod = _load_module("_ext_combined_pipeline", combined_path)
    if not hasattr(mod, "process_sentences"):
        raise AttributeError(
            "combined_pipeline.py has no 'process_sentences' function — the "
            "bundled pseudo_gloss API has changed; update this bridge."
        )
    return mod.process_sentences(sentences)


def run_pos_only(repo_root: Path, sentences: list[str]) -> dict[str, list[str]]:
    """POS-only filter: load get_parts_of_speech from pseudo_gloss_en.py."""
    pg_path = repo_root / "pseudo_gloss_en.py"
    if not pg_path.exists():
        raise FileNotFoundError(
            f"pseudo_gloss_en.py not found under {repo_root}."
        )

    # pseudo_gloss_en.py is a script — importing it executes top-level code
    # (reading a hardcoded TSV, writing a pkl). We only need its
    # `get_parts_of_speech` function and the POS filter list, so we reuse its
    # definitions without running its main block by reading the source and
    # exec-ing the safe prefix up to the "Loading SpaCy model..." line.
    src = pg_path.read_text(encoding="utf-8")
    # Cut at the first top-level call to spacy.load — everything before that
    # (helper function, SELECTED_POS list) is pure definitions we can reuse.
    cut = src.find("print(\"Loading SpaCy model")
    if cut == -1:
        cut = src.find("nlp = spacy.load")
    if cut == -1:
        raise RuntimeError(
            "Cannot locate a safe prefix cut in pseudo_gloss_en.py — the "
            "bundled pseudo_gloss layout has changed; update this bridge."
        )
    ns: dict = {"__name__": "_ext_pseudo_gloss_en"}
    exec(compile(src[:cut], str(pg_path), "exec"), ns)

    get_pos = ns["get_parts_of_speech"]
    selected_vocab = ns["selected_vocab"]

    import spacy
    nlp = spacy.load("en_core_web_sm")

    out: dict[str, list[str]] = {}
    for sent in sentences:
        try:
            triples = get_pos(sent, nlp)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to process: {sent[:80]!r} ({exc})")
            out[sent] = []
            continue
        out[sent] = [word for _lem, word, part in triples if part in selected_vocab]
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the bundled pseudo_gloss repo on a text file "
            "of sentences and emit the flat JSON this repo consumes "
            "(see README Step 0)."
        )
    )
    parser.add_argument(
        "--pseudo_gloss_repo",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "pseudo_gloss",
        help="Path to the pseudo-gloss module (default: the bundled ./pseudo_gloss).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Input text file, one sentence per line — produced by "
            "export_how2sign_sentences.py or export_openasl_sentences.py."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Output JSON file: {original_sentence: pseudo_gloss_string}. "
            "Pass this to convert_how2sign_annotations.py / "
            "generate_openasl_labels.py via --pseudo_gloss_json."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["pos_only", "combined"],
        default="pos_only",
        help=(
            "Which pseudo-gloss to build. 'pos_only' (default) is the paper's "
            "pseudo-gloss: a frozen spaCy POS filter keeping NOUN/VERB/ADJ/ADV/"
            "NUM/PRON/PROPN, via pseudo_gloss_en.py. 'combined' additionally "
            "applies ASL phrase merging and ASL grammar reordering "
            "(combined_pipeline.process_sentences) — that is a ChatSign product "
            "extension, NOT the pseudo-gloss described in the paper; it changes "
            "the decoder targets."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.pseudo_gloss_repo.exists():
        raise FileNotFoundError(f"--pseudo_gloss_repo {args.pseudo_gloss_repo} does not exist")
    if not args.input.exists():
        raise FileNotFoundError(f"--input {args.input} does not exist")

    with args.input.open("r", encoding="utf-8") as f:
        sentences = [line.rstrip("\n") for line in f]
    # Dedupe while preserving order and dropping empty lines
    seen = set()
    unique = []
    for s in sentences:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            unique.append(s)

    print(f"[in]  {len(unique)} unique sentences from {args.input}")
    print(f"[ext] repo:  {args.pseudo_gloss_repo}")
    print(f"[ext] mode:  {args.mode}")

    if args.mode == "combined":
        mapping = run_combined(args.pseudo_gloss_repo, unique)
    else:
        mapping = run_pos_only(args.pseudo_gloss_repo, unique)

    # Convert {sentence: [gloss_words]} → {sentence: "gloss_words ..."}
    flat = {k: " ".join(v).strip() for k, v in mapping.items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(flat, f, ensure_ascii=False, indent=2)

    empty = sum(1 for v in flat.values() if not v)
    print(f"[out] wrote {len(flat)} entries to {args.output}")
    if empty:
        print(
            f"[warn] {empty}/{len(flat)} entries produced an empty pseudo-gloss "
            f"(no content words). Downstream Step 1 will still accept them, "
            f"but the aligned token set will be empty for those samples."
        )


if __name__ == "__main__":
    main()
