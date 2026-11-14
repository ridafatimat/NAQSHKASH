"""
verify_ctc_labels.py
====================
Dataset-agnostic verification of the NAQSHKASH CTC label pipeline (Day 5 - Ayesha Amer).

Works on ANY dataset laid out as ``<folder>/labels/*.txt`` (+ optional ``json/`` and ``images/``).
Point it at a split (``datasets/talim_1000/train``) or at a dataset folder that contains splits.

Checks
------
1. Label files parse into logical rows (4-line layout, empty tiers allowed).
2. TXT and JSON agree (when JSON exists) and, with JSON, every row has run information.
3. Token mapping: no unknown characters, encode -> decode round trip == 100%, every token class
   that appears is inside the vocabulary, class coverage (which of the 66 classes occur).
4. Tier / run structure of the labels (upper tier only UP symbols, count tier only counts, ...).
5. (--check_images) preprocessing -> row images: number of detected rows == number of label rows,
   and CTC feasibility: for every row, ``time_steps = width // downsample`` must be >=
   ``len(label) + repeats``. Reports the smallest feasible stride per mode.

Usage
-----
    python verify_ctc_labels.py --dataset_dir datasets/talim_1000 --check_images
    python verify_ctc_labels.py --dataset_dir path/to/any_split --mode both --downsample 4
Exit code 0 = all checks passed, 1 = problems found.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

from data.ctc_decode import ctc_collapse, ids_to_text, parse_run_text, parse_tier_text
from data.ctc_labels import (
    LabelFormatError,
    discover_samples,
    load_row_labels,
    required_ctc_length,
)
from data.vocabulary import TalimVocabulary


def _pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:.2f}%" if b else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify CTC labels of a Talim dataset")
    ap.add_argument("--dataset_dir", required=True, help="split folder or dataset folder")
    ap.add_argument("--vocab", default=None, help="optional vocab.json (default: canonical vocabulary)")
    ap.add_argument("--mode", choices=["tiers", "runs", "both"], default="both")
    ap.add_argument("--check_images", action="store_true", help="run preprocessing and check widths")
    ap.add_argument("--downsample", type=int, default=4, help="CNN horizontal stride of the CRNN")
    ap.add_argument("--max_samples", type=int, default=0, help="limit samples (0 = all)")
    args = ap.parse_args()

    vocab = TalimVocabulary.load(args.vocab) if args.vocab else TalimVocabulary()
    modes = ["tiers", "runs"] if args.mode == "both" else [args.mode]

    samples = discover_samples(args.dataset_dir)
    if args.max_samples:
        samples = samples[: args.max_samples]
    if not samples:
        print(f"No label files found under {args.dataset_dir}")
        return 1

    print("=" * 78)
    print(f"CTC label verification  |  {args.dataset_dir}")
    print(f"vocabulary size {len(vocab)} (blank index {vocab.blank_index})  |  modes: {', '.join(modes)}")
    print("=" * 78)

    problems: List[str] = []
    n_rows = 0
    n_json = 0
    class_counter: Counter = Counter()
    per_split: Counter = Counter()
    row_stats: Dict[str, List[int]] = {m: [] for m in modes}
    mode_ok = {m: Counter() for m in modes}
    loaded = []  # (sample, rows)

    for s in samples:
        per_split[s.split] += 1
        try:
            rows = load_row_labels(s.label_path, s.json_path)
        except (LabelFormatError, OSError, ValueError) as e:
            problems.append(f"[parse] {s.sample_id}: {e}")
            continue
        if s.json_path is not None:
            n_json += 1
        loaded.append((s, rows))
        n_rows += len(rows)
        for i, r in enumerate(rows):
            for m in modes:
                try:
                    text = r.text(m)
                except LabelFormatError as e:
                    mode_ok[m]["no_runs"] += 1
                    problems.append(f"[{m}] {s.sample_id} row {i}: {e}")
                    continue
                try:
                    ids = vocab.encode(text)
                except KeyError as e:
                    mode_ok[m]["unknown"] += 1
                    problems.append(f"[{m}] {s.sample_id} row {i}: {e}")
                    continue
                mode_ok[m]["rows"] += 1
                row_stats[m].append(len(ids))
                if ids_to_text(ctc_collapse(ids and _interleave_blanks(ids, vocab.blank_index), vocab.blank_index), vocab) == text:
                    mode_ok[m]["roundtrip"] += 1
                else:
                    problems.append(f"[{m}] {s.sample_id} row {i}: CTC round trip differs")
                struct = parse_tier_text(text) if m == "tiers" else parse_run_text(text)
                if struct.valid:
                    mode_ok[m]["structure_ok"] += 1
                else:
                    problems.append(f"[{m}] {s.sample_id} row {i}: {struct.issues}")
                if m == "tiers":
                    class_counter.update(text)

    print(f"samples: {len(samples)} ({', '.join(f'{k}: {v}' for k, v in sorted(per_split.items()))})"
          f"  |  with JSON: {n_json}  |  logical rows: {n_rows}")

    for m in modes:
        ok = mode_ok[m]
        st = row_stats[m]
        print(f"\n[{m}] rows encoded: {ok['rows']}/{n_rows}"
              f"  unknown-token rows: {ok['unknown']}  rows without run info: {ok['no_runs']}")
        print(f"      CTC round trip (encode -> blank-interleave -> collapse -> decode): "
              f"{ok['roundtrip']}/{ok['rows']} = {_pct(ok['roundtrip'], ok['rows'])}")
        print(f"      tier/run structure valid: {ok['structure_ok']}/{ok['rows']} = {_pct(ok['structure_ok'], ok['rows'])}")
        if st:
            st_sorted = sorted(st)
            print(f"      label length (tokens): min {st_sorted[0]}  median {st_sorted[len(st)//2]}  "
                  f"max {st_sorted[-1]}  mean {sum(st)/len(st):.1f}")

    if "tiers" in modes:
        present = {c for c in class_counter if c in vocab.char_to_index}
        missing = [c for c in vocab.characters if c not in class_counter]
        print(f"\nclass coverage (tiers mode): {len(present)}/{len(vocab.characters)} non-blank classes occur")
        if missing:
            shown = ", ".join(repr(c) for c in missing[:20])
            print(f"      classes never seen: {shown}{' ...' if len(missing) > 20 else ''}")

    # ---------------- images ----------------
    if args.check_images:
        try:
            from PIL import Image
            import numpy as np
            from preprocessing.model_prep import preprocess_image
        except ImportError as e:  # pragma: no cover
            print(f"\n--check_images needs Pillow + the preprocessing package: {e}")
            return 1
        print(f"\n[images] preprocessing -> row images, CTC stride {args.downsample}")
        mism = checked_docs = 0
        feas = {m: Counter() for m in modes}
        need_stride = {m: [] for m in modes}
        widths = []
        for s, rows in loaded:
            if s.image_path is None:
                continue
            checked_docs += 1
            img = np.array(Image.open(s.image_path))
            res = preprocess_image(img)
            if res.num_rows != len(rows):
                mism += 1
                problems.append(f"[images] {s.sample_id}: {res.num_rows} detected rows vs {len(rows)} label rows")
                continue
            for r, mr in zip(rows, res.rows):
                w = mr.tensor_image.shape[1]
                widths.append(w)
                for m in modes:
                    try:
                        ids = vocab.encode(r.text(m))
                    except (KeyError, LabelFormatError):
                        continue
                    need = required_ctc_length(ids)
                    feas[m]["rows"] += 1
                    if w // args.downsample >= need:
                        feas[m]["ok"] += 1
                    else:
                        problems.append(
                            f"[ctc:{m}] {s.sample_id} row {mr.row_index}: width {w}px -> "
                            f"{w // args.downsample} steps < required {need}")
                    need_stride[m].append(w / need)
        print(f"      images checked: {checked_docs}   row-count mismatches: {mism}")
        if widths:
            ws = sorted(widths)
            print(f"      row width after 64px resize: min {ws[0]}  median {ws[len(ws)//2]}  max {ws[-1]}")
        for m in modes:
            if feas[m]["rows"]:
                worst = min(need_stride[m])
                print(f"      [{m}] CTC feasible at stride {args.downsample}: {feas[m]['ok']}/{feas[m]['rows']}"
                      f" = {_pct(feas[m]['ok'], feas[m]['rows'])}"
                      f"   | largest safe CNN stride for ALL rows: {int(worst)}")

    print("\n" + "=" * 78)
    if problems:
        print(f"PROBLEMS FOUND: {len(problems)}")
        for p in problems[:25]:
            print("  -", p)
        if len(problems) > 25:
            print(f"  ... and {len(problems) - 25} more")
        return 1
    print("ALL CHECKS PASSED")
    return 0


def _interleave_blanks(ids: List[int], blank: int) -> List[int]:
    """Simulate a frame path for ``ids``: label, blank, label, blank ... (every label once)."""
    out: List[int] = []
    for t in ids:
        out.extend([t, blank])
    return out


if __name__ == "__main__":
    sys.exit(main())
