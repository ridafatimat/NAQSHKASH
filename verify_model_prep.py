"""
verify_model_prep.py
====================
Verification and benchmark script for model preparation preprocessing modules.

Author: Ayesha Amer
Project: NAQSHKASH FYP

Checks:
1. Shape verification: Row height strictly 64, variable width >= min_width.
2. Dtype & Value Range: float32 strictly in [0.0, 1.0], zero NaNs, zero Infs.
3. Width distribution: Min, max, mean, narrowest rows, CTC width-to-label check.
4. Dot & Stroke Preservation: Connected-component preservation under 64px downscaling.
5. Saves visual verification outputs to `debug_outputs/model_prep/`.
"""

import os
import glob
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy import ndimage

from preprocessing.normalize import (
    to_grayscale,
    resize_to_height,
    normalize,
    preprocess_row,
)
from preprocessing.model_prep import preprocess_image, ModelPrepResult


def required_ctc_steps(label: str) -> int:
    """Minimum CTC time steps for a label: its length + one blank per identical neighbour pair."""
    return len(label) + sum(1 for a, b in zip(label, label[1:]) if a == b)


def test_dot_and_stroke_retention() -> Dict[str, Any]:
    """
    Test whether thin strokes, dots, and diacritics survive downscaling to 64px height.
    Creates high-res synthetic glyphs with 2x2 dots and 1px strokes and measures connected components.
    """
    canvas = np.full((160, 400), 255, dtype=np.uint8)
    
    # 5 isolated dots (e.g. Kashmiri diacritics / nuktas)
    dot_coords = [(30, 50), (35, 120), (32, 200), (38, 280), (34, 350)]
    for y, x in dot_coords:
        canvas[y:y+3, x:x+3] = 0

    # 3 thin 1px horizontal/vertical strokes
    canvas[70:120, 80] = 0
    canvas[95, 150:230] = 0
    canvas[70:120, 300] = 0

    # Count original components (ndimage.label returns (labeled_array, num_features))
    orig_ink = canvas < 128
    _, num_orig_cc = ndimage.label(orig_ink)

    # Downscale to 64px height using cv2.INTER_AREA
    processed = preprocess_row(canvas, target_height=64)

    # In normalized space: ink is < 0.8 (where 0.0 is black ink, 1.0 is white background)
    downscaled_ink = processed < 0.8
    _, num_downscaled_cc = ndimage.label(downscaled_ink)

    # Check each dot region to ensure minimum pixel intensity is significantly dark
    scale = 64.0 / 160.0
    dots_detected = 0
    for y, x in dot_coords:
        dy = int(round(y * scale))
        dx = int(round(x * scale))
        neighborhood = processed[max(0, dy-2):min(64, dy+3), max(0, dx-2):min(processed.shape[1], dx+3)]
        if neighborhood.min() < 0.75:  # Ink detected
            dots_detected += 1

    return {
        "original_cc_count": num_orig_cc,
        "downscaled_cc_count": num_downscaled_cc,
        "total_test_dots": len(dot_coords),
        "dots_preserved": dots_detected,
        "retention_rate": (dots_detected / len(dot_coords)) * 100.0,
    }


def save_model_prep_visualizations(
    sample_id: str,
    raw_img: np.ndarray,
    result: ModelPrepResult,
    output_dir: Path,
) -> None:
    """Save inspection artifacts to `debug_outputs/model_prep/`."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for row in result.rows:
        # Save normalized row image as PNG (scale [0, 1] -> [0, 255])
        u8_tensor = (np.clip(row.tensor_image, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil_row = Image.fromarray(u8_tensor, mode="L")
        out_path = output_dir / f"{sample_id}_row_{row.row_index}_64px.png"
        pil_row.save(out_path)


def verify_dataset(
    dataset_name: str,
    img_dir: str,
    json_dir: str,
    output_dir: Path,
    save_vis_count: int = 2,
    labels_dir: Optional[str] = None,
    downsample: int = 4,
    ctc_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Verify samples in a dataset split for model preparation specifications."""
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    if not json_files:
        return {"name": dataset_name, "total_docs": 0, "total_rows": 0, "all_valid": True, "widths": []}

    total_docs = 0
    total_rows = 0
    all_valid = True
    widths = []
    narrow_rows = []
    vis_saved = 0

    for jf in json_files:
        base = os.path.splitext(os.path.basename(jf))[0]
        img_f = os.path.join(img_dir, base + ".png")
        if not os.path.exists(img_f):
            continue

        with open(jf, "r", encoding="utf-8") as f:
            meta = json.load(f)

        pil_img = Image.open(img_f)
        raw_arr = np.array(pil_img)

        # Run end-to-end model preparation pipeline
        result = preprocess_image(raw_arr)
        total_docs += 1

        for row in result.rows:
            total_rows += 1
            t_img = row.tensor_image

            # 1. Height must be exactly 64
            is_h64 = (t_img.shape[0] == 64)
            # 2. Dtype float32
            is_f32 = (t_img.dtype == np.float32)
            # 3. Value range [0, 1]
            is_bounded = (t_img.min() >= 0.0 and t_img.max() <= 1.0)
            # 4. No NaN or Inf
            has_no_nan = not np.isnan(t_img).any() and not np.isinf(t_img).any()

            valid = is_h64 and is_f32 and is_bounded and has_no_nan
            if not valid:
                all_valid = False

            w = t_img.shape[1]
            widths.append(w)
            if w < 64:
                narrow_rows.append({"sample": base, "row": row.row_index, "width": w})

        if labels_dir is not None and ctc_summary is not None:
            lf = os.path.join(labels_dir, base + ".txt")
            if os.path.exists(lf):
                # rows are 4-line groups (upper, count, lower, blank); NOT split on blank lines,
                # because an empty upper/lower tier also creates a blank line inside a row
                from data.ctc_labels import load_row_labels
                blocks = [r.text("tiers") for r in load_row_labels(lf)]
                if len(blocks) != len(result.rows):
                    ctc_summary["row_count_mismatch"] += 1
                else:
                    for blk, row in zip(blocks, result.rows):
                        need = required_ctc_steps(blk)
                        have = row.tensor_image.shape[1] // downsample
                        ctc_summary["checked"] += 1
                        if have < need:
                            ctc_summary["too_short"] += 1
                            ctc_summary["examples"].append(
                                f"{base} row {row.row_index}: width {row.tensor_image.shape[1]}px -> "
                                f"{have} steps < required {need}")

        if vis_saved < save_vis_count:
            save_model_prep_visualizations(base, raw_arr, result, output_dir)
            vis_saved += 1

    return {
        "name": dataset_name,
        "total_docs": total_docs,
        "total_rows": total_rows,
        "all_valid": all_valid,
        "widths": widths,
        "narrow_rows": narrow_rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Model Preparation Preprocessing Verification")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="debug_outputs/model_prep",
        help="Output directory for visual inspection artifacts.",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default=None,
        help="Optional folder with <sample_id>.txt label files. Enables the REAL CTC width check "
             "(row labels = the 4-line row groups of each label file).",
    )
    parser.add_argument(
        "--cnn_downsample",
        type=int,
        default=4,
        help="Horizontal down-sampling factor of the CRNN CNN (time steps = width // factor).",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Any dataset/split folder with <split>/images, <split>/json (and labels). "
             "If omitted, the built-in list of datasets/<name> folders is used.",
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    print("=" * 80)
    print("NAQSHKASH Model Preparation Preprocessing Verification — Ayesha Amer")
    print("=" * 80)

    # 1. Dot and stroke preservation test
    print("\n[1] Running Dot & Thin-Stroke Retention Analysis...")
    dot_results = test_dot_and_stroke_retention()
    print(f"    - Original components detected: {dot_results['original_cc_count']}")
    print(f"    - Downscaled components (64px): {dot_results['downscaled_cc_count']}")
    print(f"    - Dots preserved: {dot_results['dots_preserved']} / {dot_results['total_test_dots']} ({dot_results['retention_rate']:.1f}%)")

    # 2. Dataset validation (Train & Validation splits only - NEVER test split)
    print("\n[2] Evaluating Dataset Splits (Train / Validation only)...")
    datasets_to_check = [
        ("baseline_normal", "datasets/baseline_normal/images", "datasets/baseline_normal/json"),
        ("count_bucket_balanced", "datasets/count_bucket_balanced/images", "datasets/count_bucket_balanced/json"),
        ("count_edge_cases", "datasets/count_edge_cases/images", "datasets/count_edge_cases/json"),
        ("layout_alignment_edge_cases", "datasets/layout_alignment_edge_cases/images", "datasets/layout_alignment_edge_cases/json"),
        ("talim_1000_preflight (train)", "datasets/talim_1000_preflight/train/images", "datasets/talim_1000_preflight/train/json"),
        ("talim_1000_preflight (validation)", "datasets/talim_1000_preflight/validation/images", "datasets/talim_1000_preflight/validation/json"),
    ]

    if args.dataset_dir:
        from data.ctc_labels import discover_samples
        seen = {}
        for smp in discover_samples(args.dataset_dir):
            split_dir = smp.label_path.parent.parent
            seen[split_dir] = None
        datasets_to_check = [
            (f"{d.parent.name}/{d.name}" if d.parent != d else d.name,
             str(d / "images"), str(d / "json"))
            for d in sorted(seen)
        ]
        if args.labels_dir is None and len(seen) == 1:
            args.labels_dir = str(next(iter(seen)) / "labels")

    all_widths = []
    total_docs_count = 0
    total_rows_count = 0
    all_datasets_valid = True
    ctc_summary = {"checked": 0, "too_short": 0, "row_count_mismatch": 0, "examples": []}

    print(f"\n{'Dataset Name':<36} | {'Docs':>5} | {'Rows':>6} | {'H==64 & [0,1]':>13} | {'Mean Width':>10}")
    print("-" * 80)

    for name, img_d, json_d in datasets_to_check:
        if not os.path.exists(json_d):
            continue
        res = verify_dataset(name, img_d, json_d, output_dir=out_dir,
                             labels_dir=args.labels_dir, downsample=args.cnn_downsample,
                             ctc_summary=ctc_summary)
        all_datasets_valid = all_datasets_valid and res["all_valid"]
        total_docs_count += res["total_docs"]
        total_rows_count += res["total_rows"]
        all_widths.extend(res["widths"])
        mean_w = np.mean(res["widths"]) if res["widths"] else 0.0
        status_str = "PASSED" if res["all_valid"] else "FAILED"
        print(f"{name:<36} | {res['total_docs']:>5} | {res['total_rows']:>6} | {status_str:>13} | {mean_w:>9.1f}px")

    print("-" * 80)
    overall_status = "PASSED" if all_datasets_valid else "FAILED"
    print(f"{'OVERALL TOTALS':<36} | {total_docs_count:>5} | {total_rows_count:>6} | {overall_status:>13} |")
    print("=" * 80)

    if all_widths:
        print("\n[3] Width Distribution Summary across Processed Rows:")
        print(f"    - Min Width:  {min(all_widths)} px")
        print(f"    - Max Width:  {max(all_widths)} px")
        print(f"    - Mean Width: {np.mean(all_widths):.1f} px")
        print(f"    - Median Width: {np.median(all_widths):.1f} px")
        if args.labels_dir is None:
            print("    - CTC Width Check: NOT RUN (pass --labels_dir to compare row widths with label lengths).")
        else:
            print(f"    - CTC Width Check (time steps = width // {args.cnn_downsample}, "
                  f"required = len(label) + repeated-neighbour count):")
            print(f"        rows checked: {ctc_summary['checked']}, too short for their label: "
                  f"{ctc_summary['too_short']}, docs where #rows != #label blocks: "
                  f"{ctc_summary['row_count_mismatch']}")
            for ex in ctc_summary["examples"][:10]:
                print(f"        e.g. {ex}")

    print(f"\nVisual debug artifacts saved to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
