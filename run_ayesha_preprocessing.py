"""
run_ayesha_preprocessing.py
===========================
Benchmark and visualization script for Day 1 & Day 2 preprocessing modules.

Author: Ayesha Amer (Day 1-2 Scope)
Project: NAQSHKASH FYP

Features:
- Evaluates `crop_margins`, `detect_rows`, and `crop_and_separate_rows` across datasets.
- Compares detected row count against ground-truth `"rows"` metadata in JSON records.
- Saves visual debug artifacts (bounding boxes, row bands, projection profiles) to `debug_outputs/`.
- Generates a summary accuracy table.
"""

import os
import glob
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from preprocessing.crop import crop_margins
from preprocessing.rows import detect_rows, separate_rows
from preprocessing.pipeline import crop_and_separate_rows, PreprocessingResult


def save_debug_visualizations(
    sample_id: str,
    raw_img: np.ndarray,
    result: PreprocessingResult,
    output_dir: Path,
) -> None:
    """
    Generate and save visual inspection artifacts to `debug_outputs/`.
    
    1. `_crop_overlay.png`: Original image with green bounding box around cropped region.
    2. `_row_overlay.png`: Cropped image with color-coded bounding boxes and row labels.
    3. `_projection_plot.png`: Horizontal projection profile with row division cut lines.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Crop Overlay
    if raw_img.ndim == 2:
        pil_raw = Image.fromarray(raw_img).convert("RGB")
    else:
        pil_raw = Image.fromarray(raw_img)

    draw_raw = ImageDraw.Draw(pil_raw)
    c_ymin, c_xmin, c_ymax, c_xmax = result.crop_bbox
    # Draw green rectangle for crop region
    draw_raw.rectangle(
        [(c_xmin, c_ymin), (c_xmax - 1, c_ymax - 1)],
        outline=(0, 200, 0),
        width=3,
    )
    crop_overlay_path = output_dir / f"{sample_id}_crop_overlay.png"
    pil_raw.save(crop_overlay_path)

    # 2. Row Overlay on Cropped Image
    cropped_arr = result.cropped_image
    if cropped_arr.ndim == 2:
        pil_crop = Image.fromarray(cropped_arr).convert("RGB")
    else:
        pil_crop = Image.fromarray(cropped_arr)

    draw_crop = ImageDraw.Draw(pil_crop)
    palette = [
        (220, 50, 50),    # Red
        (30, 144, 255),   # Blue
        (46, 139, 87),    # Green
        (255, 140, 0),    # Orange
        (148, 0, 211),    # Purple
        (0, 180, 200),    # Cyan
    ]

    for idx, row in enumerate(result.rows):
        r_ymin, r_xmin, r_ymax, r_xmax = row.bbox
        color = palette[idx % len(palette)]
        # Draw bounding box around row
        draw_crop.rectangle(
            [(r_xmin, r_ymin), (r_xmax - 1, r_ymax - 1)],
            outline=color,
            width=2,
        )
        # Add label
        label_text = f"Row {idx} ({row.height}x{row.width})"
        draw_crop.text(
            (r_xmin + 4, max(0, r_ymin + 2)),
            label_text,
            fill=color,
        )

    row_overlay_path = output_dir / f"{sample_id}_row_overlay.png"
    pil_crop.save(row_overlay_path)

    # 3. Projection Profile Plot
    if cropped_arr.ndim == 3:
        gray = (
            0.299 * cropped_arr[:, :, 0].astype(np.float32)
            + 0.587 * cropped_arr[:, :, 1].astype(np.float32)
            + 0.114 * cropped_arr[:, :, 2].astype(np.float32)
        ).astype(np.uint8)
    else:
        gray = cropped_arr.copy()

    ink_mask = gray < 128
    proj = np.sum(ink_mask, axis=1)
    h = len(proj)

    fig, ax = plt.subplots(figsize=(7, max(4, h / 70)), dpi=120)
    y_coords = np.arange(h)
    ax.plot(proj, y_coords, color="navy", lw=1.5, label="Horizontal Ink Projection")
    ax.fill_betweenx(y_coords, 0, proj, color="royalblue", alpha=0.3)

    for idx, row in enumerate(result.rows):
        r_ymin, _, r_ymax, _ = row.bbox
        color = palette[idx % len(palette)]
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        ax.axhline(r_ymin, color=color_hex, linestyle="--", alpha=0.8)
        ax.axhline(r_ymax, color=color_hex, linestyle="-.", alpha=0.8)
        ax.text(
            max(proj) * 0.7,
            (r_ymin + r_ymax) / 2.0,
            f"Row {idx}",
            color=color_hex,
            fontweight="bold",
            va="center",
        )

    ax.set_ylim(h, 0)  # Invert y-axis to match image coordinates (0 at top)
    ax.set_xlabel("Ink Pixel Count (Projection)")
    ax.set_ylabel("Vertical Coordinate Y (px)")
    ax.set_title(f"Horizontal Projection & Row Segmentation: {sample_id}")
    ax.legend(loc="upper right")
    plt.tight_layout()

    proj_plot_path = output_dir / f"{sample_id}_projection_plot.png"
    plt.savefig(proj_plot_path)
    plt.close(fig)


def benchmark_dataset(
    dataset_name: str,
    img_dir: str,
    json_dir: str,
    save_vis_count: int = 2,
    output_dir: Path = Path("debug_outputs"),
) -> Dict[str, Any]:
    """Evaluate all samples in a dataset folder and return metrics."""
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    if not json_files:
        return {"name": dataset_name, "total": 0, "correct": 0, "accuracy": 0.0, "mismatches": []}

    total = 0
    correct = 0
    mismatches = []
    vis_saved = 0

    for jf in json_files:
        base = os.path.splitext(os.path.basename(jf))[0]
        img_f = os.path.join(img_dir, base + ".png")
        if not os.path.exists(img_f):
            continue

        with open(jf, "r", encoding="utf-8") as f:
            meta = json.load(f)
        exp_rows = meta.get("rows")
        if exp_rows is None:
            continue

        pil_img = Image.open(img_f)
        raw_arr = np.array(pil_img)

        # Run pipeline
        result = crop_and_separate_rows(raw_arr)

        total += 1
        if result.num_rows == exp_rows:
            correct += 1
        else:
            mismatches.append({
                "sample_id": base,
                "expected": exp_rows,
                "detected": result.num_rows,
            })

        # Save visual artifacts for first few samples of each dataset
        if vis_saved < save_vis_count:
            save_debug_visualizations(base, raw_arr, result, output_dir)
            vis_saved += 1

    acc = (correct / total * 100.0) if total > 0 else 0.0
    return {
        "name": dataset_name,
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "mismatches": mismatches,
    }


def main():
    parser = argparse.ArgumentParser(description="NAQSHKASH Day 1-2 Preprocessing Benchmark")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="debug_outputs",
        help="Directory to save visual debug artifacts.",
    )
    parser.add_argument(
        "--max_preflight_samples",
        type=int,
        default=50,
        help="Maximum samples to evaluate from large preflight splits.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    print("=" * 78)
    print("NAQSHKASH Preprocessing Benchmark — Ayesha Amer (Day 1 & Day 2 Scope)")
    print("=" * 78)

    datasets_to_run = [
        ("baseline_normal", "datasets/baseline_normal/images", "datasets/baseline_normal/json"),
        ("count_bucket_balanced", "datasets/count_bucket_balanced/images", "datasets/count_bucket_balanced/json"),
        ("count_edge_cases", "datasets/count_edge_cases/images", "datasets/count_edge_cases/json"),
        ("layout_alignment_edge_cases", "datasets/layout_alignment_edge_cases/images", "datasets/layout_alignment_edge_cases/json"),
        ("structure_direction_edge_cases", "datasets/structure_direction_edge_cases/images", "datasets/structure_direction_edge_cases/json"),
        ("symbol_variety", "datasets/symbol_variety/images", "datasets/symbol_variety/json"),
        ("talim_1000_preflight (train)", "datasets/talim_1000_preflight/train/images", "datasets/talim_1000_preflight/train/json"),
        ("talim_1000_preflight (validation)", "datasets/talim_1000_preflight/validation/images", "datasets/talim_1000_preflight/validation/json"),
    ]

    results = []
    grand_total = 0
    grand_correct = 0
    all_mismatches = []

    for name, img_d, json_d in datasets_to_run:
        if not os.path.exists(json_d):
            continue
        res = benchmark_dataset(name, img_d, json_d, save_vis_count=2, output_dir=out_dir)
        results.append(res)
        grand_total += res["total"]
        grand_correct += res["correct"]
        if res["mismatches"]:
            all_mismatches.extend([(name, m) for m in res["mismatches"]])

    # Print Formatted Table
    print(f"\n{'Dataset Name':<38} | {'Total':>7} | {'Correct':>7} | {'Accuracy':>8}")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:<38} | {r['total']:>7} | {r['correct']:>7} | {r['accuracy']:>7.2f}%")
    print("-" * 78)
    overall_acc = (grand_correct / grand_total * 100.0) if grand_total > 0 else 0.0
    print(f"{'OVERALL TOTAL':<38} | {grand_total:>7} | {grand_correct:>7} | {overall_acc:>7.2f}%")
    print("=" * 78)

    if all_mismatches:
        print(f"\nDiscrepancies / Mismatches ({len(all_mismatches)} total):")
        for dname, m in all_mismatches:
            print(f"  [{dname}] Sample: {m['sample_id']} | Expected: {m['expected']} rows | Detected: {m['detected']} rows")
    else:
        print("\nAll evaluated samples matched the ground-truth row count with 100.0% accuracy!")

    print(f"\nVisual debug artifacts saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
