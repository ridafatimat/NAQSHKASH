"""
NAQSHKASH
Build tier-level OCR dataset for CRNN + CTC.

Input:
    Day4_row_dataset/train

Each logical Talim row may contain:
    upper
    count_line
    lower

Output:
    One OCR sample for every NON-EMPTY tier.

Important:
- Tiers are separated vertically.
- Horizontal width is preserved.
- Leading spaces in targets are preserved.
- Empty tiers are recorded but not used as OCR samples.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

SOURCE_ROOT = Path(
    "datasets/Day4_row_dataset/train"
)

OUTPUT_ROOT = Path(
    "datasets/Day4_tier_dataset/train"
)

VERTICAL_PADDING = 4


# ============================================================
# HELPERS
# ============================================================

def is_nonempty_label(text):
    """
    A tier is considered non-empty if it contains
    at least one non-whitespace character.

    Leading/trailing spaces themselves are still preserved.
    """

    return bool(
        text.strip()
    )


def make_ink_mask(image):
    """
    Convert a clean synthetic Talim row image
    into a binary ink mask.

    True = ink
    False = background
    """

    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = image

    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU,
    )

    return (
        mask > 0
    )


def find_active_runs(active):
    """
    Convert a boolean 1D array into contiguous
    [start, end) runs.
    """

    runs = []

    start = None

    for i, value in enumerate(active):

        if value and start is None:
            start = i

        elif (
            not value
            and start is not None
        ):

            runs.append(
                [start, i]
            )

            start = None

    if start is not None:

        runs.append(
            [
                start,
                len(active),
            ]
        )

    return runs


def merge_close_runs(
    runs,
    max_gap,
):
    """
    Merge runs separated by a very small vertical gap.

    Useful for detached dots and small symbol components.
    """

    if not runs:
        return []

    merged = [
        runs[0][:]
    ]

    for start, end in runs[1:]:

        previous = (
            merged[-1]
        )

        gap = (
            start
            - previous[1]
        )

        if gap <= max_gap:

            previous[1] = end

        else:

            merged.append(
                [start, end]
            )

    return merged


def reduce_runs_to_expected_count(
    runs,
    expected_count,
):
    """
    If detached symbol components create too many bands,
    repeatedly merge the pair with the smallest gap until
    the expected number of tier bands remains.

    We KNOW expected_count from the JSON-derived labels.
    """

    runs = [
        run[:]
        for run in runs
    ]

    if len(runs) < expected_count:

        return None

    while len(runs) > expected_count:

        gaps = []

        for i in range(
            len(runs) - 1
        ):

            gap = (
                runs[i + 1][0]
                - runs[i][1]
            )

            gaps.append(
                (
                    gap,
                    i,
                )
            )

        _, merge_index = min(
            gaps,
            key=lambda item: item[0],
        )

        merged_run = [
            runs[merge_index][0],
            runs[merge_index + 1][1],
        ]

        runs = (
            runs[:merge_index]
            + [merged_run]
            + runs[
                merge_index + 2:
            ]
        )

    return runs


def detect_tier_bands(
    row_image,
    expected_count,
):
    """
    Detect vertical tier bands inside one logical Talim row.

    Returns bands:
        [(y0, y1), ...]

    Top-to-bottom order is preserved.
    """

    if expected_count <= 0:
        return []

    mask = make_ink_mask(
        row_image
    )

    height, width = (
        mask.shape
    )

    projection = (
        mask.sum(
            axis=1
        )
    )

    # Require at least a tiny amount of ink
    # to consider a y-coordinate active.
    threshold = max(
        1,
        int(
            round(
                width * 0.002
            )
        ),
    )

    active = (
        projection
        >= threshold
    )

    runs = find_active_runs(
        active
    )

    # Merge tiny gaps caused by dots /
    # separated symbol components.
    small_gap = max(
        2,
        int(
            round(
                height * 0.02
            )
        ),
    )

    runs = merge_close_runs(
        runs,
        max_gap=small_gap,
    )

    runs = (
        reduce_runs_to_expected_count(
            runs,
            expected_count,
        )
    )

    if runs is None:
        return None

    padded_runs = []

    for start, end in runs:

        y0 = max(
            0,
            start
            - VERTICAL_PADDING,
        )

        y1 = min(
            height,
            end
            + VERTICAL_PADDING,
        )

        padded_runs.append(
            (
                y0,
                y1,
            )
        )

    return padded_runs


# ============================================================
# LOAD ROW METADATA
# ============================================================

def load_row_metadata():

    metadata_path = (
        SOURCE_ROOT
        / "metadata.csv"
    )

    if not metadata_path.exists():

        raise FileNotFoundError(
            f"Row metadata not found:\n"
            f"{metadata_path}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        return list(
            csv.DictReader(
                f
            )
        )


# ============================================================
# BUILD DATASET
# ============================================================

def build_tier_dataset():

    output_images = (
        OUTPUT_ROOT
        / "images"
    )

    output_labels = (
        OUTPUT_ROOT
        / "labels"
    )

    output_images.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_row_metadata()

    tier_samples = []
    layout_records = []
    failures = []

    print()
    print(
        "=" * 70
    )

    print(
        "BUILDING TIER-LEVEL TALIM DATASET"
    )

    print(
        f"Logical rows: {len(rows)}"
    )

    print(
        f"Source: {SOURCE_ROOT}"
    )

    print(
        f"Output: {OUTPUT_ROOT}"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # EACH LOGICAL ROW
    # ========================================================

    for row in rows:

        row_id = (
            row["row_id"]
        )

        image_path = (
            SOURCE_ROOT
            / row["image_path"]
        )

        row_image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if row_image is None:

            failures.append(
                {
                    "row_id":
                        row_id,

                    "reason":
                        "IMAGE_READ_FAILED",
                }
            )

            print(
                f"[FAIL] {row_id}: "
                f"image read failed"
            )

            continue

        tier_labels = {
            "upper":
                row.get(
                    "upper",
                    "",
                ),

            "count":
                row.get(
                    "count_line",
                    "",
                ),

            "lower":
                row.get(
                    "lower",
                    "",
                ),
        }

        nonempty_tiers = [
            (
                tier_name,
                label,
            )
            for tier_name, label
            in tier_labels.items()
            if is_nonempty_label(
                label
            )
        ]

        expected_count = len(
            nonempty_tiers
        )

        layout_records.append(
            {
                "row_id":
                    row_id,

                "upper_present":
                    int(
                        is_nonempty_label(
                            tier_labels[
                                "upper"
                            ]
                        )
                    ),

                "count_present":
                    int(
                        is_nonempty_label(
                            tier_labels[
                                "count"
                            ]
                        )
                    ),

                "lower_present":
                    int(
                        is_nonempty_label(
                            tier_labels[
                                "lower"
                            ]
                        )
                    ),
            }
        )

        if expected_count == 0:

            failures.append(
                {
                    "row_id":
                        row_id,

                    "reason":
                        "ALL_TIERS_EMPTY",
                }
            )

            continue

        bands = detect_tier_bands(
            row_image,
            expected_count,
        )

        if bands is None:

            failures.append(
                {
                    "row_id":
                        row_id,

                    "reason":
                        "NOT_ENOUGH_VERTICAL_BANDS",

                    "expected":
                        expected_count,
                }
            )

            print(
                f"[FAIL] {row_id}: "
                f"could not detect "
                f"{expected_count} tiers"
            )

            continue

        if len(bands) != expected_count:

            failures.append(
                {
                    "row_id":
                        row_id,

                    "reason":
                        "TIER_COUNT_MISMATCH",

                    "expected":
                        expected_count,

                    "detected":
                        len(bands),
                }
            )

            continue

        # ----------------------------------------------------
        # Top-to-bottom mapping
        # ----------------------------------------------------

        for (
            tier_index,
            (
                (
                    tier_name,
                    label,
                ),
                (
                    y0,
                    y1,
                ),
            ),
        ) in enumerate(
            zip(
                nonempty_tiers,
                bands,
            )
        ):

            tier_id = (
                f"{row_id}"
                f"_{tier_name}"
            )

            # IMPORTANT:
            # Keep full horizontal width.
            tier_image = (
                row_image[
                    y0:y1,
                    :
                ]
            )

            tier_image_path = (
                output_images
                / f"{tier_id}.png"
            )

            tier_label_path = (
                output_labels
                / f"{tier_id}.txt"
            )

            success = cv2.imwrite(
                str(
                    tier_image_path
                ),
                tier_image,
            )

            if not success:

                raise RuntimeError(
                    f"Failed to save:\n"
                    f"{tier_image_path}"
                )

            # Preserve the label EXACTLY.
            tier_label_path.write_text(
                label,
                encoding="utf-8",
            )

            tier_samples.append(
                {
                    "tier_id":
                        tier_id,

                    "row_id":
                        row_id,

                    "parent_sample_id":
                        row[
                            "parent_sample_id"
                        ],

                    "row_index":
                        row[
                            "row_index"
                        ],

                    "tier_name":
                        tier_name,

                    "tier_index":
                        tier_index,

                    "image_path":
                        (
                            f"images/"
                            f"{tier_id}.png"
                        ),

                    "label_path":
                        (
                            f"labels/"
                            f"{tier_id}.txt"
                        ),

                    "label":
                        label,

                    "label_length":
                        len(
                            label
                        ),

                    "image_height":
                        tier_image.shape[
                            0
                        ],

                    "image_width":
                        tier_image.shape[
                            1
                        ],

                    "y0":
                        y0,

                    "y1":
                        y1,
                }
            )

        print(
            f"[OK] {row_id}: "
            f"{expected_count} "
            f"non-empty tier(s)"
        )

    # ========================================================
    # WRITE TIER METADATA
    # ========================================================

    metadata_path = (
        OUTPUT_ROOT
        / "metadata.csv"
    )

    fieldnames = [
        "tier_id",
        "row_id",
        "parent_sample_id",
        "row_index",
        "tier_name",
        "tier_index",
        "image_path",
        "label_path",
        "label",
        "label_length",
        "image_height",
        "image_width",
        "y0",
        "y1",
    ]

    with metadata_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            tier_samples
        )

    # ========================================================
    # WRITE LAYOUT MANIFEST
    # ========================================================

    layout_path = (
        OUTPUT_ROOT
        / "layout_manifest.csv"
    )

    with layout_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_id",
                "upper_present",
                "count_present",
                "lower_present",
            ],
        )

        writer.writeheader()

        writer.writerows(
            layout_records
        )

    # ========================================================
    # FAILURE REPORT
    # ========================================================

    failure_path = (
        OUTPUT_ROOT
        / "tier_build_failures.csv"
    )

    if failures:

        all_fields = set()

        for failure in failures:
            all_fields.update(
                failure.keys()
            )

        with failure_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=sorted(
                    all_fields
                ),
            )

            writer.writeheader()

            writer.writerows(
                failures
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "TIER DATASET BUILD COMPLETE"
    )

    print(
        f"Logical rows examined: "
        f"{len(rows)}"
    )

    print(
        f"Tier samples created: "
        f"{len(tier_samples)}"
    )

    print(
        f"Failures: "
        f"{len(failures)}"
    )

    print(
        f"Output:"
    )

    print(
        OUTPUT_ROOT
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    build_tier_dataset()