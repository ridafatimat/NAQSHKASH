"""
NAQSHKASH
Build row-level OCR dataset from full-page Talim samples.

For each full synthetic page:

1. Load full image
2. Apply preprocessing pipeline
3. Detect/separate logical Talim rows
4. Read exact row-level ground truth from JSON logical_rows
5. Save one image + one TXT label per logical row
6. Write row-level metadata.csv

This prepares the dataset for CRNN + CTC training.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2

from preprocessing.pipeline import (
    crop_and_separate_rows,
)


# ============================================================
# CONFIG
# ============================================================

SOURCE_ROOT = Path(
    "datasets/Day4_sample/train"
)

OUTPUT_ROOT = Path(
    "datasets/Day4_row_dataset/train"
)

APPLY_DESKEW = False


# ============================================================
# HELPERS
# ============================================================

def make_row_label(
    upper: str,
    count_line: str,
    lower: str,
) -> str:
    """
    Create canonical row label.

    Important:
    - Keep spaces exactly as stored.
    - Keep empty upper/lower lines.
    - Use exactly three logical lines:
          upper
          count_line
          lower

    Example:

        " P"
        "aj"
        "W"

    becomes:

        " P\\naj\\nW"
    """

    return (
        f"{upper}\n"
        f"{count_line}\n"
        f"{lower}"
    )


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


# ============================================================
# MAIN
# ============================================================

def build_row_dataset():

    images_source = (
        SOURCE_ROOT
        / "images"
    )

    json_source = (
        SOURCE_ROOT
        / "json"
    )

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

    image_files = sorted(
        images_source.glob(
            "*.png"
        )
    )

    if not image_files:

        raise ValueError(
            f"No PNG images found in:\n"
            f"{images_source}"
        )

    metadata_rows = []

    pages_processed = 0
    rows_created = 0
    failures = []

    print()
    print(
        "=" * 70
    )

    print(
        "Building row-level Talim dataset"
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
    # LOOP THROUGH FULL PAGE IMAGES
    # ========================================================

    for image_path in image_files:

        sample_id = (
            image_path.stem
        )

        json_path = (
            json_source
            / f"{sample_id}.json"
        )

        if not json_path.exists():

            failures.append(
                {
                    "sample_id":
                        sample_id,

                    "reason":
                        "JSON_MISSING",
                }
            )

            print(
                f"[FAIL] {sample_id}: "
                f"JSON missing"
            )

            continue

        # ----------------------------------------------------
        # Load full page image
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:

            failures.append(
                {
                    "sample_id":
                        sample_id,

                    "reason":
                        "IMAGE_READ_FAILED",
                }
            )

            print(
                f"[FAIL] {sample_id}: "
                f"image read failed"
            )

            continue

        # ----------------------------------------------------
        # Read exact logical-row labels from JSON
        # ----------------------------------------------------

        data = load_json(
            json_path
        )

        logical_rows = (
            data.get(
                "logical_rows",
                [],
            )
        )

        expected_rows = len(
            logical_rows
        )

        if expected_rows == 0:

            failures.append(
                {
                    "sample_id":
                        sample_id,

                    "reason":
                        "NO_LOGICAL_ROWS_IN_JSON",
                }
            )

            print(
                f"[FAIL] {sample_id}: "
                f"no logical_rows"
            )

            continue

        # ----------------------------------------------------
        # Run preprocessing / row separation
        # ----------------------------------------------------

        result = (
            crop_and_separate_rows(
                image,
                apply_deskew=APPLY_DESKEW,
            )
        )

        detected_rows = (
            result.rows
        )

        detected_count = len(
            detected_rows
        )

        # ----------------------------------------------------
        # Critical alignment check
        # ----------------------------------------------------

        if (
            detected_count
            != expected_rows
        ):

            failures.append(
                {
                    "sample_id":
                        sample_id,

                    "reason":
                        "ROW_COUNT_MISMATCH",

                    "expected_rows":
                        expected_rows,

                    "detected_rows":
                        detected_count,
                }
            )

            print(
                f"[FAIL] {sample_id}: "
                f"expected {expected_rows}, "
                f"detected {detected_count}"
            )

            continue

        # ----------------------------------------------------
        # Save each logical row
        # ----------------------------------------------------

        for row_index, (
            row_data,
            label_info,
        ) in enumerate(
            zip(
                detected_rows,
                logical_rows,
            )
        ):

            row_id = (
                f"{sample_id}"
                f"_row_"
                f"{row_index:02d}"
            )

            row_image = (
                row_data.image
            )

            upper = (
                label_info.get(
                    "upper",
                    "",
                )
            )

            count_line = (
                label_info.get(
                    "count_line",
                    "",
                )
            )

            lower = (
                label_info.get(
                    "lower",
                    "",
                )
            )

            row_label = (
                make_row_label(
                    upper,
                    count_line,
                    lower,
                )
            )

            # -----------------------------------------------
            # Save image
            # -----------------------------------------------

            row_image_path = (
                output_images
                / f"{row_id}.png"
            )

            success = cv2.imwrite(
                str(
                    row_image_path
                ),
                row_image,
            )

            if not success:

                raise RuntimeError(
                    f"Failed to save:\n"
                    f"{row_image_path}"
                )

            # -----------------------------------------------
            # Save label
            # -----------------------------------------------

            row_label_path = (
                output_labels
                / f"{row_id}.txt"
            )

            row_label_path.write_text(
                row_label,
                encoding="utf-8",
            )

            # -----------------------------------------------
            # Metadata
            # -----------------------------------------------

            metadata_rows.append(
                {
                    "row_id":
                        row_id,

                    "parent_sample_id":
                        sample_id,

                    "row_index":
                        row_index,

                    "image_path":
                        (
                            f"images/"
                            f"{row_id}.png"
                        ),

                    "label_path":
                        (
                            f"labels/"
                            f"{row_id}.txt"
                        ),

                    "upper":
                        upper,

                    "count_line":
                        count_line,

                    "lower":
                        lower,

                    "label_length":
                        len(
                            row_label
                        ),

                    "image_height":
                        row_image.shape[
                            0
                        ],

                    "image_width":
                        row_image.shape[
                            1
                        ],
                }
            )

            rows_created += 1

        pages_processed += 1

        print(
            f"[OK] {sample_id}: "
            f"{expected_rows} rows"
        )

    # ========================================================
    # WRITE METADATA
    # ========================================================

    metadata_path = (
        OUTPUT_ROOT
        / "metadata.csv"
    )

    fieldnames = [
        "row_id",
        "parent_sample_id",
        "row_index",
        "image_path",
        "label_path",
        "upper",
        "count_line",
        "lower",
        "label_length",
        "image_height",
        "image_width",
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
            metadata_rows
        )

    # ========================================================
    # WRITE FAILURE REPORT
    # ========================================================

    failure_path = (
        OUTPUT_ROOT
        / "row_build_failures.csv"
    )

    if failures:

        all_fields = set()

        for failure in failures:
            all_fields.update(
                failure.keys()
            )

        failure_fields = sorted(
            all_fields
        )

        with failure_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=failure_fields,
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
        "ROW DATASET BUILD COMPLETE"
    )

    print(
        f"Pages processed: "
        f"{pages_processed}"
    )

    print(
        f"Rows created: "
        f"{rows_created}"
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
    build_row_dataset()