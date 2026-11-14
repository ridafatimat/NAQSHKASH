"""
NAQSHKASH

Production builder for the full talim_100000 dataset.

Reads the original 100k dataset directly from ZIP without extracting it.

Pipeline
--------
ZIP page image
    -> preprocessing
    -> logical row separation
    -> JSON logical-row ground truth
    -> upper/count/lower tier separation
    -> tier-level CRNN + CTC dataset

The original train / validation / test split is preserved.

Example
-------
python -m data.build_full_tier_dataset --split train --limit 1000

python -m data.build_full_tier_dataset --split train
python -m data.build_full_tier_dataset --split validation
python -m data.build_full_tier_dataset --split test
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from preprocessing.pipeline import crop_and_separate_rows


# ============================================================
# CONFIG
# ============================================================

ZIP_PATH = Path(
    "datasets/Final dataset/talim_100000.zip"
)

OUTPUT_BASE = Path(
    "datasets/Final tier dataset"
)

ZIP_DATASET_ROOT = "datasets/talim_100000"

APPLY_DESKEW = False

VERTICAL_PADDING = 4


# ============================================================
# HELPERS
# ============================================================

def is_nonempty_label(text):
    """
    True when the tier contains at least one
    non-whitespace character.
    """

    return bool(text.strip())


def decode_image_from_zip(zip_file, member_name):
    """
    Read PNG bytes directly from ZIP and decode
    them as a grayscale OpenCV image.
    """

    raw_bytes = zip_file.read(member_name)

    buffer = np.frombuffer(
        raw_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        buffer,
        cv2.IMREAD_GRAYSCALE,
    )

    return image


def read_json_from_zip(zip_file, member_name):
    """
    Read JSON directly from ZIP.
    """

    raw_bytes = zip_file.read(
        member_name
    )

    text = raw_bytes.decode(
        "utf-8"
    )

    return json.loads(
        text
    )


# ============================================================
# TIER-DETECTION LOGIC
# ============================================================

def make_ink_mask(image):
    """
    Convert synthetic Talim image into binary ink mask.

    True  = foreground ink
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

    return mask > 0


def find_active_runs(active):
    """
    Convert boolean 1D array into contiguous
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
    Merge vertically close runs.

    This handles detached dots and symbol parts.
    """

    if not runs:

        return []

    merged = [
        runs[0][:]
    ]

    for start, end in runs[1:]:

        previous = merged[-1]

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
    Reduce detected components until the expected
    number of Talim tiers remains.

    The expected count comes from ground-truth JSON.
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
    Detect upper/count/lower vertical bands.

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

    projection = mask.sum(
        axis=1
    )

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

    runs = reduce_runs_to_expected_count(
        runs,
        expected_count,
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
# CSV HELPERS
# ============================================================

TIER_FIELDNAMES = [
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


LAYOUT_FIELDNAMES = [
    "row_id",
    "parent_sample_id",
    "row_index",
    "upper_present",
    "count_present",
    "lower_present",
]


FAILURE_FIELDNAMES = [
    "sample_id",
    "row_id",
    "reason",
    "expected_rows",
    "detected_rows",
    "expected_tiers",
    "detected_tiers",
]


def ensure_csv(
    path,
    fieldnames,
):
    """
    Create CSV with header if it does not already exist.
    """

    if path.exists():

        return

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()


def append_csv_row(
    path,
    fieldnames,
    row,
):

    safe_row = {
        key: row.get(
            key,
            "",
        )
        for key in fieldnames
    }

    with path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writerow(
            safe_row
        )


# ============================================================
# RESUME SUPPORT
# ============================================================

def load_completed_pages(path):
    """
    Read page IDs that have already been processed.

    Makes the 80k build resumable.
    """

    if not path.exists():

        return set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return {
            line.strip()
            for line in f
            if line.strip()
        }


def mark_page_completed(
    path,
    sample_id,
):

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            sample_id
            + "\n"
        )


# ============================================================
# FIND SPLIT IMAGES INSIDE ZIP
# ============================================================

def get_split_image_members(
    zip_file,
    split,
):

    prefix = (
        f"{ZIP_DATASET_ROOT}/"
        f"{split}/images/"
    )

    image_members = [
        name
        for name in zip_file.namelist()
        if (
            name.startswith(prefix)
            and name.lower().endswith(
                ".png"
            )
        )
    ]

    return sorted(
        image_members
    )


# ============================================================
# PROCESS ONE PAGE
# ============================================================

def process_page(
    zip_file,
    split,
    image_member,
    output_root,
    metadata_path,
    layout_path,
    failure_path,
):

    sample_id = Path(
        image_member
    ).stem

    json_member = (
        f"{ZIP_DATASET_ROOT}/"
        f"{split}/json/"
        f"{sample_id}.json"
    )

    # --------------------------------------------------------
    # Check JSON
    # --------------------------------------------------------

    try:

        data = read_json_from_zip(
            zip_file,
            json_member,
        )

    except KeyError:

        append_csv_row(
            failure_path,
            FAILURE_FIELDNAMES,
            {
                "sample_id":
                    sample_id,

                "reason":
                    "JSON_MISSING",
            },
        )

        return False, 0, 0

    except Exception as exc:

        append_csv_row(
            failure_path,
            FAILURE_FIELDNAMES,
            {
                "sample_id":
                    sample_id,

                "reason":
                    (
                        "JSON_READ_FAILED: "
                        + str(exc)
                    ),
            },
        )

        return False, 0, 0

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = decode_image_from_zip(
        zip_file,
        image_member,
    )

    if image is None:

        append_csv_row(
            failure_path,
            FAILURE_FIELDNAMES,
            {
                "sample_id":
                    sample_id,

                "reason":
                    "IMAGE_READ_FAILED",
            },
        )

        return False, 0, 0

    # --------------------------------------------------------
    # Logical rows from ground truth
    # --------------------------------------------------------

    logical_rows = data.get(
        "logical_rows",
        [],
    )

    expected_rows = len(
        logical_rows
    )

    if expected_rows == 0:

        append_csv_row(
            failure_path,
            FAILURE_FIELDNAMES,
            {
                "sample_id":
                    sample_id,

                "reason":
                    "NO_LOGICAL_ROWS_IN_JSON",
            },
        )

        return False, 0, 0

    # --------------------------------------------------------
    # Preprocess / row separation
    # --------------------------------------------------------

    try:

        result = crop_and_separate_rows(
            image,
            apply_deskew=APPLY_DESKEW,
        )

    except Exception as exc:

        append_csv_row(
            failure_path,
            FAILURE_FIELDNAMES,
            {
                "sample_id":
                    sample_id,

                "reason":
                    (
                        "PREPROCESSING_FAILED: "
                        + str(exc)
                    ),
            },
        )

        return False, 0, 0

    detected_rows = (
        result.rows
    )

    detected_count = len(
        detected_rows
    )

    # Critical alignment check
    if detected_count != expected_rows:

        append_csv_row(
            failure_path,
            FAILURE_FIELDNAMES,
            {
                "sample_id":
                    sample_id,

                "reason":
                    "ROW_COUNT_MISMATCH",

                "expected_rows":
                    expected_rows,

                "detected_rows":
                    detected_count,
            },
        )

        return False, 0, 0

    rows_created = 0

    tiers_created = 0

    output_images = (
        output_root
        / "images"
    )

    output_labels = (
        output_root
        / "labels"
    )

    # --------------------------------------------------------
    # Each logical row
    # --------------------------------------------------------

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

        upper = label_info.get(
            "upper",
            "",
        )

        count_line = label_info.get(
            "count_line",
            "",
        )

        lower = label_info.get(
            "lower",
            "",
        )

        tier_labels = {
            "upper":
                upper,

            "count":
                count_line,

            "lower":
                lower,
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

        expected_tiers = len(
            nonempty_tiers
        )

        # Layout metadata
        append_csv_row(
            layout_path,
            LAYOUT_FIELDNAMES,
            {
                "row_id":
                    row_id,

                "parent_sample_id":
                    sample_id,

                "row_index":
                    row_index,

                "upper_present":
                    int(
                        is_nonempty_label(
                            upper
                        )
                    ),

                "count_present":
                    int(
                        is_nonempty_label(
                            count_line
                        )
                    ),

                "lower_present":
                    int(
                        is_nonempty_label(
                            lower
                        )
                    ),
            },
        )

        if expected_tiers == 0:

            append_csv_row(
                failure_path,
                FAILURE_FIELDNAMES,
                {
                    "sample_id":
                        sample_id,

                    "row_id":
                        row_id,

                    "reason":
                        "ALL_TIERS_EMPTY",
                },
            )

            continue

        bands = detect_tier_bands(
            row_image,
            expected_tiers,
        )

        if bands is None:

            append_csv_row(
                failure_path,
                FAILURE_FIELDNAMES,
                {
                    "sample_id":
                        sample_id,

                    "row_id":
                        row_id,

                    "reason":
                        "NOT_ENOUGH_VERTICAL_BANDS",

                    "expected_tiers":
                        expected_tiers,
                },
            )

            continue

        if len(bands) != expected_tiers:

            append_csv_row(
                failure_path,
                FAILURE_FIELDNAMES,
                {
                    "sample_id":
                        sample_id,

                    "row_id":
                        row_id,

                    "reason":
                        "TIER_COUNT_MISMATCH",

                    "expected_tiers":
                        expected_tiers,

                    "detected_tiers":
                        len(
                            bands
                        ),
                },
            )

            continue

        # ----------------------------------------------------
        # Save tiers
        # ----------------------------------------------------

        for tier_index, (
            (
                tier_name,
                label,
            ),
            (
                y0,
                y1,
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
            # preserve full horizontal width
            tier_image = row_image[
                y0:y1,
                :
            ]

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
                    "Failed to save tier image:\n"
                    f"{tier_image_path}"
                )

            # Preserve raw label exactly.
            # TalimTierDataset later removes only
            # leading alignment spaces for OCR.
            tier_label_path.write_text(
                label,
                encoding="utf-8",
            )

            append_csv_row(
                metadata_path,
                TIER_FIELDNAMES,
                {
                    "tier_id":
                        tier_id,

                    "row_id":
                        row_id,

                    "parent_sample_id":
                        sample_id,

                    "row_index":
                        row_index,

                    "tier_name":
                        tier_name,

                    "tier_index":
                        tier_index,

                    "image_path":
                        (
                            "images/"
                            f"{tier_id}.png"
                        ),

                    "label_path":
                        (
                            "labels/"
                            f"{tier_id}.txt"
                        ),

                    "label":
                        label,

                    "label_length":
                        len(
                            label
                        ),

                    "image_height":
                        tier_image.shape[0],

                    "image_width":
                        tier_image.shape[1],

                    "y0":
                        y0,

                    "y1":
                        y1,
                },
            )

            tiers_created += 1

        rows_created += 1

    return (
        True,
        rows_created,
        tiers_created,
    )


# ============================================================
# BUILD ONE SPLIT
# ============================================================

def build_split(
    split,
    limit=None,
):

    if split not in {
        "train",
        "validation",
        "test",
    }:

        raise ValueError(
            "split must be train, validation, or test"
        )

    if not ZIP_PATH.exists():

        raise FileNotFoundError(
            f"ZIP not found:\n{ZIP_PATH}"
        )

    output_root = (
        OUTPUT_BASE
        / split
    )

    output_images = (
        output_root
        / "images"
    )

    output_labels = (
        output_root
        / "labels"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_images.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        output_root
        / "metadata.csv"
    )

    layout_path = (
        output_root
        / "layout_manifest.csv"
    )

    failure_path = (
        output_root
        / "tier_build_failures.csv"
    )

    completed_path = (
        output_root
        / "completed_pages.txt"
    )

    ensure_csv(
        metadata_path,
        TIER_FIELDNAMES,
    )

    ensure_csv(
        layout_path,
        LAYOUT_FIELDNAMES,
    )

    ensure_csv(
        failure_path,
        FAILURE_FIELDNAMES,
    )

    completed_pages = (
        load_completed_pages(
            completed_path
        )
    )

    print()

    print(
        "=" * 75
    )

    print(
        "NAQSHKASH - FULL TIER DATASET BUILDER"
    )

    print(
        "Split:",
        split,
    )

    print(
        "ZIP:",
        ZIP_PATH,
    )

    print(
        "Output:",
        output_root,
    )

    print(
        "Already completed:",
        len(
            completed_pages
        ),
    )

    if limit is not None:

        print(
            "Run limit:",
            limit,
        )

    print(
        "=" * 75
    )

    pages_successful = 0
    pages_failed = 0

    rows_created = 0
    tiers_created = 0

    with zipfile.ZipFile(
        ZIP_PATH,
        "r",
    ) as zip_file:

        image_members = (
            get_split_image_members(
                zip_file,
                split,
            )
        )

        print(
            "Pages available:",
            len(
                image_members
            ),
        )

        processed_this_run = 0

        for page_number, image_member in enumerate(
            image_members,
            start=1,
        ):

            sample_id = Path(
                image_member
            ).stem

            if sample_id in completed_pages:

                continue

            if (
                limit is not None
                and processed_this_run >= limit
            ):

                break

            success, row_count, tier_count = (
                process_page(
                    zip_file=zip_file,
                    split=split,
                    image_member=image_member,
                    output_root=output_root,
                    metadata_path=metadata_path,
                    layout_path=layout_path,
                    failure_path=failure_path,
                )
            )

            if success:

                pages_successful += 1

                rows_created += (
                    row_count
                )

                tiers_created += (
                    tier_count
                )

            else:

                pages_failed += 1

            # Mark attempted page complete.
            # Failure details remain in the failure CSV.
            mark_page_completed(
                completed_path,
                sample_id,
            )

            completed_pages.add(
                sample_id
            )

            processed_this_run += 1

            # Don't print every page for the 80k run.
            if (
                processed_this_run <= 10
                or processed_this_run % 100 == 0
            ):

                print(
                    f"[{processed_this_run}] "
                    f"{sample_id} | "
                    f"success={success} | "
                    f"rows={row_count} | "
                    f"tiers={tier_count}"
                )

    print()

    print(
        "=" * 75
    )

    print(
        "BUILD RUN COMPLETE"
    )

    print(
        "Split:",
        split,
    )

    print(
        "Pages attempted this run:",
        processed_this_run,
    )

    print(
        "Successful pages:",
        pages_successful,
    )

    print(
        "Failed pages:",
        pages_failed,
    )

    print(
        "Rows created this run:",
        rows_created,
    )

    print(
        "Tier samples created this run:",
        tiers_created,
    )

    print(
        "Output:",
        output_root,
    )

    print(
        "=" * 75
    )


# ============================================================
# COMMAND LINE
# ============================================================

def main():
    global ZIP_PATH, OUTPUT_BASE

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        required=True,
        choices=[
            "train",
            "validation",
            "test",
        ],
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional maximum number of pages "
            "to process in this run."
        ),
    )

    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    args = parser.parse_args()
    ZIP_PATH, OUTPUT_BASE = args.zip_path, args.output_base

    build_split(
        split=args.split,
        limit=args.limit,
    )


if __name__ == "__main__":

    main()