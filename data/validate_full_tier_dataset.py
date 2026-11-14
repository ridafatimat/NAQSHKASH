"""
NAQSHKASH

Final validation for the production tier-level Talim dataset.

Validates:
1. Train / validation / test metadata
2. Image and label paths
3. Duplicate tier IDs
4. Empty OCR targets
5. Metadata-label consistency
6. Parent-sample leakage across splits
7. Vocabulary derived ONLY from train
8. Unknown validation/test characters
9. CTC feasibility after resize to height 64
10. Tier distribution
11. Failure counts
12. Completed-page counts

Output:
    validation_report.json

Run:
    python -m data.validate_full_tier_dataset
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    "datasets/Final tier dataset"
)

SPLITS = [
    "train",
    "validation",
    "test",
]

EXPECTED_PAGE_COUNTS = {
    "train": 80000,
    "validation": 10000,
    "test": 10000,
}

IMAGE_HEIGHT = 64

REPORT_PATH = (
    DATASET_ROOT
    / "validation_report.json"
)

PROGRESS_INTERVAL = 50000


# ============================================================
# HELPERS
# ============================================================

def clean_ocr_target(raw_label: str) -> str:
    """
    Match TalimTierDataset behavior.

    Remove only leading alignment spaces.

    Internal spaces are preserved.
    """

    return raw_label.lstrip(" ")


def resized_width(
    original_width: int,
    original_height: int,
) -> int:
    """
    Width after aspect-preserving resize
    to IMAGE_HEIGHT.

    Matches the current CRNN input design:
        fixed height = 64
        variable width
    """

    if original_height <= 0:

        raise ValueError(
            "Image height must be positive."
        )

    scale = (
        IMAGE_HEIGHT
        / original_height
    )

    width = int(
        round(
            original_width
            * scale
        )
    )

    return max(
        1,
        width,
    )


def crnn_output_timesteps(
    input_width: int,
) -> int:
    """
    Current CRNN downsamples horizontal width
    twice by factor 2.

    Equivalent to:
        width // 2 // 2
    """

    return (
        input_width
        // 2
        // 2
    )


def ctc_required_timesteps(
    target: str,
) -> int:
    """
    CTC requires extra blank positions between
    adjacent repeated characters.

    Example:
        ABC -> 3
        AAB -> 4
        AAA -> 5
    """

    if not target:

        return 0

    adjacent_repeats = sum(
        1
        for previous, current
        in zip(
            target,
            target[1:],
        )
        if previous == current
    )

    return (
        len(target)
        + adjacent_repeats
    )


def count_nonempty_lines(
    path: Path,
) -> int:

    if not path.exists():

        return 0

    count = 0

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():

                count += 1

    return count


def count_failure_rows(
    path: Path,
) -> int:

    if not path.exists():

        return 0

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        return sum(
            1
            for _ in csv.DictReader(f)
        )


# ============================================================
# VALIDATE ONE SPLIT
# ============================================================

def validate_split(
    split: str,
):

    split_root = (
        DATASET_ROOT
        / split
    )

    metadata_path = (
        split_root
        / "metadata.csv"
    )

    completed_path = (
        split_root
        / "completed_pages.txt"
    )

    failure_path = (
        split_root
        / "tier_build_failures.csv"
    )

    print()

    print(
        "=" * 75
    )

    print(
        f"VALIDATING SPLIT: {split.upper()}"
    )

    print(
        "=" * 75
    )

    if not metadata_path.exists():

        raise FileNotFoundError(
            f"Metadata missing:\n"
            f"{metadata_path}"
        )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total_samples = 0

    missing_images = 0

    missing_labels = 0

    unreadable_labels = 0

    metadata_label_mismatches = 0

    empty_raw_labels = 0

    empty_ocr_targets = 0

    duplicate_tier_ids = 0

    invalid_dimensions = 0

    ctc_infeasible = 0

    unknown_tier_names = 0

    tier_distribution = Counter()

    raw_character_counts = Counter()

    ocr_character_counts = Counter()

    target_length_distribution = Counter()

    ctc_margin_min = None

    ctc_margin_max = None

    max_target_length = 0

    max_required_timesteps = 0

    max_input_timesteps = 0

    seen_tier_ids = set()

    parent_sample_ids = set()

    row_ids = set()

    ocr_vocabulary = set()

    ctc_failure_examples = []

    empty_target_examples = []

    missing_file_examples = []

    duplicate_examples = []

    label_mismatch_examples = []

    # --------------------------------------------------------
    # Read metadata
    # --------------------------------------------------------

    with metadata_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f
        )

        required_columns = {
            "tier_id",
            "row_id",
            "parent_sample_id",
            "tier_name",
            "image_path",
            "label_path",
            "label",
            "image_height",
            "image_width",
        }

        actual_columns = set(
            reader.fieldnames or []
        )

        missing_columns = (
            required_columns
            - actual_columns
        )

        if missing_columns:

            raise ValueError(
                f"{split}: metadata missing columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:

            total_samples += 1

            tier_id = row[
                "tier_id"
            ]

            row_id = row[
                "row_id"
            ]

            parent_sample_id = row[
                "parent_sample_id"
            ]

            tier_name = row[
                "tier_name"
            ]

            # -----------------------------------------------
            # IDs
            # -----------------------------------------------

            if tier_id in seen_tier_ids:

                duplicate_tier_ids += 1

                if len(
                    duplicate_examples
                ) < 10:

                    duplicate_examples.append(
                        tier_id
                    )

            else:

                seen_tier_ids.add(
                    tier_id
                )

            parent_sample_ids.add(
                parent_sample_id
            )

            row_ids.add(
                row_id
            )

            # -----------------------------------------------
            # Tier distribution
            # -----------------------------------------------

            tier_distribution[
                tier_name
            ] += 1

            if tier_name not in {
                "upper",
                "count",
                "lower",
            }:

                unknown_tier_names += 1

            # -----------------------------------------------
            # Paths
            # -----------------------------------------------

            image_path = (
                split_root
                / row["image_path"]
            )

            label_path = (
                split_root
                / row["label_path"]
            )

            if not image_path.exists():

                missing_images += 1

                if len(
                    missing_file_examples
                ) < 10:

                    missing_file_examples.append(
                        str(
                            image_path
                        )
                    )

            if not label_path.exists():

                missing_labels += 1

                if len(
                    missing_file_examples
                ) < 10:

                    missing_file_examples.append(
                        str(
                            label_path
                        )
                    )

                continue

            # -----------------------------------------------
            # Label
            # -----------------------------------------------

            try:

                raw_label = (
                    label_path
                    .read_text(
                        encoding="utf-8"
                    )
                    .rstrip(
                        "\r\n"
                    )
                )

            except Exception:

                unreadable_labels += 1

                continue

            metadata_label = row[
                "label"
            ]

            if (
                raw_label
                != metadata_label
            ):

                metadata_label_mismatches += 1

                if len(
                    label_mismatch_examples
                ) < 10:

                    label_mismatch_examples.append(
                        {
                            "tier_id":
                                tier_id,

                            "metadata":
                                metadata_label,

                            "file":
                                raw_label,
                        }
                    )

            if not raw_label:

                empty_raw_labels += 1

            raw_character_counts.update(
                raw_label
            )

            target = clean_ocr_target(
                raw_label
            )

            if not target:

                empty_ocr_targets += 1

                if len(
                    empty_target_examples
                ) < 10:

                    empty_target_examples.append(
                        tier_id
                    )

                continue

            # -----------------------------------------------
            # OCR vocabulary
            # -----------------------------------------------

            ocr_vocabulary.update(
                target
            )

            ocr_character_counts.update(
                target
            )

            target_length = len(
                target
            )

            target_length_distribution[
                target_length
            ] += 1

            max_target_length = max(
                max_target_length,
                target_length,
            )

            # -----------------------------------------------
            # Dimensions
            # -----------------------------------------------

            try:

                image_height = int(
                    row[
                        "image_height"
                    ]
                )

                image_width = int(
                    row[
                        "image_width"
                    ]
                )

            except (
                ValueError,
                TypeError,
            ):

                invalid_dimensions += 1

                continue

            if (
                image_height <= 0
                or image_width <= 0
            ):

                invalid_dimensions += 1

                continue

            # -----------------------------------------------
            # CTC feasibility
            # -----------------------------------------------

            training_width = resized_width(
                original_width=image_width,
                original_height=image_height,
            )

            input_timesteps = (
                crnn_output_timesteps(
                    training_width
                )
            )

            required_timesteps = (
                ctc_required_timesteps(
                    target
                )
            )

            max_required_timesteps = max(
                max_required_timesteps,
                required_timesteps,
            )

            max_input_timesteps = max(
                max_input_timesteps,
                input_timesteps,
            )

            margin = (
                input_timesteps
                - required_timesteps
            )

            if ctc_margin_min is None:

                ctc_margin_min = margin

                ctc_margin_max = margin

            else:

                ctc_margin_min = min(
                    ctc_margin_min,
                    margin,
                )

                ctc_margin_max = max(
                    ctc_margin_max,
                    margin,
                )

            if (
                input_timesteps
                < required_timesteps
            ):

                ctc_infeasible += 1

                if len(
                    ctc_failure_examples
                ) < 20:

                    ctc_failure_examples.append(
                        {
                            "tier_id":
                                tier_id,

                            "target":
                                target,

                            "original_width":
                                image_width,

                            "original_height":
                                image_height,

                            "training_width":
                                training_width,

                            "input_timesteps":
                                input_timesteps,

                            "required_timesteps":
                                required_timesteps,

                            "margin":
                                margin,
                        }
                    )

            # -----------------------------------------------
            # Progress
            # -----------------------------------------------

            if (
                total_samples
                % PROGRESS_INTERVAL
                == 0
            ):

                print(
                    f"{split}: "
                    f"{total_samples:,} "
                    f"tier samples checked"
                )

    # --------------------------------------------------------
    # Page/failure summary
    # --------------------------------------------------------

    completed_pages = (
        count_nonempty_lines(
            completed_path
        )
    )

    failure_rows = (
        count_failure_rows(
            failure_path
        )
    )

    expected_pages = (
        EXPECTED_PAGE_COUNTS[
            split
        ]
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "split":
            split,

        "expected_pages":
            expected_pages,

        "completed_pages":
            completed_pages,

        "failure_rows":
            failure_rows,

        "tier_samples":
            total_samples,

        "unique_tier_ids":
            len(
                seen_tier_ids
            ),

        "unique_rows":
            len(
                row_ids
            ),

        "unique_parent_samples":
            len(
                parent_sample_ids
            ),

        "tier_distribution":
            dict(
                tier_distribution
            ),

        "missing_images":
            missing_images,

        "missing_labels":
            missing_labels,

        "unreadable_labels":
            unreadable_labels,

        "metadata_label_mismatches":
            metadata_label_mismatches,

        "empty_raw_labels":
            empty_raw_labels,

        "empty_ocr_targets":
            empty_ocr_targets,

        "duplicate_tier_ids":
            duplicate_tier_ids,

        "unknown_tier_names":
            unknown_tier_names,

        "invalid_dimensions":
            invalid_dimensions,

        "ctc_infeasible":
            ctc_infeasible,

        "vocabulary_size_without_blank":
            len(
                ocr_vocabulary
            ),

        "vocabulary":
            sorted(
                ocr_vocabulary
            ),

        "max_target_length":
            max_target_length,

        "max_required_ctc_timesteps":
            max_required_timesteps,

        "max_available_ctc_timesteps":
            max_input_timesteps,

        "minimum_ctc_margin":
            ctc_margin_min,

        "maximum_ctc_margin":
            ctc_margin_max,

        "examples": {
            "ctc_failures":
                ctc_failure_examples,

            "empty_targets":
                empty_target_examples,

            "missing_files":
                missing_file_examples,

            "duplicate_tier_ids":
                duplicate_examples,

            "label_mismatches":
                label_mismatch_examples,
        },
    }

    print()

    print(
        f"{split.upper()} SUMMARY"
    )

    print(
        "-" * 75
    )

    print(
        f"Completed pages: "
        f"{completed_pages:,} / "
        f"{expected_pages:,}"
    )

    print(
        f"Failure rows: "
        f"{failure_rows:,}"
    )

    print(
        f"Tier samples: "
        f"{total_samples:,}"
    )

    print(
        f"Rows: "
        f"{len(row_ids):,}"
    )

    print(
        f"Parent samples: "
        f"{len(parent_sample_ids):,}"
    )

    print(
        f"Tier distribution: "
        f"{dict(tier_distribution)}"
    )

    print(
        f"Duplicate tier IDs: "
        f"{duplicate_tier_ids}"
    )

    print(
        f"Missing images: "
        f"{missing_images}"
    )

    print(
        f"Missing labels: "
        f"{missing_labels}"
    )

    print(
        f"Empty OCR targets: "
        f"{empty_ocr_targets}"
    )

    print(
        f"Label mismatches: "
        f"{metadata_label_mismatches}"
    )

    print(
        f"CTC infeasible: "
        f"{ctc_infeasible}"
    )

    print(
        f"Minimum CTC margin: "
        f"{ctc_margin_min}"
    )

    print(
        f"Vocabulary size "
        f"(without blank): "
        f"{len(ocr_vocabulary)}"
    )

    return (
        result,
        parent_sample_ids,
        seen_tier_ids,
        ocr_vocabulary,
    )


# ============================================================
# GLOBAL VALIDATION
# ============================================================

def main():

    print()

    print(
        "=" * 75
    )

    print(
        "NAQSHKASH - FINAL TIER DATASET VALIDATION"
    )

    print(
        f"Dataset: {DATASET_ROOT}"
    )

    print(
        "=" * 75
    )

    results = {}

    parents_by_split = {}

    tier_ids_by_split = {}

    vocab_by_split = {}

    # --------------------------------------------------------
    # Validate each split
    # --------------------------------------------------------

    for split in SPLITS:

        (
            result,
            parent_ids,
            tier_ids,
            vocabulary,
        ) = validate_split(
            split
        )

        results[
            split
        ] = result

        parents_by_split[
            split
        ] = parent_ids

        tier_ids_by_split[
            split
        ] = tier_ids

        vocab_by_split[
            split
        ] = vocabulary

    # ========================================================
    # SPLIT LEAKAGE
    # ========================================================

    train_validation_parent_overlap = (
        parents_by_split[
            "train"
        ]
        & parents_by_split[
            "validation"
        ]
    )

    train_test_parent_overlap = (
        parents_by_split[
            "train"
        ]
        & parents_by_split[
            "test"
        ]
    )

    validation_test_parent_overlap = (
        parents_by_split[
            "validation"
        ]
        & parents_by_split[
            "test"
        ]
    )

    train_validation_tier_overlap = (
        tier_ids_by_split[
            "train"
        ]
        & tier_ids_by_split[
            "validation"
        ]
    )

    train_test_tier_overlap = (
        tier_ids_by_split[
            "train"
        ]
        & tier_ids_by_split[
            "test"
        ]
    )

    validation_test_tier_overlap = (
        tier_ids_by_split[
            "validation"
        ]
        & tier_ids_by_split[
            "test"
        ]
    )

    # ========================================================
    # VOCABULARY
    # ========================================================

    train_vocab = (
        vocab_by_split[
            "train"
        ]
    )

    validation_unknown_chars = (
        vocab_by_split[
            "validation"
        ]
        - train_vocab
    )

    test_unknown_chars = (
        vocab_by_split[
            "test"
        ]
        - train_vocab
    )

    # Blank is index 0 later.
    ctc_num_classes = (
        len(train_vocab)
        + 1
    )

    # ========================================================
    # READINESS CONDITIONS
    # ========================================================

    critical_issues = []

    for split in SPLITS:

        result = results[
            split
        ]

        if (
            result[
                "completed_pages"
            ]
            != result[
                "expected_pages"
            ]
        ):

            critical_issues.append(
                f"{split}: completed page "
                f"count mismatch"
            )

        if result[
            "missing_images"
        ]:

            critical_issues.append(
                f"{split}: missing images"
            )

        if result[
            "missing_labels"
        ]:

            critical_issues.append(
                f"{split}: missing labels"
            )

        if result[
            "unreadable_labels"
        ]:

            critical_issues.append(
                f"{split}: unreadable labels"
            )

        if result[
            "metadata_label_mismatches"
        ]:

            critical_issues.append(
                f"{split}: metadata / "
                f"label mismatches"
            )

        if result[
            "empty_ocr_targets"
        ]:

            critical_issues.append(
                f"{split}: empty OCR targets"
            )

        if result[
            "duplicate_tier_ids"
        ]:

            critical_issues.append(
                f"{split}: duplicate tier IDs"
            )

        if result[
            "invalid_dimensions"
        ]:

            critical_issues.append(
                f"{split}: invalid dimensions"
            )

        if result[
            "ctc_infeasible"
        ]:

            critical_issues.append(
                f"{split}: CTC infeasible "
                f"samples"
            )

        if result[
            "unknown_tier_names"
        ]:

            critical_issues.append(
                f"{split}: unknown tier names"
            )

    if train_validation_parent_overlap:

        critical_issues.append(
            "Parent leakage: "
            "train <-> validation"
        )

    if train_test_parent_overlap:

        critical_issues.append(
            "Parent leakage: "
            "train <-> test"
        )

    if validation_test_parent_overlap:

        critical_issues.append(
            "Parent leakage: "
            "validation <-> test"
        )

    if train_validation_tier_overlap:

        critical_issues.append(
            "Tier ID overlap: "
            "train <-> validation"
        )

    if train_test_tier_overlap:

        critical_issues.append(
            "Tier ID overlap: "
            "train <-> test"
        )

    if validation_test_tier_overlap:

        critical_issues.append(
            "Tier ID overlap: "
            "validation <-> test"
        )

    if validation_unknown_chars:

        critical_issues.append(
            "Validation contains characters "
            "not present in train vocabulary"
        )

    if test_unknown_chars:

        critical_issues.append(
            "Test contains characters "
            "not present in train vocabulary"
        )

    ready_for_training = (
        len(
            critical_issues
        )
        == 0
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    total_tiers = sum(
        results[
            split
        ][
            "tier_samples"
        ]
        for split in SPLITS
    )

    global_report = {
        "dataset_root":
            str(
                DATASET_ROOT
            ),

        "image_height":
            IMAGE_HEIGHT,

        "splits":
            results,

        "global": {
            "total_tier_samples":
                total_tiers,

            "train_validation_parent_overlap":
                len(
                    train_validation_parent_overlap
                ),

            "train_test_parent_overlap":
                len(
                    train_test_parent_overlap
                ),

            "validation_test_parent_overlap":
                len(
                    validation_test_parent_overlap
                ),

            "train_validation_tier_overlap":
                len(
                    train_validation_tier_overlap
                ),

            "train_test_tier_overlap":
                len(
                    train_test_tier_overlap
                ),

            "validation_test_tier_overlap":
                len(
                    validation_test_tier_overlap
                ),

            "train_vocabulary":
                sorted(
                    train_vocab
                ),

            "train_vocabulary_size_without_blank":
                len(
                    train_vocab
                ),

            "ctc_blank_index":
                0,

            "ctc_num_classes":
                ctc_num_classes,

            "validation_unknown_characters":
                sorted(
                    validation_unknown_chars
                ),

            "test_unknown_characters":
                sorted(
                    test_unknown_chars
                ),

            "critical_issues":
                critical_issues,

            "ready_for_training":
                ready_for_training,
        },
    }

    REPORT_PATH.write_text(
        json.dumps(
            global_report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # PRINT FINAL SUMMARY
    # ========================================================

    print()

    print(
        "=" * 75
    )

    print(
        "GLOBAL VALIDATION SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"Total tier samples: "
        f"{total_tiers:,}"
    )

    print()

    print(
        "Parent overlap:"
    )

    print(
        "  train <-> validation:",
        len(
            train_validation_parent_overlap
        ),
    )

    print(
        "  train <-> test:",
        len(
            train_test_parent_overlap
        ),
    )

    print(
        "  validation <-> test:",
        len(
            validation_test_parent_overlap
        ),
    )

    print()

    print(
        "Tier-ID overlap:"
    )

    print(
        "  train <-> validation:",
        len(
            train_validation_tier_overlap
        ),
    )

    print(
        "  train <-> test:",
        len(
            train_test_tier_overlap
        ),
    )

    print(
        "  validation <-> test:",
        len(
            validation_test_tier_overlap
        ),
    )

    print()

    print(
        f"Train vocabulary size "
        f"(without CTC blank): "
        f"{len(train_vocab)}"
    )

    print(
        f"CRNN output classes "
        f"(with blank): "
        f"{ctc_num_classes}"
    )

    print(
        "Validation unknown chars:",
        sorted(
            validation_unknown_chars
        ),
    )

    print(
        "Test unknown chars:",
        sorted(
            test_unknown_chars
        ),
    )

    print()

    if ready_for_training:

        print(
            "RESULT: PASS"
        )

        print(
            "STATUS: DATASET READY "
            "FOR CLEAN-SYNTHETIC "
            "CRNN TRAINING"
        )

    else:

        print(
            "RESULT: FAIL"
        )

        print(
            "Critical issues:"
        )

        for issue in critical_issues:

            print(
                " -",
                issue,
            )

    print()

    print(
        "Full report:"
    )

    print(
        REPORT_PATH
    )

    print(
        "=" * 75
    )


if __name__ == "__main__":

    main()