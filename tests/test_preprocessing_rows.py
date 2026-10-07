"""
tests.test_preprocessing_rows
=============================
Unit and edge-case tests for the row detection and separation module (`preprocessing.rows`).

Author: Ayesha Amer (Day 2 Scope)
Project: NAQSHKASH FYP
"""

import numpy as np
import pytest
from scipy import ndimage
from preprocessing.rows import detect_rows, separate_rows, RowData


def _generate_synthetic_talim_canvas(
    num_rows: int = 3,
    sublines_per_row: int = 3,
    line_height: int = 30,
    line_gap: int = 8,
    row_gap: int = 25,
    canvas_width: int = 200,
    channels: int = 1,
) -> np.ndarray:
    """
    Generate synthetic Talim canvas with multi-tier rows.
    In Talim: each logical row has up to 3 sublines (Upper, Count, Lower)
    separated by line_gap, and rows separated by larger row_gap.
    """
    row_block_height = sublines_per_row * line_height + (sublines_per_row - 1) * line_gap
    total_height = num_rows * row_block_height + (num_rows - 1) * row_gap + 40  # 20px outer margin

    if channels == 1:
        canvas = np.full((total_height, canvas_width), 255, dtype=np.uint8)
    else:
        canvas = np.full((total_height, canvas_width, channels), 255, dtype=np.uint8)

    y = 20
    for r in range(num_rows):
        for line_idx in range(sublines_per_row):
            # Center tier (count) is wider, upper/lower may be slightly shorter
            if line_idx == 1 or sublines_per_row == 1:
                xmin, xmax = 30, canvas_width - 30
            else:
                xmin, xmax = 50, canvas_width - 50

            # Draw ink line
            if channels == 1:
                canvas[y : y + line_height, xmin:xmax] = 0
            else:
                canvas[y : y + line_height, xmin:xmax, :] = 0

            y += line_height + line_gap
        y = y - line_gap + row_gap

    return canvas


def test_detect_rows_multi_tier_block():
    """Verify that multi-tier (3-line) rows are grouped as 1 logical row per block."""
    num_expected_rows = 4
    canvas = _generate_synthetic_talim_canvas(
        num_rows=num_expected_rows,
        sublines_per_row=3,
        line_height=30,
        line_gap=8,
        row_gap=30,
    )

    bboxes = detect_rows(canvas, padding=5)
    assert len(bboxes) == num_expected_rows, (
        f"Expected {num_expected_rows} logical rows, but detected {len(bboxes)}"
    )


def test_detect_rows_single_symbol_row():
    """Verify that a canvas with only 1 row is detected as exactly 1 row."""
    canvas = _generate_synthetic_talim_canvas(
        num_rows=1,
        sublines_per_row=1,
        line_height=35,
    )
    bboxes = detect_rows(canvas, padding=5)
    assert len(bboxes) == 1


def test_detect_rows_top_to_bottom_ordering():
    """Verify that detected row bounding boxes are strictly ordered from top to bottom."""
    canvas = _generate_synthetic_talim_canvas(num_rows=5, sublines_per_row=3)
    bboxes = detect_rows(canvas, padding=5)

    assert len(bboxes) == 5
    for i in range(len(bboxes) - 1):
        ymin_curr = bboxes[i][0]
        ymin_next = bboxes[i + 1][0]
        assert ymin_curr < ymin_next, (
            f"Row {i} (ymin={ymin_curr}) is not above Row {i+1} (ymin={ymin_next})"
        )


def test_detect_rows_tight_spacing():
    """Verify row detection when row_gap is close to line_gap (tight spacing)."""
    canvas = _generate_synthetic_talim_canvas(
        num_rows=3,
        sublines_per_row=3,
        line_height=25,
        line_gap=6,
        row_gap=16,
    )
    bboxes = detect_rows(canvas, padding=2)
    assert len(bboxes) == 3


def test_detect_rows_with_ruling_lines():
    """Verify row detection when horizontal ruling lines exist in the background."""
    canvas = _generate_synthetic_talim_canvas(num_rows=3, sublines_per_row=3)
    # Add thin 1px ruling lines between rows
    canvas[100, :] = 180
    canvas[200, :] = 180

    bboxes = detect_rows(canvas, padding=5)
    assert len(bboxes) == 3


def test_detect_rows_slight_tilt():
    """Verify row detection on an image with a slight 1-degree rotation."""
    canvas = _generate_synthetic_talim_canvas(num_rows=3, sublines_per_row=3)
    # Rotate by 1.2 degrees with white background fill
    tilted = ndimage.rotate(canvas, angle=1.2, reshape=False, cval=255)
    tilted = (tilted > 128).astype(np.uint8) * 255

    bboxes = detect_rows(tilted, padding=5)
    assert len(bboxes) == 3


def test_separate_rows_dtype_and_channels_preserved():
    """Verify separate_rows produces RowData preserving exact channels, dtype, and slice integrity."""
    canvas_rgb = _generate_synthetic_talim_canvas(num_rows=3, channels=3)
    row_objects = separate_rows(canvas_rgb, padding=5)

    assert len(row_objects) == 3
    for idx, row in enumerate(row_objects):
        assert isinstance(row, RowData)
        assert row.row_index == idx
        assert row.image.ndim == 3
        assert row.image.shape[2] == 3
        assert row.image.dtype == np.uint8
        assert row.height == row.image.shape[0]
        assert row.width == row.image.shape[1]


def test_separate_rows_input_not_mutated():
    """Verify separate_rows does not mutate the input array."""
    canvas = _generate_synthetic_talim_canvas(num_rows=2, channels=1)
    canvas_copy = canvas.copy()

    _ = separate_rows(canvas)
    np.testing.assert_array_equal(canvas, canvas_copy)
