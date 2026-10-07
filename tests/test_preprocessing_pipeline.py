"""
tests.test_preprocessing_pipeline
=================================

Integration tests for the NAQSHKASH preprocessing pipeline.

Ayesha Day 1-2:
    - margin cropping
    - logical row detection
    - row separation

Rida Day 3:
    - deskew integration before cropping
    - positive/negative rotational integration tests
    - straight-image stability
    - deskew-disable debugging mode

Project: NAQSHKASH FYP
"""

import numpy as np
import pytest

from preprocessing.pipeline import (
    crop_and_separate_rows,
    PreprocessingResult,
)

from preprocessing.rows import (
    RowData,
)

from preprocessing.deskew import (
    rotate_image,
    DeskewConfig,
)


# ============================================================
# SYNTHETIC TALIM DOCUMENT HELPER
# ============================================================

def _generate_framed_talim_document(
    num_rows: int = 4,
    outer_margin: int = 50,
) -> np.ndarray:
    """
    Generate a simple synthetic multi-row Talim-style document.

    Each logical row contains three horizontal sub-lines:
        upper tier
        count tier
        lower tier

    This helper is intentionally simple because its purpose is
    integration testing rather than OCR-recognition realism.
    """

    row_height = 80
    row_gap = 20

    content_height = (
        num_rows * row_height
        + (num_rows - 1) * row_gap
    )

    content_width = 250

    total_height = (
        content_height
        + 2 * outer_margin
    )

    total_width = (
        content_width
        + 2 * outer_margin
    )

    canvas = np.full(
        (
            total_height,
            total_width,
        ),
        255,
        dtype=np.uint8,
    )

    y = outer_margin

    for _ in range(num_rows):

        # Upper tier
        canvas[
            y : y + 25,
            outer_margin + 20 :
            total_width - outer_margin - 20
        ] = 0

        # Count tier
        canvas[
            y + 30 : y + 55,
            outer_margin + 10 :
            total_width - outer_margin - 10
        ] = 0

        # Lower tier
        canvas[
            y + 60 : y + 80,
            outer_margin + 30 :
            total_width - outer_margin - 30
        ] = 0

        y += (
            row_height
            + row_gap
        )

    return canvas


# ============================================================
# TEST-SPECIFIC DESKEW CONFIGURATION
# ============================================================

# IMPORTANT:
#
# Ayesha's integration fixture consists of large rectangular
# bars rather than real Talim symbols.
#
# Rida's production deskew configuration intentionally uses a
# stricter confidence threshold.
#
# We lower confidence ONLY for these synthetic integration
# fixtures so that we can verify:
#
#     deskew
#       ->
#     crop
#       ->
#     row detection
#       ->
#     row separation
#
# The production deskew configuration remains unchanged.

INTEGRATION_DESKEW_CONFIG = DeskewConfig(
    min_confidence=0.20
)


# ============================================================
# AYESHA ORIGINAL PIPELINE TESTS
# ============================================================

def test_crop_and_separate_rows_end_to_end():
    """
    Verify that the crop-and-row pipeline executes end-to-end
    and returns structured results.
    """

    doc = _generate_framed_talim_document(
        num_rows=4,
        outer_margin=60,
    )

    result = crop_and_separate_rows(
        doc,
        crop_padding=10,
        row_padding=5,
    )

    assert isinstance(
        result,
        PreprocessingResult,
    )

    assert result.num_rows == 4

    assert len(
        result.rows
    ) == 4

    assert len(
        result.global_row_bboxes
    ) == 4

    # Cropped image should be smaller because
    # large outer margins were removed.

    assert (
        result.cropped_image.shape[0]
        <
        doc.shape[0]
    )

    assert (
        result.cropped_image.shape[1]
        <
        doc.shape[1]
    )


def test_global_vs_local_coordinate_mapping():
    """
    Verify that row bounding boxes map correctly into the
    deskewed/cropped image coordinate system.
    """

    doc = _generate_framed_talim_document(
        num_rows=3,
        outer_margin=40,
    )

    result = crop_and_separate_rows(
        doc,
        crop_padding=10,
        row_padding=5,
    )

    (
        crop_ymin,
        crop_xmin,
        _,
        _,
    ) = result.crop_bbox

    for row_obj, global_bbox in zip(
        result.rows,
        result.global_row_bboxes,
    ):

        (
            row_ymin,
            row_xmin,
            row_ymax,
            row_xmax,
        ) = row_obj.bbox

        expected_global_bbox = (

            crop_ymin
            + row_ymin,

            crop_xmin
            + row_xmin,

            crop_ymin
            + row_ymax,

            crop_xmin
            + row_xmax,

        )

        assert (
            global_bbox
            ==
            expected_global_bbox
        )


def test_pipeline_no_resizing_or_normalization():
    """
    Verify that preprocessing currently returns raw uint8 row
    crops without model resizing or normalization.
    """

    doc = _generate_framed_talim_document(
        num_rows=2
    )

    result = crop_and_separate_rows(
        doc
    )

    for row in result.rows:

        assert isinstance(
            row,
            RowData,
        )

        assert (
            row.image.dtype
            ==
            np.uint8
        )

        assert (
            row.image.min()
            ==
            0
        )

        assert (
            row.image.max()
            ==
            255
        )

        assert (
            row.height
            ==
            row.image.shape[0]
        )


# ============================================================
# RIDA DAY 3 - DESKEW INTEGRATION TESTS
# ============================================================

def test_pipeline_integrates_deskew_before_crop():
    """
    Verify that deskewing occurs before margin cropping
    and logical row separation.
    """

    # Known four-row synthetic document.
    doc = _generate_framed_talim_document(
        num_rows=4,
        outer_margin=60,
    )

    # Apply known +5 degree skew.
    tilted = rotate_image(
        doc,
        angle=5.0,
        expand_canvas=True,
    )

    result = crop_and_separate_rows(
        tilted,
        crop_padding=10,
        row_padding=5,
        deskew_config=INTEGRATION_DESKEW_CONFIG,
    )

    # --------------------------------------------------------
    # DESKEW CHECK
    # --------------------------------------------------------

    assert abs(
        result.deskew_angle
        - 5.0
    ) <= 0.5

    assert (
        result.deskew_corrected
        is True
    )

    # +5 degree input skew should require
    # approximately -5 degree correction.

    assert abs(
        result.applied_rotation
        + 5.0
    ) <= 0.5

    # --------------------------------------------------------
    # INTEGRATION CHECK
    # --------------------------------------------------------

    assert (
        result.num_rows
        ==
        4
    )

    assert len(
        result.rows
    ) == 4


def test_pipeline_handles_negative_skew():
    """
    Verify integration when the document is rotated in the
    opposite direction.
    """

    doc = _generate_framed_talim_document(
        num_rows=3,
        outer_margin=50,
    )

    tilted = rotate_image(
        doc,
        angle=-6.0,
        expand_canvas=True,
    )

    result = crop_and_separate_rows(
        tilted,
        crop_padding=10,
        row_padding=5,
        deskew_config=INTEGRATION_DESKEW_CONFIG,
    )

    # --------------------------------------------------------
    # DESKEW CHECK
    # --------------------------------------------------------

    assert abs(
        result.deskew_angle
        - (-6.0)
    ) <= 0.5

    assert (
        result.deskew_corrected
        is True
    )

    # -6 degree input should require
    # approximately +6 degree correction.

    assert abs(
        result.applied_rotation
        - 6.0
    ) <= 0.5

    # --------------------------------------------------------
    # ROW CHECK
    # --------------------------------------------------------

    assert (
        result.num_rows
        ==
        3
    )

    assert len(
        result.rows
    ) == 3


def test_pipeline_straight_image_not_unnecessarily_rotated():
    """
    Verify that an already-straight document remains stable.

    This test intentionally uses the normal production
    DeskewConfig rather than the relaxed synthetic-test config.
    """

    doc = _generate_framed_talim_document(
        num_rows=4,
        outer_margin=60,
    )

    result = crop_and_separate_rows(
        doc
    )

    assert abs(
        result.deskew_angle
    ) < 0.5

    assert (
        result.applied_rotation
        ==
        0.0
    )

    assert (
        result.num_rows
        ==
        4
    )


def test_pipeline_can_disable_deskew_for_debugging():
    """
    Verify that deskew can be disabled explicitly for
    debugging / ablation tests.
    """

    doc = _generate_framed_talim_document(
        num_rows=2
    )

    result = crop_and_separate_rows(
        doc,
        apply_deskew=False,
    )

    assert (
        result.deskew_reason
        ==
        "DESKEW_DISABLED"
    )

    assert (
        result.deskew_corrected
        is False
    )

    assert (
        result.applied_rotation
        ==
        0.0
    )

    # Rest of pipeline should still function.
    assert (
        result.num_rows
        ==
        2
    )