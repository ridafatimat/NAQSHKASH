"""
NAQSHKASH
Integrated preprocessing pipeline tests.

Tests:
- deskew integration
- crop integration
- row detection
- row separation
- coordinate mapping
- straight-image stability
- positive/negative skew
- ability to disable deskew
- preservation of uint8 image format

Note:
Hareem's cleaning step may slightly alter exact pixel intensities.
Therefore, tests should verify that images remain uint8 in the
0-255 range rather than requiring exact black pixels to remain 0.
"""

from pathlib import Path
import sys

import cv2
import numpy as np


# ============================================================
# PROJECT IMPORT SETUP
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


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
# SYNTHETIC TEST CONFIG
# ============================================================

# These artificial rectangular fixtures produce lower deskew
# confidence than real Talim pages even when the angle estimate
# itself is correct.
#
# We relax confidence ONLY for these integration tests.
#
# Production DeskewConfig remains unchanged.
INTEGRATION_DESKEW_CONFIG = DeskewConfig(
    min_confidence=0.05
)


# ============================================================
# SYNTHETIC TALIM DOCUMENT GENERATOR
# ============================================================

def _generate_framed_talim_document(
    num_rows=4,
    outer_margin=50,
):
    """
    Create a synthetic Talim-like document.

    Each logical row contains up to three vertical tiers:
        upper
        count
        lower

    Different row patterns are intentionally used so that
    logical row grouping can be tested.

    Returns
    -------
    np.ndarray
        uint8 grayscale image.
    """

    page_width = 350

    row_gap = 30

    top_margin = outer_margin
    bottom_margin = outer_margin

    # Different row patterns.
    patterns = [
        # upper, count, lower
        (True, True, True),
        (False, True, True),
        (True, True, False),
        (True, True, True),
    ]

    # Approximate height per logical row.
    logical_row_height = 70

    page_height = (
        top_margin
        + num_rows * logical_row_height
        + max(
            0,
            num_rows - 1,
        ) * row_gap
        + bottom_margin
    )

    image = np.full(
        (
            page_height,
            page_width,
        ),
        255,
        dtype=np.uint8,
    )

    x_start = outer_margin
    x_end = (
        page_width
        - outer_margin
    )

    current_y = (
        top_margin
    )

    for row_index in range(
        num_rows
    ):

        upper_present, count_present, lower_present = (
            patterns[
                row_index
                % len(patterns)
            ]
        )

        # ----------------------------------------
        # Upper tier
        # ----------------------------------------

        if upper_present:

            cv2.rectangle(
                image,
                (
                    x_start + 20,
                    current_y,
                ),
                (
                    x_start + 45,
                    current_y + 10,
                ),
                0,
                thickness=-1,
            )

        # ----------------------------------------
        # Count tier
        # ----------------------------------------

        if count_present:

            cv2.rectangle(
                image,
                (
                    x_start,
                    current_y + 20,
                ),
                (
                    x_end,
                    current_y + 35,
                ),
                0,
                thickness=-1,
            )

        # ----------------------------------------
        # Lower tier
        # ----------------------------------------

        if lower_present:

            cv2.rectangle(
                image,
                (
                    x_start + 10,
                    current_y + 45,
                ),
                (
                    x_start + 35,
                    current_y + 55,
                ),
                0,
                thickness=-1,
            )

        current_y += (
            logical_row_height
            + row_gap
        )

    return image


# ============================================================
# TEST 1
# END-TO-END
# ============================================================

def test_crop_and_separate_rows_end_to_end():
    """
    Verify that the complete pipeline executes
    end-to-end and returns the expected logical rows.
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

    assert (
        result.num_rows
        ==
        4
    )

    assert (
        len(
            result.rows
        )
        ==
        4
    )

    assert (
        len(
            result.global_row_bboxes
        )
        ==
        4
    )

    for row in result.rows:

        assert isinstance(
            row,
            RowData,
        )

        assert (
            row.image.size
            >
            0
        )

        assert (
            row.height
            >
            0
        )

        assert (
            row.width
            >
            0
        )


# ============================================================
# TEST 2
# GLOBAL VS LOCAL COORDINATES
# ============================================================

def test_global_vs_local_coordinate_mapping():
    """
    Verify that row bounding boxes are correctly
    mapped from cropped-image coordinates into the
    deskewed/preprocessed image coordinate system.
    """

    doc = _generate_framed_talim_document(
        num_rows=3,
        outer_margin=50,
    )

    result = crop_and_separate_rows(
        doc,
        apply_deskew=False,
        apply_cleaning=False,
        apply_ruling_line_removal=False,
    )

    (
        crop_ymin,
        crop_xmin,
        _,
        _,
    ) = result.crop_bbox

    assert (
        len(
            result.rows
        )
        ==
        len(
            result.global_row_bboxes
        )
    )

    for (
        row,
        global_bbox,
    ) in zip(
        result.rows,
        result.global_row_bboxes,
    ):

        (
            local_ymin,
            local_xmin,
            local_ymax,
            local_xmax,
        ) = row.bbox

        expected_global = (
            crop_ymin
            + local_ymin,

            crop_xmin
            + local_xmin,

            crop_ymin
            + local_ymax,

            crop_xmin
            + local_xmax,
        )

        assert (
            global_bbox
            ==
            expected_global
        )


# ============================================================
# TEST 3
# NO MODEL RESIZING / NORMALIZATION
# ============================================================

def test_pipeline_no_resizing_or_normalization():
    """
    Verify that preprocessing returns uint8 row crops
    without model resizing or model normalization.

    Hareem's cleaning step may slightly change exact
    pixel intensities, so we do NOT require the darkest
    pixel to remain exactly 0.
    """

    doc = _generate_framed_talim_document(
        num_rows=2
    )

    result = crop_and_separate_rows(
        doc
    )

    assert (
        result.num_rows
        ==
        2
    )

    for row in result.rows:

        assert isinstance(
            row,
            RowData,
        )

        # Still normal image data.
        assert (
            row.image.dtype
            ==
            np.uint8
        )

        # Must remain inside normal uint8 range.
        assert (
            row.image.min()
            >=
            0
        )

        assert (
            row.image.max()
            <=
            255
        )

        # Must not become float/model-normalized data.
        assert np.issubdtype(
            row.image.dtype,
            np.integer,
        )

        # Must remain a valid image crop.
        assert (
            row.image.shape[0]
            >
            0
        )

        assert (
            row.image.shape[1]
            >
            0
        )


# ============================================================
# TEST 4
# POSITIVE SKEW
# ============================================================

def test_pipeline_integrates_deskew_before_crop():
    """
    Verify that deskewing occurs before crop and
    logical-row separation.
    """

    doc = _generate_framed_talim_document(
        num_rows=4,
        outer_margin=60,
    )

    # Artificial +5 degree skew.
    tilted = rotate_image(
        doc,
        angle=5.0,
        expand_canvas=True,
    )

    result = crop_and_separate_rows(
        tilted,
        crop_padding=10,
        row_padding=5,
        deskew_config=(
            INTEGRATION_DESKEW_CONFIG
        ),
    )

    # ----------------------------------------
    # DESKEW
    # ----------------------------------------

    assert abs(
        result.deskew_angle
        - 5.0
    ) <= 0.5

    assert (
        result.deskew_corrected
        is True
    )

    # +5 skew requires approximately -5 correction.
    assert abs(
        result.applied_rotation
        + 5.0
    ) <= 0.5

    # ----------------------------------------
    # LOGICAL ROWS
    # ----------------------------------------

    assert (
        result.num_rows
        ==
        4
    )


# ============================================================
# TEST 5
# NEGATIVE SKEW
# ============================================================

def test_pipeline_handles_negative_skew():
    """
    Verify the pipeline when the document is skewed
    in the opposite direction.
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
        deskew_config=(
            INTEGRATION_DESKEW_CONFIG
        ),
    )

    # ----------------------------------------
    # DESKEW
    # ----------------------------------------

    assert abs(
        result.deskew_angle
        - (-6.0)
    ) <= 0.5

    assert (
        result.deskew_corrected
        is True
    )

    # -6 skew requires approximately +6 correction.
    assert abs(
        result.applied_rotation
        - 6.0
    ) <= 0.5

    # ----------------------------------------
    # ROWS
    # ----------------------------------------

    assert (
        result.num_rows
        ==
        3
    )


# ============================================================
# TEST 6
# STRAIGHT IMAGE STABILITY
# ============================================================

def test_pipeline_straight_image_not_unnecessarily_rotated():
    """
    Verify that a straight Talim document remains stable.

    Uses the normal production DeskewConfig.
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


# ============================================================
# TEST 7
# DESKEW DISABLED
# ============================================================

def test_pipeline_can_disable_deskew_for_debugging():
    """
    Verify that deskew can be explicitly disabled
    without breaking the rest of the preprocessing
    pipeline.
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

    assert (
        result.num_rows
        ==
        2
    )