
"""
preprocessing.pipeline
======================

Integrated NAQSHKASH preprocessing pipeline.

Team responsibilities
---------------------
Rida:
    - rotational deskewing
    - preprocessing integration

Hareem:
    - image cleaning / noise reduction
    - ruling-line removal

Ayesha:
    - margin cropping
    - logical row detection
    - row separation

Current preprocessing flow
--------------------------
1. Deskew input document
2. Clean / denoise image
3. Remove ruling lines
4. Remove outer margins
5. Detect logical Talim rows
6. Separate logical rows

Project: NAQSHKASH FYP
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

from .deskew import (
    deskew,
    DeskewConfig,
)

from .crop import (
    crop_margins,
)

from .rows import (
    detect_rows,
    separate_rows,
    RowData,
)

from .cleaning import (
    clean_image,
)

from .ruling_lines import (
    remove_ruling_lines,
)


# ============================================================
# PIPELINE RESULT
# ============================================================

@dataclass
class PreprocessingResult:
    """
    Structured result returned by the integrated preprocessing
    pipeline.

    Coordinate note
    ---------------
    If deskewing rotates the image, cropping and row detection
    operate in DESKEWED IMAGE SPACE.

    Therefore:

        crop_bbox
        global_row_bboxes

    refer to coordinates in the deskewed image.

    The deskew affine matrix is retained so that later stages
    can map coordinates back to the original input image if
    required.
    """

    # --------------------------------------------------------
    # Original input
    # --------------------------------------------------------

    original_shape: Tuple[int, ...]

    # --------------------------------------------------------
    # Preprocessed / deskewed image
    # --------------------------------------------------------

    deskewed_image: np.ndarray

    # --------------------------------------------------------
    # Rida - deskew information
    # --------------------------------------------------------

    deskew_angle: float
    applied_rotation: float
    deskew_confidence: float
    deskew_corrected: bool
    deskew_reason: str
    deskew_matrix: Optional[np.ndarray]

    # --------------------------------------------------------
    # Ayesha - crop information
    # --------------------------------------------------------

    cropped_image: np.ndarray

    crop_bbox: Tuple[
        int,
        int,
        int,
        int,
    ]

    # --------------------------------------------------------
    # Ayesha - row information
    # --------------------------------------------------------

    rows: List[RowData]

    num_rows: int

    global_row_bboxes: List[
        Tuple[
            int,
            int,
            int,
            int,
        ]
    ] = field(
        default_factory=list
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def crop_and_separate_rows(
    image: np.ndarray,
    crop_padding: int = 10,
    row_padding: int = 5,
    noise_filter: bool = True,
    deskew_config: Optional[
        DeskewConfig
    ] = None,
    apply_deskew: bool = True,
    apply_cleaning: bool = True,
    apply_ruling_line_removal: bool = True,
) -> PreprocessingResult:
    """
    Run the integrated NAQSHKASH preprocessing pipeline.

    Pipeline
    --------
    Input image
        ↓
    Deskew
        ↓
    Image cleaning / denoising
        ↓
    Ruling-line removal
        ↓
    Margin cropping
        ↓
    Logical row detection
        ↓
    Row separation

    Parameters
    ----------
    image:
        Input Talim image as a NumPy array.

        Supported:
            grayscale: H x W
            colour:    H x W x C

    crop_padding:
        Padding around the document content after margin
        detection.

    row_padding:
        Padding around each detected logical Talim row.

    noise_filter:
        Enable Ayesha's speckle-noise filtering during
        crop detection.

    deskew_config:
        Optional Rida DeskewConfig.

        If None, the normal default configuration is used.

    apply_deskew:
        Whether rotational deskewing should run.

        Default:
            True

        Mainly useful for debugging and ablation tests.

    apply_cleaning:
        Whether Hareem's image cleaning / denoising stage
        should run.

        Default:
            True

    apply_ruling_line_removal:
        Whether Hareem's ruling-line removal stage should run.

        Default:
            True

    Returns
    -------
    PreprocessingResult
        Combined result containing:

        - preprocessing image
        - deskew metadata
        - crop information
        - detected rows
        - global row bounding boxes
    """

    # ========================================================
    # STEP 0
    # VALIDATE INPUT
    # ========================================================

    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Expected numpy.ndarray, got {type(image)}"
        )

    if image.size == 0:
        raise ValueError(
            "Cannot preprocess an empty image."
        )

    if image.ndim not in (2, 3):
        raise ValueError(
            f"Unsupported image shape: {image.shape}. "
            "Expected a grayscale or colour image."
        )

    # Preserve original shape before any transformation.

    original_shape = image.shape

    # Always work on our own copy so that the caller's
    # original image is never modified.

    working_image = image.copy()

    # ========================================================
    # STEP 1
    # RIDA - DESKEW
    # ========================================================

    if apply_deskew:

        deskew_result = deskew(
            working_image,
            deskew_config,
        )

        working_image = deskew_result.image

        deskew_angle = (
            deskew_result.skew_angle
        )

        applied_rotation = (
            deskew_result.applied_rotation
        )

        deskew_confidence = (
            deskew_result.confidence
        )

        deskew_corrected = (
            deskew_result.corrected
        )

        deskew_reason = (
            deskew_result.reason
        )

        deskew_matrix = (
            deskew_result.matrix
        )

    else:

        # Debug / ablation mode:
        # pipeline continues without rotational correction.

        deskew_angle = 0.0

        applied_rotation = 0.0

        deskew_confidence = 0.0

        deskew_corrected = False

        deskew_reason = (
            "DESKEW_DISABLED"
        )

        deskew_matrix = None

    # ========================================================
    # STEP 2
    # HAREEM - IMAGE CLEANING / DENOISING
    # ========================================================

    if apply_cleaning:

        working_image = clean_image(
            working_image
        )

    # ========================================================
    # STEP 3
    # HAREEM - RULING-LINE REMOVAL
    # ========================================================

    if apply_ruling_line_removal:

        working_image = remove_ruling_lines(
            working_image
        )

    # ========================================================
    # STEP 4
    # AYESHA - MARGIN CROPPING
    # ========================================================

    cropped_image, crop_bbox = crop_margins(
        working_image,
        padding=crop_padding,
        return_bbox=True,
        noise_filter=noise_filter,
    )

    (
        crop_ymin,
        crop_xmin,
        _,
        _,
    ) = crop_bbox

    # ========================================================
    # STEP 5
    # AYESHA - LOGICAL ROW DETECTION
    # ========================================================

    row_bboxes = detect_rows(
        cropped_image,
        padding=row_padding,
    )

    # ========================================================
    # STEP 6
    # AYESHA - ROW SEPARATION
    # ========================================================

    row_data_list = separate_rows(
        cropped_image,
        row_bboxes=row_bboxes,
        padding=row_padding,
    )

    # ========================================================
    # STEP 7
    # MAP LOCAL ROW BOXES INTO DESKEWED IMAGE SPACE
    # ========================================================

    global_bboxes = []

    for row_bbox in row_bboxes:

        (
            row_ymin,
            row_xmin,
            row_ymax,
            row_xmax,
        ) = row_bbox

        global_bbox = (
            crop_ymin + row_ymin,
            crop_xmin + row_xmin,
            crop_ymin + row_ymax,
            crop_xmin + row_xmax,
        )

        global_bboxes.append(
            global_bbox
        )

    # ========================================================
    # RETURN STRUCTURED RESULT
    # ========================================================

    return PreprocessingResult(

        # Original input
        original_shape=original_shape,

        # Final image after deskew + Hareem preprocessing
        deskewed_image=working_image,

        # Deskew metadata
        deskew_angle=deskew_angle,

        applied_rotation=applied_rotation,

        deskew_confidence=deskew_confidence,

        deskew_corrected=deskew_corrected,

        deskew_reason=deskew_reason,

        deskew_matrix=deskew_matrix,

        # Crop
        cropped_image=cropped_image,

        crop_bbox=crop_bbox,

        # Rows
        rows=row_data_list,

        num_rows=len(
            row_data_list
        ),

        global_row_bboxes=global_bboxes,
    )

