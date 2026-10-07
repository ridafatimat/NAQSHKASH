"""
preprocessing.pipeline
======================

Integrated NAQSHKASH preprocessing pipeline.

Team responsibilities
---------------------
Rida:
    - rotational deskewing
    - preprocessing integration

Ayesha:
    - margin cropping
    - logical row detection
    - row separation

Current pipeline
----------------
1. Deskew input document
2. Remove outer margins
3. Detect logical Talim rows
4. Separate logical rows

Later preprocessing stages can add:
- image cleaning / denoising
- ruling-line removal
- grayscale normalization
- resize to fixed model height
- tensor conversion

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


# ============================================================
# PIPELINE RESULT
# ============================================================

@dataclass
class PreprocessingResult:
    """
    Structured result returned by the integrated preprocessing pipeline.

    Coordinate note
    ---------------
    If deskewing rotates the image, cropping and row detection operate
    in DESKEWED IMAGE SPACE.

    Therefore:
        crop_bbox
        global_row_bboxes

    refer to coordinates in the deskewed image.

    The deskew affine matrix is retained so later stages can map
    coordinates back to the original input image if required.
    """

    # --------------------------------------------------------
    # Original input
    # --------------------------------------------------------

    original_shape: Tuple[int, ...]

    # --------------------------------------------------------
    # Rida - deskew information
    # --------------------------------------------------------

    deskewed_image: np.ndarray

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
) -> PreprocessingResult:
    """
    Run the integrated preprocessing pipeline.

    Pipeline
    --------
    Input image
        ↓
    Deskew
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
        Padding around the document content after margin detection.

    row_padding:
        Padding around each detected logical Talim row.

    noise_filter:
        Enable Ayesha's speckle-noise filtering during crop detection.

    deskew_config:
        Optional Rida DeskewConfig.

        If None, the normal default configuration is used.

    apply_deskew:
        Whether deskewing should run.

        Default:
            True

        This option is mainly useful for debugging or ablation tests.

    Returns
    -------
    PreprocessingResult
        Combined deskew + crop + row-separation result.
    """

    # --------------------------------------------------------
    # VALIDATE INPUT
    # --------------------------------------------------------

    if not isinstance(
        image,
        np.ndarray,
    ):

        raise TypeError(
            f"Expected numpy.ndarray, "
            f"got {type(image)}"
        )

    if image.size == 0:

        raise ValueError(
            "Cannot preprocess an empty image."
        )

    # Preserve original shape before any transformation.

    original_shape = image.shape


    # ========================================================
    # STEP 1
    # RIDA - DESKEW
    # ========================================================

    if apply_deskew:

        deskew_result = deskew(
            image,
            deskew_config,
        )

        working_image = (
            deskew_result.image
        )

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
        # pipeline runs without rotational correction.

        working_image = (
            image.copy()
        )

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
    # AYESHA - MARGIN CROPPING
    # ========================================================

    cropped_image, crop_bbox = (
        crop_margins(
            working_image,
            padding=crop_padding,
            return_bbox=True,
            noise_filter=noise_filter,
        )
    )

    crop_ymin, crop_xmin, _, _ = (
        crop_bbox
    )


    # ========================================================
    # STEP 3
    # AYESHA - LOGICAL ROW DETECTION
    # ========================================================

    row_bboxes = detect_rows(
        cropped_image,
        padding=row_padding,
    )


    # ========================================================
    # STEP 4
    # AYESHA - ROW SEPARATION
    # ========================================================

    row_data_list = separate_rows(
        cropped_image,
        row_bboxes=row_bboxes,
        padding=row_padding,
    )


    # ========================================================
    # STEP 5
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

            crop_ymin
            + row_ymin,

            crop_xmin
            + row_xmin,

            crop_ymin
            + row_ymax,

            crop_xmin
            + row_xmax,

        )

        global_bboxes.append(
            global_bbox
        )


    # ========================================================
    # RETURN STRUCTURED RESULT
    # ========================================================

    return PreprocessingResult(

        original_shape=original_shape,

        # Deskew metadata
        deskewed_image=working_image,

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