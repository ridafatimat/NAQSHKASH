"""
preprocessing.model_prep
========================
End-to-end model preprocessing pipeline producing normalized, model-ready row outputs.

Author: Ayesha Amer
Project: NAQSHKASH FYP

Design Contract:
----------------
- Integrates existing stages: deskewing -> margin cropping -> logical row separation.
- Provides an optional `clean_fn` hook (default=None) where Hareem's noise-reduction
  and ruling-line removal module can be plugged in per-row without code conflicts.
- Applies row transformations: to_grayscale -> resize to 64px height -> normalize to [0.0, 1.0].
- Returns structured `ModelPrepResult` with `ModelReadyRow` dataclasses.
- Polarity: 0.0 = dark ink (black), 1.0 = white background (paper).
- Variable width: Keeps exact aspect-ratio without horizontal padding (padding is handled in DataLoader batching).
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Callable
import numpy as np

from .pipeline import crop_and_separate_rows, PreprocessingResult
from .rows import RowData
from .normalize import preprocess_row, to_grayscale, resize_to_height, normalize


@dataclass
class ModelReadyRow:
    """
    Data structure representing a model-ready preprocessed logical Talim row.
    
    Attributes
    ----------
    row_index : int
        0-indexed vertical position from top to bottom (0 is topmost row).
    tensor_image : np.ndarray
        2D float32 normalized image of shape (64, W) with pixel values in [0.0, 1.0].
        Polarity: 0.0 = dark ink, 1.0 = light paper background.
    raw_image : np.ndarray
        Original unresized row slice from separate_rows.
    bbox : Tuple[int, int, int, int]
        Bounding box (ymin, xmin, ymax, xmax) relative to the cropped document image.
    global_bbox : Tuple[int, int, int, int]
        Bounding box (ymin, xmin, ymax, xmax) mapped back into full/deskewed image coordinates.
    height : int
        Normalized height in pixels (always 64).
    width : int
        Variable width W of the normalized row image in pixels.
    original_height : int
        Height of the raw unresized row slice in pixels.
    original_width : int
        Width of the raw unresized row slice in pixels.
    metadata : Dict[str, Any]
        Extraction and subline metadata from row separation.
    """
    row_index: int
    tensor_image: np.ndarray
    raw_image: np.ndarray
    bbox: Tuple[int, int, int, int]
    global_bbox: Tuple[int, int, int, int]
    height: int
    width: int
    original_height: int
    original_width: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPrepResult:
    """
    Structured container for complete model preparation preprocessing results.
    
    Attributes
    ----------
    original_shape : Tuple[int, ...]
        Dimensions of raw input image (H, W) or (H, W, C).
    deskewed_image : np.ndarray
        Image array after rotational deskewing (or unchanged copy if deskew disabled).
    deskew_angle : float
        Detected skew angle in degrees.
    applied_rotation : float
        Applied rotation angle in degrees.
    deskew_confidence : float
        Confidence score of skew estimation.
    deskew_corrected : bool
        Whether rotational deskew was applied.
    deskew_reason : str
        Reason string for deskew decision.
    deskew_matrix : Optional[np.ndarray]
        2x3 affine rotation matrix if rotated, else None.
    cropped_image : np.ndarray
        Document image after outer margin removal.
    crop_bbox : Tuple[int, int, int, int]
        Bounding box (ymin, xmin, ymax, xmax) of cropped content in deskewed space.
    rows : List[ModelReadyRow]
        Ordered list of model-ready row dataclasses containing normalized (64, W) float32 arrays.
    num_rows : int
        Count of detected logical rows.
    global_row_bboxes : List[Tuple[int, int, int, int]]
        Bounding boxes for each row in deskewed image space.
    raw_result : PreprocessingResult
        Underlying PreprocessingResult returned by crop_and_separate_rows.
    """
    original_shape: Tuple[int, ...]
    deskewed_image: np.ndarray
    deskew_angle: float
    applied_rotation: float
    deskew_confidence: float
    deskew_corrected: bool
    deskew_reason: str
    deskew_matrix: Optional[np.ndarray]
    cropped_image: np.ndarray
    crop_bbox: Tuple[int, int, int, int]
    rows: List[ModelReadyRow]
    num_rows: int
    global_row_bboxes: List[Tuple[int, int, int, int]]
    raw_result: PreprocessingResult


def preprocess_image(
    image: np.ndarray,
    clean_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    target_height: int = 64,
    min_width: int = 16,
    crop_padding: int = 10,
    row_padding: int = 5,
    noise_filter: bool = True,
    apply_deskew: bool = True,
    deskew_config: Optional[Any] = None,
) -> ModelPrepResult:
    """
    Execute the end-to-end NAQSHKASH model preprocessing pipeline on an input Talim document.
    
    Pipeline Order:
    ---------------
    1. Rotational Deskewing (via integrated Rida deskew in crop_and_separate_rows).
    2. Margin Cropping (via crop_margins in crop_and_separate_rows).
    3. 3-Tier Logical Row Detection & Separation (via detect_rows & separate_rows).
    4. Optional Row-level Cleaning (Hareem clean_fn hook applied to each row if provided).
    5. Grayscale Conversion, Aspect-Preserving Resize to 64px Height, and Float32 Normalization to [0.0, 1.0].
    
    Parameters
    ----------
    image : np.ndarray
        Input Talim image (2D grayscale or 3D BGR/RGB).
    clean_fn : Optional[Callable[[np.ndarray], np.ndarray]], default=None
        Optional hook for Hareem's future noise-reduction & ruling-line removal module.
        If provided, clean_fn is applied to each separated row crop before grayscale/resize/normalization.
    target_height : int, default=64
        Fixed target height in pixels for the model input row tensors.
    min_width : int, default=16
        Minimum allowable width in pixels for the normalized row images.
    crop_padding : int, default=10
        Padding added around detected document content during margin removal.
    row_padding : int, default=5
        Padding added around each logical row crop during row separation.
    noise_filter : bool, default=True
        Whether to filter isolated speckle noise during initial margin detection.
    apply_deskew : bool, default=True
        Whether rotational deskewing is applied. Passed through to crop_and_separate_rows.
    deskew_config : Optional[DeskewConfig], default=None
        Optional custom DeskewConfig passed to crop_and_separate_rows.
        
    Returns
    -------
    ModelPrepResult
        Complete container containing model-ready rows (shape (64, W), float32, [0, 1])
        and all associated bounding boxes and transformation metadata.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)}")
    if image.size == 0:
        raise ValueError("Cannot preprocess an empty image.")

    # Step 1-3: Execute existing integrated deskew -> crop -> row separation
    raw_result = crop_and_separate_rows(
        image=image,
        crop_padding=crop_padding,
        row_padding=row_padding,
        noise_filter=noise_filter,
        deskew_config=deskew_config,
        apply_deskew=apply_deskew,
    )

    model_ready_rows: List[ModelReadyRow] = []

    for idx, (raw_row, g_bbox) in enumerate(zip(raw_result.rows, raw_result.global_row_bboxes)):
        row_img = raw_row.image

        # Step 4: Optional row-level cleaning / ruling-line removal hook (Hareem's module)
        if clean_fn is not None:
            cleaned_img = clean_fn(row_img)
            if not isinstance(cleaned_img, np.ndarray):
                raise TypeError(
                    f"clean_fn must return a numpy.ndarray, got {type(cleaned_img)}"
                )
            row_working_img = cleaned_img
        else:
            row_working_img = row_img

        # Step 5: Grayscale -> Aspect-preserving resize to 64px -> Normalize to [0.0, 1.0]
        tensor_img = preprocess_row(
            row_working_img,
            target_height=target_height,
            min_width=min_width,
        )

        h, w = tensor_img.shape[:2]

        model_row = ModelReadyRow(
            row_index=idx,
            tensor_image=tensor_img,
            raw_image=raw_row.image,
            bbox=raw_row.bbox,
            global_bbox=g_bbox,
            height=h,
            width=w,
            original_height=raw_row.height,
            original_width=raw_row.width,
            metadata=dict(raw_row.metadata),
        )
        model_ready_rows.append(model_row)

    return ModelPrepResult(
        original_shape=raw_result.original_shape,
        deskewed_image=raw_result.deskewed_image,
        deskew_angle=raw_result.deskew_angle,
        applied_rotation=raw_result.applied_rotation,
        deskew_confidence=raw_result.deskew_confidence,
        deskew_corrected=raw_result.deskew_corrected,
        deskew_reason=raw_result.deskew_reason,
        deskew_matrix=raw_result.deskew_matrix,
        cropped_image=raw_result.cropped_image,
        crop_bbox=raw_result.crop_bbox,
        rows=model_ready_rows,
        num_rows=len(model_ready_rows),
        global_row_bboxes=raw_result.global_row_bboxes,
        raw_result=raw_result,
    )
