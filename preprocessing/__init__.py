"""
preprocessing
=============
Core Preprocessing Package for NAQSHKASH Kashmiri Carpet Talim OCR.

Author: Ayesha Amer (Day 1 & Day 2 Scope), Rida Fatima Tanvir (Deskewing / Integration)
Team Members: Ayesha Amer, Rida Fatima Tanvir, Hareem Sajid

Modules:
--------
- `deskew`: Rotational deskewing and skew estimation.
- `crop`: Margin removal and document bounding-box extraction.
- `rows`: 3-tier logical row detection and row separation.
- `pipeline`: Integrated end-to-end preprocessing pipeline.
- `normalize`: Row grayscale conversion, 64px aspect-preserving resizing, and [0, 1] normalization.
- `model_prep`: Model-ready row preprocessing pipeline and data containers.

Public API:
-----------
- `deskew`: Estimate and correct skew.
- `DeskewConfig`: Configuration dataclass for deskewing.
- `estimate_skew`: Skew angle estimation.
- `rotate_image`: Canvas-expanding rotation utility.
- `crop_margins`: Remove margins around text.
- `detect_rows`: Locate row bounding boxes `(ymin, xmin, ymax, xmax)`.
- `separate_rows`: Extract row crop arrays preserving original resolution & dtype.
- `crop_and_separate_rows`: End-to-end preprocessing pipeline.
- `RowData`: Dataclass holding individual row crop and metadata.
- `PreprocessingResult`: Dataclass holding full pipeline outputs.
- `to_grayscale`: Convert image arrays to 2D uint8 grayscale.
- `resize_to_height`: Fixed-height (64px) aspect-preserving resizing with cv2.INTER_AREA.
- `normalize`: Convert pixel values to float32 in [0.0, 1.0] (0.0=dark ink, 1.0=white background).
- `preprocess_row`: Chain grayscale -> resize -> normalize on a single row crop.
- `preprocess_image`: End-to-end document pipeline returning model-ready row tensors.
- `ModelReadyRow`: Dataclass containing (64, W) float32 normalized row tensor and metadata.
- `ModelPrepResult`: Dataclass containing model-ready rows and transformation metadata.
"""

from .deskew import (
    deskew,
    DeskewConfig,
    estimate_skew,
    rotate_image,
)
from .crop import (
    crop_margins,
)
from .rows import (
    detect_rows,
    separate_rows,
    RowData,
)
from .pipeline import (
    crop_and_separate_rows,
    PreprocessingResult,
)
from .normalize import (
    to_grayscale,
    resize_to_height,
    normalize,
    preprocess_row,
)
from .model_prep import (
    preprocess_image,
    ModelReadyRow,
    ModelPrepResult,
)

__all__ = [
    "deskew",
    "DeskewConfig",
    "estimate_skew",
    "rotate_image",
    "crop_margins",
    "detect_rows",
    "separate_rows",
    "RowData",
    "crop_and_separate_rows",
    "PreprocessingResult",
    "to_grayscale",
    "resize_to_height",
    "normalize",
    "preprocess_row",
    "preprocess_image",
    "ModelReadyRow",
    "ModelPrepResult",
]
