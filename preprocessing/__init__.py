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
]
