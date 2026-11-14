"""
tests.test_preprocessing_model_prep
====================================
Integration tests for the model preparation preprocessing pipeline (`preprocessing.model_prep`).

Author: Ayesha Amer
Project: NAQSHKASH FYP
"""

import numpy as np
import pytest

from preprocessing.model_prep import (
    preprocess_image,
    ModelPrepResult,
    ModelReadyRow,
)
from preprocessing.pipeline import PreprocessingResult


def _generate_synthetic_document(num_rows: int = 3) -> np.ndarray:
    """Helper to generate a framed multi-row synthetic document."""
    row_height = 80
    row_gap = 25
    margin = 40
    content_w = 300
    content_h = num_rows * row_height + (num_rows - 1) * row_gap
    total_h = content_h + 2 * margin
    total_w = content_w + 2 * margin

    canvas = np.full((total_h, total_w), 255, dtype=np.uint8)

    y = margin
    for _ in range(num_rows):
        # Upper tier
        canvas[y : y + 20, margin + 30 : total_w - margin - 30] = 0
        # Count tier
        canvas[y + 25 : y + 55, margin + 10 : total_w - margin - 10] = 0
        # Lower tier
        canvas[y + 60 : y + 80, margin + 40 : total_w - margin - 40] = 0
        y += row_height + row_gap

    return canvas


def test_preprocess_image_full_pipeline():
    """Verify that preprocess_image executes end-to-end and returns ModelReadyRows of height 64."""
    doc = _generate_synthetic_document(num_rows=3)
    result = preprocess_image(doc, apply_deskew=False)

    assert isinstance(result, ModelPrepResult)
    assert result.num_rows == 3
    assert len(result.rows) == 3
    assert isinstance(result.raw_result, PreprocessingResult)

    for idx, row in enumerate(result.rows):
        assert isinstance(row, ModelReadyRow)
        assert row.row_index == idx
        # Shape: height must be strictly 64
        assert row.tensor_image.shape[0] == 64
        assert row.height == 64
        assert row.width == row.tensor_image.shape[1]
        assert row.width >= 16
        # Data type and value range
        assert row.tensor_image.dtype == np.float32
        assert row.tensor_image.min() >= 0.0
        assert row.tensor_image.max() <= 1.0
        assert not np.isnan(row.tensor_image).any()
        # Bounding boxes
        assert len(row.bbox) == 4
        assert len(row.global_bbox) == 4


def test_preprocess_image_clean_fn_hook():
    """Verify that the optional clean_fn hook (Hareem's stage) is executed per row."""
    doc = _generate_synthetic_document(num_rows=2)

    called_count = 0

    def mock_clean_fn(row_crop: np.ndarray) -> np.ndarray:
        nonlocal called_count
        called_count += 1
        cleaned = row_crop.copy()
        cleaned[0:2, 0:2] = 255
        return cleaned

    result = preprocess_image(doc, clean_fn=mock_clean_fn, apply_deskew=False)

    assert called_count == 2
    assert result.num_rows == 2


def test_preprocess_image_input_immutability():
    """Verify that the input document array is never modified in-place."""
    doc = _generate_synthetic_document(num_rows=2)
    doc_copy = doc.copy()

    _ = preprocess_image(doc)
    np.testing.assert_array_equal(doc, doc_copy)
