"""
tests.test_preprocessing_normalize
==================================
Unit and edge-case tests for Day 3 normalization, grayscale conversion, and resizing.

Author: Ayesha Amer (Day 3 Scope)
Project: NAQSHKASH FYP
"""

import numpy as np
import pytest
import cv2

from preprocessing.normalize import (
    to_grayscale,
    resize_to_height,
    normalize,
    preprocess_row,
)


def test_to_grayscale_2d_and_3d():
    """Verify grayscale conversion from 2D grayscale, 3D BGR, and float arrays."""
    # 1. 2D grayscale
    gray_in = np.full((50, 100), 180, dtype=np.uint8)
    gray_out = to_grayscale(gray_in)
    assert gray_out.ndim == 2
    assert gray_out.shape == (50, 100)
    assert gray_out.dtype == np.uint8
    np.testing.assert_array_equal(gray_out, gray_in)

    # 2. 3D BGR color
    bgr_in = np.zeros((60, 120, 3), dtype=np.uint8)
    bgr_in[:, :, 0] = 255  # Blue channel
    gray_bgr = to_grayscale(bgr_in)
    assert gray_bgr.ndim == 2
    assert gray_bgr.shape == (60, 120)
    assert gray_bgr.dtype == np.uint8
    # Blue luminance contribution ~ 0.114 * 255 = ~29
    assert 25 <= gray_bgr[0, 0] <= 35

    # 3. Float in [0, 1]
    flt_in = np.full((40, 80), 0.5, dtype=np.float32)
    gray_flt = to_grayscale(flt_in)
    assert gray_flt.dtype == np.uint8
    assert 126 <= gray_flt[0, 0] <= 129


def test_resize_to_height_aspect_ratio():
    """Verify aspect-ratio preservation when resizing to fixed height 64."""
    # H=128, W=512 -> scale = 64/128 = 0.5 -> new_w = 256
    img = np.full((128, 512), 255, dtype=np.uint8)
    resized = resize_to_height(img, height=64)
    assert resized.shape == (64, 256)

    # H=32, W=96 -> scale = 64/32 = 2.0 -> new_w = 192
    img2 = np.full((32, 96), 255, dtype=np.uint8)
    resized2 = resize_to_height(img2, height=64)
    assert resized2.shape == (64, 192)


def test_resize_to_height_min_width_guard():
    """Verify min_width prevents degenerate/tiny widths on very tall/narrow crops."""
    # H=200, W=10 -> natural scaled width = round(10 * 64/200) = 3
    img = np.full((200, 10), 255, dtype=np.uint8)
    resized = resize_to_height(img, height=64, min_width=20)
    assert resized.shape == (64, 20)


def test_resize_to_height_stroke_and_dot_preservation():
    """Verify cv2.INTER_AREA area downscaling preserves thin strokes and isolated dots."""
    # Large canvas H=128, W=256 with background 255 (white)
    canvas = np.full((128, 256), 255, dtype=np.uint8)
    # Draw a 2x2 dot (black ink 0)
    canvas[60:62, 100:102] = 0
    # Draw a thin 1px horizontal stroke
    canvas[80, 50:150] = 0

    resized = resize_to_height(canvas, height=64)
    assert resized.shape == (64, 128)

    # The dot should not disappear; minimum pixel value in dot region must be significantly darker than 255
    dot_y = int(60 * (64 / 128))
    dot_x = int(100 * (64 / 128))
    dot_neighborhood = resized[dot_y - 2 : dot_y + 3, dot_x - 2 : dot_x + 3]
    assert dot_neighborhood.min() < 200, "Small dot vanished during resizing!"

    # The thin stroke should also be preserved
    stroke_y = int(80 * (64 / 128))
    stroke_region = resized[stroke_y - 1 : stroke_y + 2, 25:75]
    assert stroke_region.min() < 200, "Thin stroke vanished during resizing!"


def test_normalize_value_range_and_dtype():
    """Verify normalization produces float32 arrays in [0.0, 1.0] with no NaNs/Infs."""
    img_u8 = np.array([[0, 128, 255], [64, 192, 255]], dtype=np.uint8)
    norm = normalize(img_u8)

    assert norm.dtype == np.float32
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0
    assert not np.isnan(norm).any()
    assert not np.isinf(norm).any()
    assert norm[0, 0] == 0.0
    assert norm[0, 2] == 1.0
    assert abs(norm[0, 1] - (128.0 / 255.0)) < 1e-4


def test_normalize_polarity_convention():
    """
    Verify polarity convention:
    0.0 = dark ink (black)
    1.0 = white background (paper)
    """
    black_pixel = np.array([[0]], dtype=np.uint8)
    white_pixel = np.array([[255]], dtype=np.uint8)

    assert normalize(black_pixel)[0, 0] == 0.0
    assert normalize(white_pixel)[0, 0] == 1.0


def test_preprocess_row_e2e():
    """Verify full preprocess_row workflow produces (64, W) float32 array in [0.0, 1.0]."""
    # 3D BGR row crop of size 80x320
    bgr_row = np.full((80, 320, 3), 255, dtype=np.uint8)
    # Draw dark ink
    bgr_row[20:60, 40:280, :] = 0

    processed = preprocess_row(bgr_row, target_height=64, min_width=16)

    assert processed.ndim == 2
    assert processed.shape[0] == 64
    assert processed.shape[1] == 256  # 320 * (64 / 80) = 256
    assert processed.dtype == np.float32
    assert processed.min() == 0.0
    assert processed.max() == 1.0


def test_inputs_not_mutated():
    """Verify input arrays are never mutated in-place by any normalization function."""
    original = np.random.randint(0, 256, (70, 140, 3), dtype=np.uint8)
    copy_orig = original.copy()

    _ = to_grayscale(original)
    np.testing.assert_array_equal(original, copy_orig)

    gray = to_grayscale(original)
    copy_gray = gray.copy()

    _ = resize_to_height(gray, height=64)
    np.testing.assert_array_equal(gray, copy_gray)

    _ = normalize(gray)
    np.testing.assert_array_equal(gray, copy_gray)

    _ = preprocess_row(original)
    np.testing.assert_array_equal(original, copy_orig)


def test_deterministic_output():
    """Verify that multiple runs produce bitwise identical outputs."""
    img = np.random.randint(0, 256, (90, 200, 3), dtype=np.uint8)
    res1 = preprocess_row(img)
    res2 = preprocess_row(img)
    np.testing.assert_array_equal(res1, res2)


def test_invalid_inputs_raise_errors():
    """Verify proper error handling for invalid input types and empty arrays."""
    with pytest.raises(TypeError):
        to_grayscale([1, 2, 3])  # Not a numpy array

    with pytest.raises(ValueError):
        to_grayscale(np.array([]))  # Empty array

    with pytest.raises(ValueError):
        resize_to_height(np.full((50, 50), 255, dtype=np.uint8), height=0)

    with pytest.raises(ValueError):
        resize_to_height(np.full((50, 50), 255, dtype=np.uint8), min_width=-5)
