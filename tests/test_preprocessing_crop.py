"""
tests.test_preprocessing_crop
=============================
Unit and edge-case tests for the margin cropping module (`preprocessing.crop`).

Author: Ayesha Amer (Day 1 Scope)
Project: NAQSHKASH FYP
"""

import numpy as np
import pytest
from preprocessing.crop import crop_margins, _binarize_ink


def _create_synthetic_doc(
    height: int = 300,
    width: int = 400,
    margin_top: int = 40,
    margin_bottom: int = 50,
    margin_left: int = 60,
    margin_right: int = 70,
    channels: int = 1,
    bg_val: int = 255,
    ink_val: int = 0,
) -> np.ndarray:
    """Helper to generate synthetic document image with specified margins and text blocks."""
    if channels == 1:
        img = np.full((height, width), bg_val, dtype=np.uint8)
    else:
        img = np.full((height, width, channels), bg_val, dtype=np.uint8)

    # Draw synthetic text blocks
    text_ymin = margin_top
    text_ymax = height - margin_bottom
    text_xmin = margin_left
    text_xmax = width - margin_right

    # Draw multiple horizontal stroke bars mimicking text
    for y in range(text_ymin, text_ymax, 20):
        if channels == 1:
            img[y:y+8, text_xmin:text_xmax] = ink_val
        else:
            img[y:y+8, text_xmin:text_xmax, :] = ink_val

    return img


def test_crop_margins_basic_grayscale():
    """Verify margin removal on standard grayscale image with exact padding."""
    img = _create_synthetic_doc(
        height=200, width=300,
        margin_top=30, margin_bottom=40,
        margin_left=50, margin_right=60,
        channels=1,
    )
    padding = 5
    cropped, bbox = crop_margins(img, padding=padding, return_bbox=True)

    ymin, xmin, ymax, xmax = bbox
    assert ymin == 30 - padding
    assert xmin == 50 - padding
    assert ymax <= 200 - 40 + padding + 8
    assert xmax == 300 - 60 + padding

    assert cropped.shape[0] == ymax - ymin
    assert cropped.shape[1] == xmax - xmin
    assert cropped.dtype == np.uint8


def test_crop_margins_bgr_color():
    """Verify 3-channel BGR/RGB images preserve 3D shape and channels."""
    img = _create_synthetic_doc(channels=3)
    cropped, bbox = crop_margins(img, padding=10, return_bbox=True)

    assert cropped.ndim == 3
    assert cropped.shape[2] == 3
    assert cropped.dtype == np.uint8
    assert len(bbox) == 4


def test_crop_margins_input_not_mutated():
    """Verify that crop_margins strictly preserves input immutability."""
    img = _create_synthetic_doc(channels=3)
    img_copy = img.copy()

    _ = crop_margins(img, padding=10, return_bbox=True)

    np.testing.assert_array_equal(img, img_copy, err_msg="Input image was mutated in-place!")


def test_crop_margins_blank_image_fallback():
    """Verify safe fallback on blank uniform image (returns full image and bounds)."""
    blank = np.full((150, 200), 255, dtype=np.uint8)
    cropped, bbox = crop_margins(blank, padding=10, return_bbox=True)

    assert bbox == (0, 0, 150, 200)
    assert cropped.shape == (150, 200)
    np.testing.assert_array_equal(cropped, blank)


def test_crop_margins_dark_borders():
    """Verify that dark scan borders / shadows do not prevent content extraction."""
    img = _create_synthetic_doc(height=200, width=200, margin_top=40, margin_bottom=40, margin_left=40, margin_right=40)
    # Add dark vignette/border at the extreme 2 pixels
    img[0:2, :] = 30
    img[-2:, :] = 30
    img[:, 0:2] = 30
    img[:, -2:] = 30

    cropped, bbox = crop_margins(img, padding=5, return_bbox=True)
    ymin, xmin, ymax, xmax = bbox
    assert ymin >= 0
    assert ymax <= 200
    assert cropped.size > 0


def test_crop_margins_speckle_noise():
    """Verify that isolated 1-2px noise in the margin does not expand the bounding box."""
    img = _create_synthetic_doc(
        height=300, width=300,
        margin_top=60, margin_bottom=60,
        margin_left=60, margin_right=60,
    )
    # Place isolated 1px speckles in the outer corners
    img[5, 5] = 0
    img[290, 10] = 0
    img[10, 290] = 0

    cropped, bbox = crop_margins(img, padding=5, return_bbox=True, noise_filter=True)
    ymin, xmin, ymax, xmax = bbox

    # The crop should ignore the isolated speckles at y=5, 290 and x=5, 290
    assert ymin >= 50, f"Speckle noise at y=5 was not filtered, got ymin={ymin}"
    assert ymax <= 250, f"Speckle noise at y=290 was not filtered, got ymax={ymax}"


def test_crop_margins_edge_touching_text():
    """Verify that text touching the 0-boundary is handled cleanly without clipping."""
    img = np.full((100, 100), 255, dtype=np.uint8)
    img[0:20, 0:30] = 0  # Text flush against top-left corner

    cropped, bbox = crop_margins(img, padding=5, return_bbox=True)
    ymin, xmin, ymax, xmax = bbox

    assert ymin == 0
    assert xmin == 0
    assert ymax == 25
    assert xmax == 35


def test_crop_margins_padding_clamping():
    """Verify that large padding values are safely clamped to image bounds."""
    img = _create_synthetic_doc(height=100, width=100, margin_top=20, margin_bottom=20, margin_left=20, margin_right=20)
    cropped, bbox = crop_margins(img, padding=500, return_bbox=True)

    ymin, xmin, ymax, xmax = bbox
    assert ymin == 0
    assert xmin == 0
    assert ymax == 100
    assert xmax == 100
    assert cropped.shape == (100, 100)


# ---------------------------------------------------------------------------
# Added after review: dark scanner border / polarity robustness
# ---------------------------------------------------------------------------

def test_crop_dark_scanner_border_does_not_flip_polarity():
    """A bright page with text, surrounded by a dark scanner border, must crop to the TEXT."""
    page = np.full((300, 400), 235, dtype=np.uint8)
    page[100:130, 150:250] = 20          # text block
    framed = np.full((360, 460), 30, dtype=np.uint8)   # dark scanner border
    framed[30:330, 30:430] = page
    cropped, (ymin, xmin, ymax, xmax) = crop_margins(framed, padding=5, return_bbox=True)
    # crop must be tight around the text block (130..160 x 180..280 in framed coords), not the whole scan
    assert ymax - ymin < 60
    assert xmax - xmin < 130
    assert ymin <= 130 and ymax >= 160 and xmin <= 180 and xmax >= 280


def test_crop_dense_ink_crop_keeps_polarity():
    """Tightly cropped blocks where ink covers >50% of the pixels must still be dark-ink."""
    img = np.full((100, 100), 255, dtype=np.uint8)
    img[5:95, 5:95] = 0
    _, bbox = crop_margins(img, padding=0, return_bbox=True)
    assert bbox == (5, 5, 95, 95)
