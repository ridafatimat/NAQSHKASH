"""
preprocessing.crop
==================
Module for document cropping and margin removal in Kashmiri Carpet Talim OCR.

Author: Ayesha Amer (Day 1 Scope)
Project: NAQSHKASH FYP

Provides robust automatic margin detection and content bounding-box cropping:
- Supports 2D Grayscale (H, W) and 3D Color (H, W, C - BGR / RGB) arrays.
- Preserves input immutability (never mutates input arrays).
- Handles uneven lighting, dark borders, and speckle noise.
- Preserves thin strokes, diacritics/dots, and edge-touching symbols.
- Safe fallback on blank/uniform images.
"""

from typing import Tuple, Union, Optional
import numpy as np
from scipy import ndimage


def _binarize_ink(
    image: np.ndarray,
    threshold: Optional[int] = None,
    dark_ink: Optional[bool] = None,
) -> np.ndarray:
    """
    Produce a boolean ink mask where True indicates text ink and False indicates background.
    
    Parameters
    ----------
    image : np.ndarray
        Input image, 2D grayscale (H, W) or 3D color (H, W, C).
    threshold : Optional[int]
        Explicit binarization threshold (0-255). If None, Otsu's thresholding is computed.
    dark_ink : Optional[bool]
        Whether ink is darker than background. If None, automatically detected from border pixels.
        
    Returns
    -------
    np.ndarray
        2D boolean array of shape (H, W) where True = ink pixel.
    """
    if image.ndim == 3:
        # Standard luminance conversion: 0.299 R + 0.587 G + 0.114 B
        gray = (
            0.299 * image[:, :, 0].astype(np.float32)
            + 0.587 * image[:, :, 1].astype(np.float32)
            + 0.114 * image[:, :, 2].astype(np.float32)
        ).astype(np.uint8)
    elif image.ndim == 2:
        gray = image.copy()
    else:
        raise ValueError(f"Unsupported image shape {image.shape}. Expected 2D or 3D array.")

    # Detect polarity if not specified.
    # Paper is "bright" if EITHER the outer 1px ring OR the whole image is mostly bright:
    #  * ring only  -> fails on scans with a dark scanner border / shadow (ring is dark but
    #                  the page inside is bright paper);
    #  * median only -> fails on dense, tightly cropped crops where ink covers >50% of pixels.
    # Using "either is bright" handles both; a genuinely inverted page (light ink on dark
    # paper) is dark in both statistics and is still detected.
    if dark_ink is None:
        border_pixels = np.concatenate([
            gray[0, :], gray[-1, :],
            gray[:, 0], gray[:, -1]
        ])
        ring_bright = float(np.median(border_pixels)) > 127.0
        global_bright = float(np.median(gray)) > 127.0
        is_dark_ink = ring_bright or global_bright
    else:
        is_dark_ink = dark_ink

    # Otsu thresholding if not provided
    if threshold is None:
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        total = float(gray.size)
        current_max = -1.0
        computed_thresh = 128
        sum_total = float(np.dot(np.arange(256), hist))
        sum_b = 0.0
        w_b = 0.0

        for i in range(256):
            w_b += float(hist[i])
            if w_b == 0:
                continue
            w_f = total - w_b
            if w_f == 0:
                break
            sum_b += float(i * hist[i])
            m_b = sum_b / w_b
            m_f = (sum_total - sum_b) / w_f
            var_between = w_b * w_f * ((m_b - m_f) ** 2)
            if var_between > current_max:
                current_max = var_between
                computed_thresh = i
        thresh = computed_thresh
    else:
        thresh = threshold

    if is_dark_ink:
        # For dark ink, pixels with intensity <= threshold are ink
        ink_mask = gray <= thresh
        # If threshold captured almost everything (e.g. uniform image), handle gracefully
        if np.all(ink_mask) and np.std(gray) < 5.0:
            ink_mask = np.zeros_like(gray, dtype=bool)
    else:
        # For light ink on dark background
        ink_mask = gray >= thresh
        if np.all(ink_mask) and np.std(gray) < 5.0:
            ink_mask = np.zeros_like(gray, dtype=bool)

    return _remove_border_artifacts(ink_mask)


def _remove_border_artifacts(mask: np.ndarray, span_ratio: float = 0.6) -> np.ndarray:
    """
    Remove scanner-border / frame / shadow-edge blobs from a boolean ink mask.

    A connected component is treated as an artefact only if it TOUCHES the image edge and
    spans at least `span_ratio` of the image width or height (a page frame or dark scanner
    bar). Small edge-touching glyphs (real symbols cut by the border) are preserved.
    """
    if not mask.any():
        return mask
    H, W = mask.shape
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    out = mask.copy()
    for k, sl in enumerate(ndimage.find_objects(labeled), start=1):
        ys, xs = sl
        touches = ys.start == 0 or xs.start == 0 or ys.stop == H or xs.stop == W
        if not touches:
            continue
        if (ys.stop - ys.start) >= span_ratio * H or (xs.stop - xs.start) >= span_ratio * W:
            out[labeled == k] = False
    return out


def crop_margins(
    image: np.ndarray,
    padding: int = 10,
    return_bbox: bool = False,
    noise_filter: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Remove empty margins around text in a Talim document image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image as numpy array (uint8). Can be 2D grayscale (H, W) or 3D BGR/RGB (H, W, C).
        The input array is never modified in-place.
    padding : int, default=10
        Extra margin pixels added around detected ink boundary. Clamped to image dimensions.
    return_bbox : bool, default=False
        If True, returns a tuple `(cropped_image, bbox)` where bbox is `(ymin, xmin, ymax, xmax)`.
        If False, returns `cropped_image` only.
    noise_filter : bool, default=True
        Whether to filter isolated 1-2px speckle noise during margin detection to avoid
        stray noise pixels inflating the bounding box.
        
    Returns
    -------
    cropped_image : np.ndarray
        Cropped sub-array with margins removed (same dtype and channel structure as input).
    bbox : Tuple[int, int, int, int] (only if return_bbox=True)
        Bounding box in `(ymin, xmin, ymax, xmax)` coordinates relative to original input.
        
    Notes
    -----
    Coordinate Convention:
        `ymin` : Top row index (inclusive)
        `xmin` : Left column index (inclusive)
        `ymax` : Bottom row index (exclusive)
        `xmax` : Right column index (exclusive)
        Slicing: `image[ymin:ymax, xmin:xmax]`
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)}")
    if image.size == 0:
        raise ValueError("Cannot crop empty image array.")

    H, W = image.shape[:2]
    
    # Generate ink mask
    ink_mask = _binarize_ink(image)

    # Filter isolated speckle noise if requested
    if noise_filter and np.any(ink_mask):
        clean_mask = ndimage.binary_opening(ink_mask, structure=np.ones((2, 2), dtype=bool))
        if np.any(clean_mask):
            eval_mask = clean_mask
        else:
            eval_mask = ink_mask
    else:
        eval_mask = ink_mask

    y_indices, x_indices = np.where(eval_mask)

    # Fallback on blank image: return full image copy
    if len(y_indices) == 0 or len(x_indices) == 0:
        bbox = (0, 0, H, W)
        cropped = image.copy()
        if return_bbox:
            return cropped, bbox
        return cropped

    # Compute bounding box
    raw_ymin = int(y_indices.min())
    raw_ymax = int(y_indices.max() + 1)
    raw_xmin = int(x_indices.min())
    raw_xmax = int(x_indices.max() + 1)

    # Apply padding and clamp
    ymin = max(0, raw_ymin - padding)
    ymax = min(H, raw_ymax + padding)
    xmin = max(0, raw_xmin - padding)
    xmax = min(W, raw_xmax + padding)

    bbox = (ymin, xmin, ymax, xmax)
    cropped = image[ymin:ymax, xmin:xmax].copy()

    if return_bbox:
        return cropped, bbox
    return cropped
