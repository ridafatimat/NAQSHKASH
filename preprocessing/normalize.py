"""
preprocessing.normalize
=======================
Row-level normalization, grayscale conversion, and aspect-preserving resizing.

Author: Ayesha Amer (Day 3 Scope)
Project: NAQSHKASH FYP

Design Contract:
----------------
- Target Height: Exactly 64 pixels (fixed height for CRNN model input).
- Target Width: Variable width (aspect-ratio preserving).
  * No horizontal padding is applied here (padding and batching are handled in Day 5).
  * Guard against tiny widths with configurable min_width (default 16).
- Downscaling Interpolation: cv2.INTER_AREA to preserve thin strokes, dots, and diacritics.
- Upscaling Interpolation: cv2.INTER_LINEAR for smooth magnification.
- Polarity Convention:
  * 0.0 = dark ink (black).
  * 1.0 = light background (white paper).
  * Range: float32 in [0.0, 1.0].
  * Downstream rule: Any future DataLoader padding (Day 5) must use 1.0 (white background),
    and Hareem's image cleaning stage must preserve this polarity convention.
"""

from typing import Union, Optional
import numpy as np
import cv2


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an input image array to a 2D uint8 grayscale array without mutating the input.
    
    Parameters
    ----------
    image : np.ndarray
        Input image. Can be:
        - 2D array (H, W) uint8 or float.
        - 3D array (H, W, C) where C in {1, 3, 4} (BGR, RGB, or BGRA).
        
    Returns
    -------
    np.ndarray
        2D uint8 grayscale array of shape (H, W) in [0, 255].
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)}")
    if image.size == 0:
        raise ValueError("Cannot convert an empty image to grayscale.")

    # 2D Grayscale
    if image.ndim == 2:
        if image.dtype == np.uint8:
            return image.copy()
        elif np.issubdtype(image.dtype, np.floating):
            # If float in [0, 1], scale to [0, 255]
            if image.max() <= 1.0 and image.min() >= 0.0:
                return np.clip(image * 255.0, 0, 255).astype(np.uint8)
            else:
                return np.clip(image, 0, 255).astype(np.uint8)
        else:
            return np.clip(image, 0, 255).astype(np.uint8)

    # 3D Array
    elif image.ndim == 3:
        H, W, C = image.shape
        if C == 1:
            return to_grayscale(image[:, :, 0])
        elif C == 3:
            # Convert using standard OpenCV BGR to Gray luminance weights (0.299 R + 0.587 G + 0.114 B)
            if image.dtype == np.uint8:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                img_u8 = np.clip(image, 0, 255).astype(np.uint8)
                return cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
        elif C == 4:
            # BGRA -> Gray (composite with white background if alpha exists)
            img_u8 = np.clip(image, 0, 255).astype(np.uint8)
            return cv2.cvtColor(img_u8, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Unsupported channel count {C}. Expected 1, 3, or 4 channels.")

    else:
        raise ValueError(f"Unsupported array dimensions {image.ndim}. Expected 2D or 3D array.")


def resize_to_height(
    image: np.ndarray,
    height: int = 64,
    min_width: int = 16,
) -> np.ndarray:
    """
    Resize an image to a fixed height while strictly preserving the aspect ratio.
    
    No horizontal padding is applied (variable width output). Uses area-based
    interpolation (cv2.INTER_AREA) for downscaling to guarantee that thin strokes,
    subtle dots, and diacritics do not vanish.
    
    Parameters
    ----------
    image : np.ndarray
        Input 2D (H, W) or 3D (H, W, C) image array.
    height : int, default=64
        Fixed target height in pixels. Must be > 0.
    min_width : int, default=16
        Minimum allowable width in pixels. Guards against degenerate zero/tiny widths.
        
    Returns
    -------
    np.ndarray
        Resized image array of shape (height, W_new) or (height, W_new, C) where W_new >= min_width.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)}")
    if image.size == 0:
        raise ValueError("Cannot resize an empty image.")
    if height <= 0:
        raise ValueError(f"Target height must be positive, got {height}")
    if min_width <= 0:
        raise ValueError(f"min_width must be positive, got {min_width}")

    orig_h, orig_w = image.shape[:2]
    if orig_h == 0 or orig_w == 0:
        raise ValueError(f"Invalid image dimensions: shape={image.shape}")

    scale = float(height) / float(orig_h)
    new_w = max(int(min_width), int(round(orig_w * scale)))

    if orig_h == height and orig_w == new_w:
        return image.copy()

    # Choose optimal antialiasing interpolation based on scale
    if scale < 1.0:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    resized = cv2.resize(image, (new_w, height), interpolation=interpolation)
    return resized


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize an image array to float32 values in the range [0.0, 1.0].
    
    Polarity Convention:
    --------------------
    0.0 = Dark text ink (black)
    1.0 = Light paper background (white)
    
    Parameters
    ----------
    image : np.ndarray
        Input image array (e.g. uint8 in [0, 255] or float in [0.0, 1.0]).
        
    Returns
    -------
    np.ndarray
        float32 array with all values strictly clamped in [0.0, 1.0].
        Guaranteed to contain no NaNs and no Infs.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)}")
    if image.size == 0:
        raise ValueError("Cannot normalize an empty image.")

    if image.dtype == np.uint8:
        norm = image.astype(np.float32) / 255.0
    elif np.issubdtype(image.dtype, np.floating):
        # If float array already in [0.0, 1.0]
        if image.max() <= 1.0 and image.min() >= 0.0:
            norm = image.astype(np.float32).copy()
        else:
            # Float in [0, 255] range
            norm = image.astype(np.float32) / 255.0
    else:
        norm = image.astype(np.float32) / 255.0

    # Ensure bounds and numerical stability
    norm = np.clip(norm, 0.0, 1.0)
    if np.isnan(norm).any() or np.isinf(norm).any():
        norm = np.nan_to_num(norm, nan=1.0, posinf=1.0, neginf=0.0)

    return norm


def preprocess_row(
    row_image: np.ndarray,
    target_height: int = 64,
    min_width: int = 16,
) -> np.ndarray:
    """
    Execute full row-level transformation: Grayscale -> Resize to 64px height -> Normalize.
    
    Parameters
    ----------
    row_image : np.ndarray
        Raw unresized row crop (2D or 3D uint8/float).
    target_height : int, default=64
        Fixed target height in pixels.
    min_width : int, default=16
        Minimum width threshold in pixels.
        
    Returns
    -------
    np.ndarray
        2D float32 array of shape (target_height, W) where W >= min_width,
        with values in [0.0, 1.0] (0.0 = dark ink, 1.0 = light background).
    """
    gray = to_grayscale(row_image)
    resized = resize_to_height(gray, height=target_height, min_width=min_width)
    normalized = normalize(resized)
    return normalized
