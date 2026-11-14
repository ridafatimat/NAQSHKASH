import cv2
import numpy as np


def clean_image(
    image: np.ndarray,
    median_kernel: int = 3,
) -> np.ndarray:
    """
    Clean a Talim image using conservative denoising
    and local contrast normalization.

    The function is designed to preserve thin Talim
    strokes while reducing small sensor/speckle noise.

    Parameters
    ----------
    image:
        Grayscale or BGR image.

    median_kernel:
        Odd kernel size for median filtering.

    Returns
    -------
    np.ndarray
        Cleaned grayscale image.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")

    if image.size == 0:
        raise ValueError("image cannot be empty")

    if image.ndim == 3:
        if image.shape[2] != 3:
            raise ValueError(
                "Color image must have 3 channels"
            )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    elif image.ndim == 2:
        gray = image.copy()

    else:
        raise ValueError(
            f"Unsupported image shape: {image.shape}"
        )

    if gray.dtype != np.uint8:
        gray = cv2.normalize(
            gray,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

    if median_kernel < 1 or median_kernel % 2 == 0:
        raise ValueError(
            "median_kernel must be a positive odd number"
        )

    # Conservative denoising.
    # Avoid large kernels because small Talim strokes
    # must be preserved.
    if median_kernel > 1:
        gray = cv2.medianBlur(
            gray,
            median_kernel,
        )

    # Local contrast enhancement.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    cleaned = clahe.apply(gray)

    return cleaned