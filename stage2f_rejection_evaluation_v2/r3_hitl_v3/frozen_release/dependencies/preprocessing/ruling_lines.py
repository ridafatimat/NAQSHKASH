import cv2
import numpy as np


def remove_ruling_lines(
    image: np.ndarray,
    min_line_length_ratio: float = 0.20,
    max_line_thickness: int = 3,
) -> np.ndarray:
    """
    Remove long horizontal ruling lines while preserving
    normal Talim strokes.

    Parameters
    ----------
    image:
        Grayscale or BGR image.

    min_line_length_ratio:
        Minimum detected horizontal line length as a
        fraction of image width.

    max_line_thickness:
        Maximum ruling-line thickness to remove.

    Returns
    -------
    np.ndarray
        Grayscale image with detected ruling lines removed.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")

    if image.size == 0:
        raise ValueError("image cannot be empty")

    if not 0 < min_line_length_ratio <= 1:
        raise ValueError(
            "min_line_length_ratio must be between 0 and 1"
        )

    if max_line_thickness < 1:
        raise ValueError(
            "max_line_thickness must be positive"
        )

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

    # Create binary ink mask.
    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]

    height, width = binary.shape

    # Only consider lines that are genuinely long
    # relative to the image.
    min_line_length = max(
        20,
        int(width * min_line_length_ratio),
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            min_line_length,
            max_line_thickness,
        ),
    )

    detected_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    # Remove only pixels identified as ruling lines.
    cleaned_binary = cv2.subtract(
        binary,
        detected_lines,
    )

    result = cv2.bitwise_not(
        cleaned_binary
    )

    return result.astype(np.uint8)