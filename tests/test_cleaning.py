import cv2
import numpy as np
import pytest

from preprocessing.cleaning import clean_image


def test_clean_grayscale_image():
    image = np.full(
        (100, 200),
        255,
        dtype=np.uint8,
    )

    result = clean_image(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_clean_bgr_image_returns_grayscale():
    image = np.full(
        (100, 200, 3),
        255,
        dtype=np.uint8,
    )

    result = clean_image(image)

    assert result.shape == (100, 200)
    assert result.dtype == np.uint8


def test_clean_does_not_modify_input():
    image = np.random.randint(
        0,
        256,
        (100, 200),
        dtype=np.uint8,
    )

    original = image.copy()

    clean_image(image)

    assert np.array_equal(
        image,
        original,
    )


def test_empty_image_rejected():
    image = np.empty(
        (0, 100),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        clean_image(image)


def test_invalid_kernel_rejected():
    image = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        clean_image(
            image,
            median_kernel=4,
        )


def test_noise_reduction_preserves_shape():
    image = np.full(
        (100, 200),
        255,
        dtype=np.uint8,
    )

    cv2.rectangle(
        image,
        (50, 40),
        (150, 60),
        0,
        -1,
    )

    noisy = image.copy()

    noise = np.random.randint(
        0,
        100,
        noisy.shape,
        dtype=np.uint8,
    )

    noisy[
        noise < 5
    ] = 0

    result = clean_image(noisy)

    assert result.shape == image.shape