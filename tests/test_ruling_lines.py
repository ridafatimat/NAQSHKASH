import cv2
import numpy as np
import pytest

from preprocessing.ruling_lines import remove_ruling_lines


def test_horizontal_ruling_line_removed():

    image = np.full(
        (100, 300),
        255,
        dtype=np.uint8,
    )

    cv2.line(
        image,
        (20, 50),
        (280, 50),
        0,
        2,
    )

    result = remove_ruling_lines(image)

    # Center of the ruling line should become
    # background after removal.
    assert result[50, 150] > 200


def test_non_line_image_preserved():

    image = np.full(
        (100, 300),
        255,
        dtype=np.uint8,
    )

    cv2.circle(
        image,
        (150, 50),
        10,
        0,
        -1,
    )

    result = remove_ruling_lines(image)

    assert result.shape == image.shape


def test_bgr_input():

    image = np.full(
        (100, 300, 3),
        255,
        dtype=np.uint8,
    )

    result = remove_ruling_lines(image)

    assert result.shape == (100, 300)


def test_input_not_modified():

    image = np.full(
        (100, 300),
        255,
        dtype=np.uint8,
    )

    original = image.copy()

    remove_ruling_lines(image)

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
        remove_ruling_lines(image)