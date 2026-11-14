"""
NAQSHKASH
CRNN architecture tests.
"""

import torch

from models.crnn import (
    CRNN,
)


def test_crnn_forward():

    model = CRNN(
        num_classes=66
    )

    images = torch.randn(
        4,
        1,
        64,
        160,
    )

    logits = model(
        images
    )

    # Width 160 / 4 = 40
    assert (
        logits.shape[0]
        ==
        40
    )

    assert (
        logits.shape[1]
        ==
        4
    )

    assert (
        logits.shape[2]
        ==
        66
    )


def test_crnn_output_lengths():

    widths = torch.tensor(
        [
            48,
            80,
            100,
            160,
        ],
        dtype=torch.long,
    )

    lengths = (
        CRNN.output_lengths(
            widths
        )
    )

    assert (
        lengths.tolist()
        ==
        [
            12,
            20,
            25,
            40,
        ]
    )