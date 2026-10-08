"""
NAQSHKASH
Rida - Day 4

Dataset loader validation.
"""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from data.dataset import TalimDataset


DATASET_PATH = Path(
    r"C:\FYP Dataset Generation\Day4_sample\train"
)


def test_dataset_loads():

    dataset = TalimDataset(
        DATASET_PATH
    )

    assert len(dataset) == 100


def test_first_sample():

    dataset = TalimDataset(
        DATASET_PATH
    )

    sample = dataset[0]

    assert (
        sample["sample_id"]
        ==
        "train_000001"
    )

    assert isinstance(
        sample["image"],
        torch.Tensor,
    )

    assert (
        sample["image"].ndim
        ==
        3
    )

    assert (
        sample["image"].shape[0]
        ==
        1
    )

    assert (
        sample["image"].dtype
        ==
        torch.float32
    )

    assert (
        float(
            sample["image"].min()
        )
        >= 0.0
    )

    assert (
        float(
            sample["image"].max()
        )
        <= 1.0
    )

    assert isinstance(
        sample["label_text"],
        str,
    )

    assert len(
        sample["label_text"]
    ) > 0


def test_first_five_samples():

    dataset = TalimDataset(
        DATASET_PATH
    )

    for index in range(5):

        sample = dataset[
            index
        ]

        print()
        print(
            "=" * 60
        )

        print(
            "Sample:",
            sample["sample_id"],
        )

        print(
            "Image shape:",
            tuple(
                sample["image"]
                .shape
            ),
        )

        print(
            "Rows:",
            sample["rows"],
        )

        print(
            "Cells per row:",
            sample[
                "cells_per_row"
            ],
        )

        print(
            "Label:"
        )

        print(
            sample[
                "label_text"
            ]
        )

        assert (
            sample["image"]
            .shape[0]
            ==
            1
        )