from pathlib import Path

import torch

from data.row_dataset import (
    TalimRowDataset,
)


DATASET_PATH = Path(
    r"C:\FYP Dataset Generation\Day4_row_dataset\train"
)


def test_row_dataset_loads():

    dataset = TalimRowDataset(
        DATASET_PATH
    )

    assert (
        len(dataset)
        >
        100
    )


def test_first_row():

    dataset = TalimRowDataset(
        DATASET_PATH
    )

    sample = dataset[0]

    assert isinstance(
        sample["image"],
        torch.Tensor,
    )

    assert (
        sample["image"].shape[0]
        ==
        1
    )

    assert isinstance(
        sample["label_text"],
        str,
    )

    assert (
        len(
            sample["label_text"]
        )
        >
        0
    )


def test_print_first_rows():

    dataset = TalimRowDataset(
        DATASET_PATH
    )

    for i in range(
        min(
            5,
            len(dataset),
        )
    ):

        sample = dataset[i]

        print()
        print(
            "=" * 60
        )

        print(
            "Row:",
            sample["row_id"],
        )

        print(
            "Parent:",
            sample["parent_sample_id"],
        )

        print(
            "Image shape:",
            tuple(
                sample["image"].shape
            ),
        )

        print(
            "Label:"
        )

        print(
            repr(
                sample["label_text"]
            )
        )