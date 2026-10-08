from pathlib import Path

import torch

from data.tier_dataset import (
    TalimTierDataset,
)


DATASET_PATH = Path(
    r"C:\FYP Dataset Generation\Day4_tier_dataset\train"
)


def test_tier_dataset_loads():

    dataset = TalimTierDataset(
        DATASET_PATH
    )

    # Every logical row has at least
    # one non-empty tier.
    assert len(dataset) >= 232


def test_tier_sample():

    dataset = TalimTierDataset(
        DATASET_PATH
    )

    sample = dataset[0]

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

    assert isinstance(
        sample["label_text"],
        str,
    )

    assert (
        sample["label_text"].strip()
        != ""
    )

    assert (
        sample["tier_name"]
        in {
            "upper",
            "count",
            "lower",
        }
    )


def test_tier_labels_are_single_line():

    dataset = TalimTierDataset(
        DATASET_PATH
    )

    for i in range(
        len(dataset)
    ):

        text = (
            dataset[i][
                "label_text"
            ]
        )

        assert "\n" not in text
        assert "\r" not in text


def test_print_first_tiers():

    dataset = TalimTierDataset(
        DATASET_PATH
    )

    for i in range(
        min(
            10,
            len(dataset),
        )
    ):

        sample = dataset[i]

        print()
        print(
            "=" * 60
        )

        print(
            "Tier:",
            sample["tier_id"],
        )

        print(
            "Type:",
            sample["tier_name"],
        )

        print(
            "Image:",
            tuple(
                sample["image"].shape
            ),
        )

        print(
            "Target:",
            repr(
                sample[
                    "label_text"
                ]
            ),
        )