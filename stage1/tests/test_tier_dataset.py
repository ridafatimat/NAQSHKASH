from pathlib import Path

import torch

from data.tier_dataset import (
    TalimTierDataset,
)


DATASET_PATH = Path(
    "datasets/Day4_tier_dataset/train"
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

    assert isinstance(
        sample["raw_label_text"],
        str,
    )

    assert isinstance(
        sample["leading_offset"],
        int,
    )

    assert (
        sample["leading_offset"]
        >=
        0
    )

    assert (
        sample["label_text"]
        !=
        ""
    )

    assert (
        sample["tier_name"]
        in {
            "upper",
            "count",
            "lower",
        }
    )


def test_ocr_target_has_no_leading_spaces():

    dataset = TalimTierDataset(
        DATASET_PATH
    )

    for i in range(
        len(dataset)
    ):

        sample = dataset[i]

        ocr_text = (
            sample[
                "label_text"
            ]
        )

        assert not ocr_text.startswith(
            " "
        )


def test_raw_label_reconstruction():

    dataset = TalimTierDataset(
        DATASET_PATH
    )

    for i in range(
        len(dataset)
    ):

        sample = dataset[i]

        raw_label = (
            sample[
                "raw_label_text"
            ]
        )

        ocr_text = (
            sample[
                "label_text"
            ]
        )

        leading_offset = (
            sample[
                "leading_offset"
            ]
        )

        reconstructed = (
            " " * leading_offset
            +
            ocr_text
        )

        assert (
            reconstructed
            ==
            raw_label
        )


def test_internal_spaces_are_preserved():

    dataset = TalimTierDataset(
        DATASET_PATH
    )

    found_internal_space_example = False

    for i in range(
        len(dataset)
    ):

        sample = dataset[i]

        raw_label = (
            sample[
                "raw_label_text"
            ]
        )

        ocr_text = (
            sample[
                "label_text"
            ]
        )

        expected_ocr_text = (
            raw_label.lstrip(
                " "
            )
        )

        assert (
            ocr_text
            ==
            expected_ocr_text
        )

        if " " in ocr_text:

            found_internal_space_example = True

    # Dataset should contain at least one target
    # where internal spacing remains after
    # leading layout spaces are removed.
    assert found_internal_space_example


def test_tier_labels_are_single_line():

    dataset = TalimTierDataset(
        DATASET_PATH
    )

    for i in range(
        len(dataset)
    ):

        sample = dataset[i]

        assert (
            "\n"
            not in sample[
                "label_text"
            ]
        )

        assert (
            "\r"
            not in sample[
                "label_text"
            ]
        )

        assert (
            "\n"
            not in sample[
                "raw_label_text"
            ]
        )

        assert (
            "\r"
            not in sample[
                "raw_label_text"
            ]
        )


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
            sample[
                "tier_id"
            ],
        )

        print(
            "Type:",
            sample[
                "tier_name"
            ],
        )

        print(
            "Image:",
            tuple(
                sample[
                    "image"
                ].shape
            ),
        )

        print(
            "Raw label:",
            repr(
                sample[
                    "raw_label_text"
                ]
            ),
        )

        print(
            "OCR target:",
            repr(
                sample[
                    "label_text"
                ]
            ),
        )

        print(
            "Leading offset:",
            sample[
                "leading_offset"
            ],
        )