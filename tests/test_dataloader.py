"""
NAQSHKASH
Rida - Day 5

Tests for:
- fixed-height resizing
- aspect-ratio preservation
- variable-width batching
- right padding
- white padding
- real PyTorch DataLoader
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

from data.collate import (
    resize_preserve_aspect_ratio,
    talim_collate_fn,
)

from data.dataloader import (
    create_talim_dataloader,
)


DATASET_PATH = Path(
    r"C:\FYP Dataset Generation\Day4_sample\train"
)


# ============================================================
# TEST 1
# Resize to fixed height
# ============================================================

def test_resize_preserves_fixed_height():

    dataset = TalimDataset(
        DATASET_PATH
    )

    sample = dataset[0]

    original = sample[
        "image"
    ]

    resized = (
        resize_preserve_aspect_ratio(
            original,
            target_height=64,
        )
    )

    assert (
        resized.shape[0]
        ==
        1
    )

    assert (
        resized.shape[1]
        ==
        64
    )

    assert (
        resized.shape[2]
        >
        0
    )


# ============================================================
# TEST 2
# Aspect ratio preserved
# ============================================================

def test_resize_preserves_aspect_ratio():

    dataset = TalimDataset(
        DATASET_PATH
    )

    sample = dataset[0]

    original = sample[
        "image"
    ]

    _, original_height, original_width = (
        original.shape
    )

    resized = (
        resize_preserve_aspect_ratio(
            original,
            target_height=64,
        )
    )

    _, new_height, new_width = (
        resized.shape
    )

    original_ratio = (
        original_width
        / original_height
    )

    new_ratio = (
        new_width
        / new_height
    )

    assert abs(
        original_ratio
        - new_ratio
    ) < 0.02


# ============================================================
# TEST 3
# Variable-width collate
# ============================================================

def test_collate_variable_width_images():

    dataset = TalimDataset(
        DATASET_PATH
    )

    samples = [
        dataset[0],
        dataset[1],
        dataset[2],
        dataset[3],
    ]

    batch = talim_collate_fn(
        samples
    )

    images = batch[
        "images"
    ]

    widths = batch[
        "image_widths"
    ]

    assert images.ndim == 4

    assert (
        images.shape[0]
        ==
        4
    )

    assert (
        images.shape[1]
        ==
        1
    )

    assert (
        images.shape[2]
        ==
        64
    )

    assert (
        images.shape[3]
        ==
        int(
            widths.max()
        )
    )

    assert (
        len(
            batch[
                "label_texts"
            ]
        )
        ==
        4
    )

    assert (
        len(
            batch[
                "sample_ids"
            ]
        )
        ==
        4
    )


# ============================================================
# TEST 4
# True widths preserved
# ============================================================

def test_image_widths_are_preserved_before_padding():

    dataset = TalimDataset(
        DATASET_PATH
    )

    samples = [
        dataset[0],
        dataset[1],
        dataset[2],
    ]

    batch = talim_collate_fn(
        samples
    )

    widths = batch[
        "image_widths"
    ]

    assert torch.all(
        widths > 0
    )

    # Samples should have different widths
    # after fixed-height resizing.

    assert (
        len(
            torch.unique(
                widths
            )
        )
        >
        1
    )


# ============================================================
# TEST 5
# Right padding should be white
# ============================================================

def test_padding_is_white():

    dataset = TalimDataset(
        DATASET_PATH
    )

    samples = [
        dataset[0],
        dataset[1],
    ]

    batch = talim_collate_fn(
        samples
    )

    images = batch[
        "images"
    ]

    widths = batch[
        "image_widths"
    ]

    max_width = (
        images.shape[-1]
    )

    for index, width in enumerate(
        widths.tolist()
    ):

        if width < max_width:

            padding_area = (
                images[
                    index,
                    :,
                    :,
                    width:
                ]
            )

            assert torch.allclose(
                padding_area,
                torch.ones_like(
                    padding_area
                ),
            )


# ============================================================
# TEST 6
# Real DataLoader
# ============================================================

def test_real_dataloader_batch():

    loader = (
        create_talim_dataloader(
            DATASET_PATH,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )
    )

    batch = next(
        iter(
            loader
        )
    )

    assert (
        batch[
            "images"
        ].shape[0]
        ==
        4
    )

    assert (
        batch[
            "images"
        ].shape[1]
        ==
        1
    )

    assert (
        batch[
            "images"
        ].shape[2]
        ==
        64
    )

    assert (
        len(
            batch[
                "sample_ids"
            ]
        )
        ==
        4
    )


# ============================================================
# TEST 7
# Print first batch
# ============================================================

def test_print_first_batch():

    loader = (
        create_talim_dataloader(
            DATASET_PATH,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )
    )

    batch = next(
        iter(
            loader
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        "Batch tensor shape:",
        tuple(
            batch[
                "images"
            ].shape
        ),
    )

    print(
        "True resized widths:",
        batch[
            "image_widths"
        ].tolist(),
    )

    print(
        "Sample IDs:",
        batch[
            "sample_ids"
        ],
    )

    print()
    print(
        "Labels:"
    )

    for sample_id, label in zip(
        batch[
            "sample_ids"
        ],
        batch[
            "label_texts"
        ],
    ):

        print()
        print(
            "---",
            sample_id,
            "---"
        )

        print(
            label
        )

    print()
    print(
        "=" * 70
    )