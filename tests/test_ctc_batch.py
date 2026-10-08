"""
NAQSHKASH
Integration test:

DataLoader
    ↓
raw labels
    ↓
CTC encoder
    ↓
flattened targets + lengths
"""

from pathlib import Path
import sys


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


from data.dataloader import (
    create_talim_dataloader,
)

from data.vocabulary import (
    build_vocabulary_from_labels,
)

from data.encoder import (
    CTCLabelEncoder,
)


DATASET_PATH = Path(
    r"C:\FYP Dataset Generation\Day4_sample\train"
)


def test_real_batch_can_be_encoded_for_ctc():

    loader = (
        create_talim_dataloader(
            DATASET_PATH,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )
    )

    batch = next(
        iter(loader)
    )

    vocabulary = (
        build_vocabulary_from_labels(
            DATASET_PATH
            / "labels"
        )
    )

    encoder = (
        CTCLabelEncoder(
            vocabulary
        )
    )

    encoded = (
        encoder.encode_batch(
            batch[
                "label_texts"
            ]
        )
    )

    targets = (
        encoded[
            "targets"
        ]
    )

    lengths = (
        encoded[
            "target_lengths"
        ]
    )

    assert (
        len(lengths)
        ==
        4
    )

    assert (
        targets.numel()
        ==
        int(
            lengths.sum()
        )
    )

    assert (
        int(
            targets.min()
        )
        >
        0
    )

    assert (
        int(
            targets.max()
        )
        <
        len(
            vocabulary
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        "Batch images:",
        tuple(
            batch[
                "images"
            ].shape
        ),
    )

    print(
        "Image widths:",
        batch[
            "image_widths"
        ].tolist(),
    )

    print(
        "Target lengths:",
        lengths.tolist(),
    )

    print(
        "Flattened targets:",
        targets.shape,
    )

    print(
        "Vocabulary size:",
        len(
            vocabulary
        ),
    )

    print(
        "=" * 70
    )