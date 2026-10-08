"""
NAQSHKASH
Vocabulary + CTC encoder tests.
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


from data.dataset import (
    TalimDataset,
)

from data.vocabulary import (
    TalimVocabulary,
    build_vocabulary_from_labels,
    normalize_label_text,
)

from data.encoder import (
    CTCLabelEncoder,
)

from data.ctc_decode import (
    greedy_ctc_decode,
)


DATASET_PATH = Path(
    r"C:\FYP Dataset Generation\Day4_sample\train"
)


LABELS_PATH = (
    DATASET_PATH
    / "labels"
)


# ============================================================
# TEST 1
# Build real vocabulary
# ============================================================

def test_build_real_vocabulary():

    vocabulary = (
        build_vocabulary_from_labels(
            LABELS_PATH
        )
    )

    assert (
        len(vocabulary)
        >
        1
    )

    assert (
        vocabulary.blank_index
        ==
        0
    )

    assert (
        0
        not in
        vocabulary.char_to_index.values()
    )


# ============================================================
# TEST 2
# Important formatting chars preserved
# ============================================================

def test_space_and_newline_are_in_vocabulary():

    vocabulary = (
        build_vocabulary_from_labels(
            LABELS_PATH
        )
    )

    assert (
        " "
        in
        vocabulary.char_to_index
    )

    assert (
        "\n"
        in
        vocabulary.char_to_index
    )


# ============================================================
# TEST 3
# Encode/decode one real label
# ============================================================

def test_real_label_round_trip():

    dataset = TalimDataset(
        DATASET_PATH
    )

    vocabulary = (
        build_vocabulary_from_labels(
            LABELS_PATH
        )
    )

    sample = dataset[0]

    original = normalize_label_text(
        sample[
            "label_text"
        ]
    )

    encoded = (
        vocabulary.encode(
            original
        )
    )

    decoded = (
        vocabulary.decode(
            encoded
        )
    )

    assert (
        decoded
        ==
        original
    )


# ============================================================
# TEST 4
# Encoder produces LongTensor
# ============================================================

def test_encoder_single_label():

    dataset = TalimDataset(
        DATASET_PATH
    )

    vocabulary = (
        build_vocabulary_from_labels(
            LABELS_PATH
        )
    )

    encoder = CTCLabelEncoder(
        vocabulary
    )

    encoded = (
        encoder.encode_text(
            dataset[0][
                "label_text"
            ]
        )
    )

    assert isinstance(
        encoded,
        torch.Tensor,
    )

    assert (
        encoded.dtype
        ==
        torch.long
    )

    assert (
        encoded.ndim
        ==
        1
    )

    assert (
        encoded.numel()
        >
        0
    )


# ============================================================
# TEST 5
# Batch encoding
# ============================================================

def test_batch_encoding():

    dataset = TalimDataset(
        DATASET_PATH
    )

    vocabulary = (
        build_vocabulary_from_labels(
            LABELS_PATH
        )
    )

    encoder = CTCLabelEncoder(
        vocabulary
    )

    label_texts = [
        dataset[0]["label_text"],
        dataset[1]["label_text"],
        dataset[2]["label_text"],
    ]

    result = (
        encoder.encode_batch(
            label_texts
        )
    )

    targets = result[
        "targets"
    ]

    lengths = result[
        "target_lengths"
    ]

    assert (
        targets.dtype
        ==
        torch.long
    )

    assert (
        lengths.dtype
        ==
        torch.long
    )

    assert (
        len(lengths)
        ==
        3
    )

    assert (
        targets.numel()
        ==
        int(
            lengths.sum()
        )
    )


# ============================================================
# TEST 6
# Newline normalization
# ============================================================

def test_newline_normalization():

    text = (
        "ABC\r\nDEF\rGHI"
    )

    normalized = (
        normalize_label_text(
            text
        )
    )

    assert (
        normalized
        ==
        "ABC\nDEF\nGHI"
    )


# ============================================================
# TEST 7
# Greedy CTC collapse
# ============================================================

def test_greedy_ctc_decode():

    vocabulary = (
        TalimVocabulary(
            characters=[
                "A",
                "B",
            ]
        )
    )

    a = (
        vocabulary
        .char_to_index["A"]
    )

    b = (
        vocabulary
        .char_to_index["B"]
    )

    blank = (
        vocabulary.blank_index
    )

    predicted = [
        blank,
        a,
        a,
        blank,
        b,
        b,
        blank,
    ]

    decoded = (
        greedy_ctc_decode(
            predicted,
            vocabulary,
        )
    )

    assert (
        decoded
        ==
        "AB"
    )


# ============================================================
# TEST 8
# Print actual vocabulary
# ============================================================

def test_print_real_vocabulary():

    vocabulary = (
        build_vocabulary_from_labels(
            LABELS_PATH
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        "Vocabulary size "
        "(including CTC blank):",
        len(
            vocabulary
        ),
    )

    print()
    print(
        "CTC blank index:",
        vocabulary.blank_index,
    )

    print()
    print(
        "Characters:"
    )

    for char, index in (
        vocabulary
        .char_to_index
        .items()
    ):

        print(
            f"{index:3d} -> "
            f"{repr(char)}"
        )

    print(
        "=" * 70
    )