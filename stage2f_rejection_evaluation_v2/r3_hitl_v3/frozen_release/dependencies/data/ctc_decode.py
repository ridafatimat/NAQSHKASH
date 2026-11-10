"""
NAQSHKASH
Basic greedy CTC decoding utility.

Used later to decode CRNN predictions.
"""

from __future__ import annotations

from typing import Iterable, List

from .vocabulary import (
    TalimVocabulary,
)


def collapse_ctc_sequence(
    indices: Iterable[int],
    blank_index: int = 0,
) -> List[int]:
    """
    Apply standard CTC collapse.

    Example:
        [0, 3, 3, 0, 5, 5, 0]

    becomes:
        [3, 5]

    Rules:
    - remove blank tokens
    - collapse consecutive repeated predictions
    """

    output = []

    previous = None

    for index in indices:

        index = int(
            index
        )

        if (
            index != blank_index
            and index != previous
        ):
            output.append(
                index
            )

        previous = index

    return output


def greedy_ctc_decode(
    indices: Iterable[int],
    vocabulary: TalimVocabulary,
) -> str:
    """
    Collapse a predicted CTC sequence and convert back to text.
    """

    collapsed = (
        collapse_ctc_sequence(
            indices,
            blank_index=vocabulary.blank_index,
        )
    )

    return vocabulary.decode(
        collapsed
    )