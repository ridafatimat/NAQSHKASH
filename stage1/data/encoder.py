"""
NAQSHKASH
Rida - CTC Label Encoder

Converts raw Talim label strings into flattened PyTorch
targets required by torch.nn.CTCLoss.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from .vocabulary import (
    TalimVocabulary,
    normalize_label_text,
)


class CTCLabelEncoder:
    """
    Encode Talim text labels for CTC training.
    """

    def __init__(
        self,
        vocabulary: TalimVocabulary,
    ):

        self.vocabulary = (
            vocabulary
        )


    def encode_text(
        self,
        text: str,
    ) -> torch.Tensor:
        """
        Encode one Talim label string.

        Returns:
            1D LongTensor
        """

        text = normalize_label_text(
            text
        )

        encoded = (
            self.vocabulary.encode(
                text
            )
        )

        return torch.tensor(
            encoded,
            dtype=torch.long,
        )


    def encode_batch(
        self,
        label_texts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a list of labels into CTC-compatible targets.

        Example:

        labels:
            ["ABC", "DE"]

        encoded:
            [1, 2, 3]
            [4, 5]

        flattened targets:
            [1, 2, 3, 4, 5]

        target_lengths:
            [3, 2]
        """

        encoded_labels = []

        target_lengths = []

        for text in label_texts:

            encoded = (
                self.encode_text(
                    text
                )
            )

            encoded_labels.append(
                encoded
            )

            target_lengths.append(
                encoded.numel()
            )

        if encoded_labels:

            targets = torch.cat(
                encoded_labels,
                dim=0,
            )

        else:

            targets = torch.empty(
                0,
                dtype=torch.long,
            )

        target_lengths_tensor = (
            torch.tensor(
                target_lengths,
                dtype=torch.long,
            )
        )

        return {
            "targets":
                targets,

            "target_lengths":
                target_lengths_tensor,
        }