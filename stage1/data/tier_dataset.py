"""
NAQSHKASH
Tier-level OCR Dataset for CRNN + CTC.

This dataset keeps the original Talim tier label for layout reconstruction,
but provides a separate OCR target for model training.

Example:
    raw label:      "    N"
    OCR target:     "N"
    leading offset: 4
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TalimTierDataset(Dataset):

    def __init__(
        self,
        dataset_root,
        transform=None,
    ):

        self.dataset_root = Path(
            dataset_root
        ).resolve()

        self.transform = transform

        self.metadata_path = (
            self.dataset_root
            / "metadata.csv"
        )

        if not self.metadata_path.exists():

            raise FileNotFoundError(
                f"metadata.csv not found:\n"
                f"{self.metadata_path}"
            )

        self.samples = (
            self._load_metadata()
        )

        if not self.samples:

            raise ValueError(
                "Tier dataset contains "
                "no samples."
            )

    def _load_metadata(self):

        samples = []

        with self.metadata_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as f:

            reader = csv.DictReader(
                f
            )

            for row in reader:

                image_path = (
                    self.dataset_root
                    / row[
                        "image_path"
                    ]
                ).resolve()

                label_path = (
                    self.dataset_root
                    / row[
                        "label_path"
                    ]
                ).resolve()

                if not image_path.exists():

                    raise FileNotFoundError(
                        f"Missing tier image:\n"
                        f"{image_path}"
                    )

                if not label_path.exists():

                    raise FileNotFoundError(
                        f"Missing tier label:\n"
                        f"{label_path}"
                    )

                samples.append(
                    {
                        "tier_id":
                            row[
                                "tier_id"
                            ],

                        "row_id":
                            row[
                                "row_id"
                            ],

                        "parent_sample_id":
                            row[
                                "parent_sample_id"
                            ],

                        "tier_name":
                            row[
                                "tier_name"
                            ],

                        "image_path":
                            image_path,

                        "label_path":
                            label_path,
                    }
                )

        return samples

    def __len__(
        self
    ):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):

        sample = (
            self.samples[
                index
            ]
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = cv2.imread(
            str(
                sample[
                    "image_path"
                ]
            ),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:

            raise ValueError(
                f"Could not read image:\n"
                f"{sample['image_path']}"
            )

        # ----------------------------------------------------
        # Optional image transform
        # ----------------------------------------------------

        if self.transform is not None:

            image = self.transform(
                image
            )

        # ----------------------------------------------------
        # Normalize image to [0, 1]
        # ----------------------------------------------------

        image = (
            image.astype(
                np.float32
            )
            / 255.0
        )

        # [H, W]
        #
        # ->
        #
        # [1, H, W]

        image_tensor = (
            torch.from_numpy(
                image
            )
            .unsqueeze(0)
        )

        # ----------------------------------------------------
        # Load original Talim tier label
        # ----------------------------------------------------

        raw_label_text = (
            sample[
                "label_path"
            ]
            .read_text(
                encoding="utf-8",
            )
            .rstrip(
                "\r\n"
            )
        )

        # ----------------------------------------------------
        # Separate OCR content from layout information
        # ----------------------------------------------------
        #
        # Talim canonical rendering may contain leading spaces
        # because the symbol belongs to a later horizontal run.
        #
        # Example:
        #
        #     raw_label_text = "    N"
        #
        # becomes:
        #
        #     ocr_text       = "N"
        #     leading_offset = 4
        #
        # We remove ONLY leading spaces.
        #
        # Internal spaces are preserved because they may
        # represent meaningful Talim structure.
        #
        # Example:
        #
        #     " E B  QM"
        #
        # becomes:
        #
        #     ocr_text       = "E B  QM"
        #     leading_offset = 1
        #
        # ----------------------------------------------------

        without_leading_spaces = (
            raw_label_text.lstrip(
                " "
            )
        )

        leading_offset = (
            len(
                raw_label_text
            )
            -
            len(
                without_leading_spaces
            )
        )

        ocr_text = (
            without_leading_spaces
        )

        # ----------------------------------------------------
        # Return training sample
        # ----------------------------------------------------

        return {

            # Image used by CRNN.
            "image":
                image_tensor,

            # OCR target used by CRNN + CTC.
            "label_text":
                ocr_text,

            # Original Talim label.
            # Preserved for reconstruction / validation.
            "raw_label_text":
                raw_label_text,

            # Number of leading layout spaces removed
            # from the OCR target.
            "leading_offset":
                leading_offset,

            # General sample identifier expected by
            # the existing collate function.
            "sample_id":
                sample[
                    "tier_id"
                ],

            "tier_id":
                sample[
                    "tier_id"
                ],

            "row_id":
                sample[
                    "row_id"
                ],

            "parent_sample_id":
                sample[
                    "parent_sample_id"
                ],

            # upper / count / lower
            "tier_name":
                sample[
                    "tier_name"
                ],

            # Keep paths available for debugging.
            "image_path":
                str(
                    sample[
                        "image_path"
                    ]
                ),

            "label_path":
                str(
                    sample[
                        "label_path"
                    ]
                ),
        }