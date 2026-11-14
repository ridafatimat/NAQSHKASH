"""
NAQSHKASH
Row-level Talim dataset for CRNN + CTC.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TalimRowDataset(Dataset):

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

        self.samples = self._load_metadata()

        if len(self.samples) == 0:
            raise ValueError(
                f"No row samples found in:\n"
                f"{self.dataset_root}"
            )

    def _load_metadata(self):

        samples = []

        with self.metadata_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as f:

            reader = csv.DictReader(f)

            required_columns = {
                "row_id",
                "parent_sample_id",
                "row_index",
                "image_path",
                "label_path",
            }

            if reader.fieldnames is None:
                raise ValueError(
                    "metadata.csv has no header."
                )

            missing = (
                required_columns
                - set(reader.fieldnames)
            )

            if missing:
                raise ValueError(
                    f"metadata.csv missing columns: "
                    f"{sorted(missing)}"
                )

            for row in reader:

                image_path = (
                    self.dataset_root
                    / row["image_path"]
                ).resolve()

                label_path = (
                    self.dataset_root
                    / row["label_path"]
                ).resolve()

                if not image_path.exists():
                    raise FileNotFoundError(
                        f"Missing row image:\n"
                        f"{image_path}"
                    )

                if not label_path.exists():
                    raise FileNotFoundError(
                        f"Missing row label:\n"
                        f"{label_path}"
                    )

                samples.append(
                    {
                        "row_id":
                            row["row_id"],

                        "parent_sample_id":
                            row[
                                "parent_sample_id"
                            ],

                        "row_index":
                            int(
                                row[
                                    "row_index"
                                ]
                            ),

                        "image_path":
                            image_path,

                        "label_path":
                            label_path,
                    }
                )

        return samples

    def __len__(self):

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
                f"Could not load row image:\n"
                f"{sample['image_path']}"
            )

        if self.transform is not None:
            image = self.transform(
                image
            )

        image = image.astype(
            np.float32
        )

        image = (
            image
            / 255.0
        )

        image_tensor = (
            torch.from_numpy(
                image
            )
            .unsqueeze(0)
        )

        label_text = (
            sample[
                "label_path"
            ]
            .read_text(
                encoding="utf-8",
            )
        )

        # Normalize line endings only.
        # Do NOT strip leading spaces/newlines,
        # because they may be meaningful.
        label_text = (
            label_text
            .replace(
                "\r\n",
                "\n",
            )
            .replace(
                "\r",
                "\n",
            )
        )

        return {
            "image":
                image_tensor,

            "label_text":
                label_text,

            "sample_id":
                sample[
                    "row_id"
                ],

            "row_id":
                sample[
                    "row_id"
                ],

            "parent_sample_id":
                sample[
                    "parent_sample_id"
                ],

            "row_index":
                sample[
                    "row_index"
                ],
        }