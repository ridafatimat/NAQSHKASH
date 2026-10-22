"""
NAQSHKASH
Rida - Day 4

PyTorch Dataset loader for Talim OCR.

Expected dataset structure:

dataset_root/
├── images/
├── labels/
├── json/
└── metadata.csv

Example:
datasets/Day4_sample/train
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TalimDataset(Dataset):
    """
    Dataset for loading Talim OCR samples.

    Each item returns:
        {
            "image": image tensor,
            "label_text": raw canonical TXT label,
            "sample_id": sample ID,
            "image_path": image path,
            "label_path": label path,
            "rows": expected logical row count,
            "cells_per_row": expected cells per row
        }
    """

    def __init__(
        self,
        dataset_root: str | Path,
        transform: Optional[Callable] = None,
    ):
        self.dataset_root = Path(
            dataset_root
        ).resolve()

        self.transform = transform

        self.metadata_path = (
            self.dataset_root
            / "metadata.csv"
        )

        self.images_dir = (
            self.dataset_root
            / "images"
        )

        self.labels_dir = (
            self.dataset_root
            / "labels"
        )

        self.json_dir = (
            self.dataset_root
            / "json"
        )

        self._validate_dataset_structure()

        self.samples = (
            self._load_metadata()
        )

        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found in "
                f"{self.dataset_root}"
            )


    # ========================================================
    # STRUCTURE VALIDATION
    # ========================================================

    def _validate_dataset_structure(
        self,
    ) -> None:

        required_paths = [
            self.dataset_root,
            self.metadata_path,
            self.images_dir,
            self.labels_dir,
            self.json_dir,
        ]

        for path in required_paths:

            if not path.exists():

                raise FileNotFoundError(
                    f"Required dataset path "
                    f"does not exist:\n{path}"
                )


    # ========================================================
    # METADATA
    # ========================================================

    def _load_metadata(
        self,
    ) -> list[dict]:

        samples = []

        with self.metadata_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(
                file
            )

            if reader.fieldnames is None:

                raise ValueError(
                    "metadata.csv has no header."
                )

            required_columns = {
                "sample_id",
                "png_path",
                "txt_path",
            }

            missing = (
                required_columns
                - set(reader.fieldnames)
            )

            if missing:

                raise ValueError(
                    "metadata.csv is missing "
                    f"required columns: "
                    f"{sorted(missing)}"
                )

            for row in reader:

                sample_id = (
                    row["sample_id"]
                    .strip()
                )

                image_path = (
                    self.dataset_root
                    / row["png_path"]
                ).resolve()

                label_path = (
                    self.dataset_root
                    / row["txt_path"]
                ).resolve()

                json_path = None

                if row.get("json_path"):

                    json_path = (
                        self.dataset_root
                        / row["json_path"]
                    ).resolve()

                if not image_path.exists():

                    raise FileNotFoundError(
                        f"Missing image for "
                        f"{sample_id}:\n"
                        f"{image_path}"
                    )

                if not label_path.exists():

                    raise FileNotFoundError(
                        f"Missing label for "
                        f"{sample_id}:\n"
                        f"{label_path}"
                    )

                samples.append(
                    {
                        "sample_id":
                            sample_id,

                        "image_path":
                            image_path,

                        "label_path":
                            label_path,

                        "json_path":
                            json_path,

                        "rows":
                            int(
                                row.get(
                                    "rows",
                                    0,
                                )
                            ),

                        "cells_per_row":
                            int(
                                row.get(
                                    "cells_per_row",
                                    0,
                                )
                            ),
                    }
                )

        return samples


    # ========================================================
    # PYTORCH DATASET INTERFACE
    # ========================================================

    def __len__(
        self,
    ) -> int:

        return len(
            self.samples
        )


    def __getitem__(
        self,
        index: int,
    ) -> dict:

        sample = (
            self.samples[index]
        )

        image_path = (
            sample["image_path"]
        )

        # Load image as grayscale.
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:

            raise ValueError(
                f"OpenCV could not read "
                f"image:\n{image_path}"
            )

        # Optional preprocessing.
        if self.transform is not None:

            image = self.transform(
                image
            )

        # Convert image to float32.
        image = image.astype(
            np.float32
        )

        # Normalize pixel values:
        # 0..255 -> 0..1
        image = (
            image
            / 255.0
        )

        # Convert:
        #
        # H x W
        #
        # into:
        #
        # 1 x H x W
        #
        image_tensor = (
            torch.from_numpy(
                image
            )
            .unsqueeze(0)
        )

        # ----------------------------------------------------
        # LOAD RAW TALIM LABEL
        # ----------------------------------------------------

        label_text = (
            sample["label_path"]
            .read_text(
                encoding="utf-8",
            )
        )

        # Remove only final file newline.
        #
        # Do NOT use .strip()
        # because spaces / line structure may be meaningful.
        label_text = (
            label_text.rstrip(
                "\r\n"
            )
        )

        return {

            "image":
                image_tensor,

            "label_text":
                label_text,

            "sample_id":
                sample["sample_id"],

            "image_path":
                str(
                    sample["image_path"]
                ),

            "label_path":
                str(
                    sample["label_path"]
                ),

            "rows":
                sample["rows"],

            "cells_per_row":
                sample[
                    "cells_per_row"
                ],
        }