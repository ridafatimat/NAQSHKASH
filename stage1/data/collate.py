"""
NAQSHKASH
Rida - Day 5

Variable-width batching for Talim OCR.

This module:
1. Resizes images to a fixed height.
2. Preserves aspect ratio.
3. Keeps width variable.
4. Pads each image on the right.
5. Returns true resized widths for future CTC sequence handling.
6. Preserves Talim tier/layout metadata for reconstruction.
"""

from __future__ import annotations

from typing import List, Dict, Any

import torch
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================

TARGET_HEIGHT = 64

# Images are normalized:
# black = 0.0
# white = 1.0
#
# Therefore padding should be white.
PAD_VALUE = 1.0


# ============================================================
# RESIZE
# ============================================================

def resize_preserve_aspect_ratio(
    image: torch.Tensor,
    target_height: int = TARGET_HEIGHT,
) -> torch.Tensor:
    """
    Resize a grayscale tensor to fixed height
    while preserving aspect ratio.

    Input:
        image shape = [1, H, W]

    Output:
        [1, target_height, new_width]
    """

    if not isinstance(image, torch.Tensor):

        raise TypeError(
            f"Expected torch.Tensor, got {type(image)}"
        )

    if image.ndim != 3:

        raise ValueError(
            f"Expected [C,H,W], got {tuple(image.shape)}"
        )

    channels, height, width = image.shape

    if channels != 1:

        raise ValueError(
            f"Expected 1-channel grayscale image, got {channels}"
        )

    if height <= 0 or width <= 0:

        raise ValueError(
            f"Invalid image size: {height}x{width}"
        )

    # --------------------------------------------------------
    # Preserve aspect ratio
    # --------------------------------------------------------

    scale = (
        target_height
        /
        float(height)
    )

    new_width = max(
        1,
        int(
            round(
                width
                *
                scale
            )
        ),
    )

    # F.interpolate expects:
    #
    # [N, C, H, W]

    resized = F.interpolate(
        image.unsqueeze(0),
        size=(
            target_height,
            new_width,
        ),
        mode="bilinear",
        align_corners=False,
    )

    # Remove temporary batch dimension.
    resized = resized.squeeze(0)

    return resized


# ============================================================
# COLLATE
# ============================================================

def talim_collate_fn(
    batch: List[Dict[str, Any]],
    target_height: int = TARGET_HEIGHT,
) -> Dict[str, Any]:
    """
    Custom collate function for Talim OCR.

    Process:

        each image
            ↓
        resize to fixed height
            ↓
        preserve width ratio
            ↓
        find widest image
            ↓
        right-pad other images
            ↓
        stack into one batch

    The function also preserves Talim metadata when available.

    Returns:

        {
            "images": [B, 1, H, W],

            "image_widths":
                actual resized widths,

            "label_texts":
                OCR targets used by CRNN + CTC,

            "raw_label_texts":
                original Talim tier labels,

            "leading_offsets":
                removed leading layout spaces,

            "tier_names":
                upper / count / lower,

            "sample_ids":
                sample identifiers,

            "rows":
                page-level metadata when available,

            "cells_per_row":
                page-level metadata when available
        }
    """

    if not batch:

        raise ValueError(
            "Cannot collate an empty batch."
        )

    resized_images = []

    image_widths = []

    label_texts = []

    raw_label_texts = []

    leading_offsets = []

    tier_names = []

    sample_ids = []

    rows = []

    cells_per_row = []

    # ========================================================
    # Resize each image
    # ========================================================

    for sample in batch:

        image = sample[
            "image"
        ]

        resized = resize_preserve_aspect_ratio(
            image,
            target_height=target_height,
        )

        resized_images.append(
            resized
        )

        image_widths.append(
            resized.shape[-1]
        )

        # ----------------------------------------------------
        # OCR target
        # ----------------------------------------------------

        label_texts.append(
            sample[
                "label_text"
            ]
        )

        # ----------------------------------------------------
        # Original Talim label
        # ----------------------------------------------------

        raw_label_texts.append(
            sample.get(
                "raw_label_text",
                sample[
                    "label_text"
                ],
            )
        )

        # ----------------------------------------------------
        # Horizontal layout information
        # ----------------------------------------------------

        leading_offsets.append(
            sample.get(
                "leading_offset",
                0,
            )
        )

        # ----------------------------------------------------
        # Tier information
        # ----------------------------------------------------

        tier_names.append(
            sample.get(
                "tier_name",
                None,
            )
        )

        # ----------------------------------------------------
        # ID
        # ----------------------------------------------------

        sample_ids.append(
            sample[
                "sample_id"
            ]
        )

        # ----------------------------------------------------
        # Existing page-level metadata
        # ----------------------------------------------------

        rows.append(
            sample.get(
                "rows",
                0,
            )
        )

        cells_per_row.append(
            sample.get(
                "cells_per_row",
                0,
            )
        )

    # ========================================================
    # Find max width inside current batch
    # ========================================================

    max_width = max(
        image_widths
    )

    padded_images = []

    # ========================================================
    # Pad each image on the right
    # ========================================================

    for image in resized_images:

        current_width = (
            image.shape[-1]
        )

        padding_needed = (
            max_width
            -
            current_width
        )

        # F.pad format:
        #
        # (left, right, top, bottom)

        padded = F.pad(
            image,
            (
                0,
                padding_needed,
                0,
                0,
            ),
            mode="constant",
            value=PAD_VALUE,
        )

        padded_images.append(
            padded
        )

    # ========================================================
    # Stack images
    # ========================================================

    images = torch.stack(
        padded_images,
        dim=0,
    )

    widths_tensor = torch.tensor(
        image_widths,
        dtype=torch.long,
    )

    offsets_tensor = torch.tensor(
        leading_offsets,
        dtype=torch.long,
    )

    # ========================================================
    # Return batch
    # ========================================================

    return {

        "images":
            images,

        "image_widths":
            widths_tensor,

        # Clean OCR targets.
        "label_texts":
            label_texts,

        # Original Talim formatted labels.
        "raw_label_texts":
            raw_label_texts,

        # Positional layout information.
        "leading_offsets":
            offsets_tensor,

        # upper / count / lower
        "tier_names":
            tier_names,

        "sample_ids":
            sample_ids,

        "rows":
            rows,

        "cells_per_row":
            cells_per_row,
    }