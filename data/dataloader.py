"""
NAQSHKASH
Rida - Day 5

PyTorch DataLoader factory for Talim OCR.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from .dataset import TalimDataset
from .collate import talim_collate_fn


def create_talim_dataloader(
    dataset_root,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
):
    """
    Create DataLoader for Talim OCR.

    Parameters
    ----------
    dataset_root:
        Path to dataset split.

    batch_size:
        Samples per batch.

    shuffle:
        Whether to shuffle samples.

    num_workers:
        Number of loading worker processes.

        Keep 0 during development on Windows.

    Returns
    -------
    torch.utils.data.DataLoader
    """

    dataset = TalimDataset(
        dataset_root
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=talim_collate_fn,
    )

    return loader