"""
NAQSHKASH
DataLoader for tier-level CRNN training.
"""

from torch.utils.data import DataLoader

from .tier_dataset import (
    TalimTierDataset,
)

from .collate import (
    talim_collate_fn,
)


def create_tier_dataloader(
    dataset_root,
    batch_size=8,
    shuffle=True,
    num_workers=0,
):

    dataset = TalimTierDataset(
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