"""Deterministic width-budget batches; no image decoding during epoch planning."""
import hashlib
import json
import random
import struct
from pathlib import Path

from torch.utils.data import Sampler, Subset


def resized_widths(dataset, cache_path, height=64):
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = dataset.indices if isinstance(dataset, Subset) else range(len(base))
    # Metadata and ordered relative identities make caches portable across mounts.
    records = [base.samples[i] for i in indices]
    identities = [(str(s['image_path'].relative_to(base.dataset_root)),
                   str(s['label_path'].relative_to(base.dataset_root)), s['tier_id']) for s in records]
    fingerprint = hashlib.sha256(base.metadata_path.read_bytes() +
                                 json.dumps(identities).encode()).hexdigest()
    signature = hashlib.sha256(json.dumps([
        (s['image_path'].stat().st_size, s['image_path'].stat().st_mtime_ns)
        for s in records]).encode()).hexdigest()
    path = Path(cache_path)
    if path.exists():
        cached = json.loads(path.read_text())
        if (cached['fingerprint'] == fingerprint and cached['height'] == height
                and cached.get('signature') == signature):
            return cached['widths'], fingerprint
    widths = []
    for sample in records:
        # Dataset tier images are PNG: IHDR gives dimensions without decoding pixels.
        with sample['image_path'].open('rb') as f:
            header = f.read(24)
        if header[:8] == b'\x89PNG\r\n\x1a\n':
            w, h = struct.unpack('>II', header[16:24])
        else:
            import cv2
            image = cv2.imread(str(sample['image_path']), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Unreadable image: {sample['image_path']}")
            h, w = image.shape
        if not w or not h:
            raise ValueError('Invalid PNG dimensions')
        widths.append(max(1, round(w * (height / float(h)))))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(fingerprint=fingerprint, signature=signature, height=height, widths=widths)))
    temporary.replace(path)
    return widths, fingerprint


class WidthBudgetBatchSampler(Sampler):
    VERSION = 1

    def __init__(self, widths, width_budget, max_batch_size, seed=42, pool_size=2048):
        self.widths = list(widths)
        if not widths or min(widths) < 1 or min(width_budget, max_batch_size, pool_size) < 1:
            raise ValueError('Widths, budget, batch size and pool size must be positive')
        if max(widths) > width_budget:
            raise ValueError(f'Widest sample ({max(widths)}) exceeds --width-budget {width_budget}; increase budget. No samples were dropped.')
        self.width_budget, self.max_batch_size = width_budget, max_batch_size
        self.seed, self.pool_size = seed, pool_size
        self.set_epoch(0)

    def set_epoch(self, epoch, start_batch=0):
        rng = random.Random(self.seed + epoch)
        indices = list(range(len(self.widths)))
        rng.shuffle(indices)
        batches = []
        for offset in range(0, len(indices), self.pool_size):
            pool = sorted(indices[offset:offset + self.pool_size], key=self.widths.__getitem__)
            batch = []
            for index in pool:
                width = self.widths[index]
                if batch and (len(batch) == self.max_batch_size or width * (len(batch) + 1) > self.width_budget):
                    batches.append(batch)
                    batch = []
                batch.append(index)
            if batch:
                batches.append(batch)
        rng.shuffle(batches)
        if not 0 <= start_batch <= len(batches):
            raise ValueError('Resume next_batch is outside reconstructed epoch')
        self.epoch, self.start_batch, self.batches = epoch, start_batch, batches

    def __iter__(self):
        return iter(self.batches[self.start_batch:])

    def __len__(self):
        return len(self.batches) - self.start_batch
