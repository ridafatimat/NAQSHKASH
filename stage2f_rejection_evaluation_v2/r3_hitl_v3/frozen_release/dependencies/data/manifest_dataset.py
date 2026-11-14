"""Approved manifest reader. No label I/O or image inventory at startup."""
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ManifestTierDataset(Dataset):
    def __init__(self, root, manifest, vocabulary, split, limit=None):
        self.root = Path(root).resolve()
        self.path = self.root / manifest
        self.identity = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.samples, self.excluded = [], 0
        with self.path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            required = {'image_path', 'training_label', 'tier_name', 'resized_width',
                        'label_length', 'ctc_feasible', 'split'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f'Missing manifest columns: {required - set(reader.fieldnames or [])}')
            for row in reader:
                if row['split'] != split:
                    raise ValueError(f'Expected {split} manifest, got {row["split"]}')
                if row['ctc_feasible'].lower() in ('0', 'false'):
                    self.excluded += 1
                    continue
                if row['ctc_feasible'].lower() not in ('1', 'true'):
                    raise ValueError('Invalid ctc_feasible value')
                text, width = row['training_label'], int(row['resized_width'])
                if not text or len(text) != int(row['label_length']):
                    raise ValueError('Empty label or label_length mismatch')
                vocabulary.encode(text)
                required_steps = len(text) + sum(a == b for a, b in zip(text, text[1:]))
                if width < 4 or required_steps > width // 4:
                    raise ValueError('Manifest width is CTC infeasible')
                relative = Path(row['image_path'])
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('image_path must remain relative to dataset root')
                if limit is None or len(self.samples) < limit:
                    self.samples.append(dict(width=width, image_path=row['image_path'],
                                             training_label=text, tier_name=row['tier_name'],
                                             tier_id=row.get('tier_id', str(len(self.samples))),
                                             raw_label=row.get('raw_label', text),
                                             leading_offset=row.get('leading_offset', '0')))
        if not self.samples:
            raise ValueError('Empty manifest dataset')
        self.widths = [row['width'] for row in self.samples]
        print(f'{split}: {len(self.samples):,} samples; excluded infeasible={self.excluded}', flush=True)
        for index in range(min(100, len(self))):
            self[index]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        path = self.root / row['image_path']
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f'Missing or undecodable image: {path}')
        h, w = image.shape
        if min(h, w) < 1 or max(1, round(w * 64 / h)) != row['width']:
            raise ValueError(f'Image dimensions differ from manifest: {path}')
        return dict(image=torch.from_numpy(image.astype(np.float32) / 255).unsqueeze(0),
                    label_text=row['training_label'], raw_label_text=row.get('raw_label', row['training_label']),
                    leading_offset=int(row.get('leading_offset', 0)),
                    tier_name=row['tier_name'], sample_id=row.get('tier_id', str(index)))
