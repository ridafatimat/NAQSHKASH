"""Checkpoint CLI evaluates every requested sample and preserves saved vocabulary."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch
from models.crnn import CRNN


def test_checkpoint_cli(tmp_path):
    checkpoint = tmp_path / 'best.pt'
    torch.save({'model_state_dict': CRNN(2).state_dict(), 'char_to_idx': {'A': 1},
                'global_step': 7}, checkpoint)
    for split in ('validation', 'test'):
        directory = tmp_path / split
        (directory / 'images').mkdir(parents=True)
        (directory / 'labels').mkdir()
        cv2.imwrite(str(directory / 'images/0.png'), np.full((64, 32), 255, np.uint8))
        (directory / 'labels/0.txt').write_text('A', encoding='utf-8')
        with (directory / 'metadata.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['tier_id', 'row_id', 'parent_sample_id', 'tier_name', 'image_path', 'label_path'])
            writer.writeheader()
            writer.writerow(dict(tier_id='0', row_id='row0', parent_sample_id=split, tier_name='upper', image_path='images/0.png', label_path='labels/0.txt'))
    output = tmp_path / 'results.json'
    subprocess.run([sys.executable, '-m', 'evaluation.evaluate_checkpoint', '--checkpoint', str(checkpoint),
                    '--dataset-root', str(tmp_path), '--output', str(output)],
                   check=True, capture_output=True, text=True, timeout=60,
                   env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    result = json.loads(output.read_text())
    assert result['checkpoint_global_step'] == 7
    assert result['num_classes'] == 2
    for split in ('validation', 'test'):
        assert result['full_' + split]['samples'] == 1
        assert 0 <= result['full_' + split]['exact_accuracy'] <= 1
