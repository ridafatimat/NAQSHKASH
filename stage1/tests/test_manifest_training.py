import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
import torch

from data.manifest_dataset import ManifestTierDataset
from data.width_sampler import WidthBudgetBatchSampler
from training.manifest_training import load_vocabulary, lr_factor


def fixture_dataset(root, count=8):
    root.mkdir(parents=True, exist_ok=True)
    fields = ['split', 'tier_id', 'image_path', 'training_label', 'tier_name',
              'resized_width', 'label_length', 'ctc_feasible']
    for split in ('train', 'validation'):
        with (root / (split + '.csv')).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i in range(count):
                image = np.full((64, 16), 255, np.uint8)
                image[10:40, 5:10] = 0
                cv2.imwrite(str(root / f'{split}{i}.png'), image)
                writer.writerow(dict(split=split, tier_id=str(i), image_path=f'{split}{i}.png',
                                     training_label='A', tier_name=('upper', 'count', 'lower')[i % 3],
                                     resized_width=16, label_length=1, ctc_feasible=1))


def test_manifest_no_label_io_budget_and_seed(tmp_path):
    fixture_dataset(tmp_path)
    vocab, _ = load_vocabulary(Path(__file__).parents[1] / 'config/vocabulary.json')
    dataset = ManifestTierDataset(tmp_path, 'train.csv', vocab, 'train')
    assert dataset[0]['label_text'] == 'A'
    sampler = WidthBudgetBatchSampler(dataset.widths, 48, 2, 7)
    original = list(sampler)
    assert all(max(dataset.widths[i] for i in b) * len(b) <= 48 for b in original)
    sampler.set_epoch(0, 1)
    assert list(sampler) == original[1:]
    sampler.set_epoch(0)
    assert list(sampler) == original
    sampler.set_epoch(1)
    assert list(sampler) != original
    assert lr_factor(0, 1000, 500) == .002
    assert lr_factor(500, 1000, 500) == 1
    assert lr_factor(1000, 1000, 500) == .001


def test_exact_process_resume(tmp_path):
    fixture_dataset(tmp_path / 'dataset')
    base = [sys.executable, '-m', 'training.train_crnn', '--dataset-root', str(tmp_path / 'dataset'),
            '--train-manifest', 'train.csv', '--val-manifest', 'validation.csv',
            '--total-steps', '12', '--warmup-steps', '3', '--num-workers', '0', '--max-batch-size', '2',
            '--validate-every-steps', '6', '--log-every-steps', '3', '--no-amp']
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    def run(output, extra=()):
        result = subprocess.run(base + ['--output-root', str(output)] + list(extra),
                                env=env, capture_output=True, text=True, timeout=240)
        assert result.returncode == 0, result.stdout + result.stderr
        return torch.load(output / 'last.pt', weights_only=True)
    continuous = run(tmp_path / 'continuous')
    partial = run(tmp_path / 'resumed', ['--stop-after-steps', '5'])
    assert partial['global_step'] == 5 and partial['next_batch'] == 1
    resumed = run(tmp_path / 'resumed', ['--resume', 'auto'])
    assert resumed['global_step'] == continuous['global_step'] == 12
    assert resumed['progress'] == continuous['progress']
    assert resumed['scheduler_state_dict'] == continuous['scheduler_state_dict']
    assert resumed['optimizer_state_dict']['param_groups'] == continuous['optimizer_state_dict']['param_groups']
    for key, value in continuous['model_state_dict'].items():
        assert torch.equal(value, resumed['model_state_dict'][key]), key
    for key, value in continuous['optimizer_state_dict']['state'].items():
        for field, tensor in value.items():
            assert torch.equal(tensor, resumed['optimizer_state_dict']['state'][key][field])
    assert resumed['scaler_state_dict'] == {}
    rows = list(csv.DictReader((tmp_path / 'resumed/metrics.csv').open(encoding='utf-8')))
    assert sum(r['global_step'] == '3' for r in rows) == 1
    assert rows[-1]['global_step'] == '12'
    assert (tmp_path / 'resumed/best.pt').exists()
    result = subprocess.run(base + ['--output-root', str(tmp_path / 'resumed'), '--resume', 'auto',
                                    '--width-budget', '100'], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'configuration mismatch' in result.stderr


def test_manifest_exclusion_and_identity(tmp_path):
    fixture_dataset(tmp_path)
    vocab, _ = load_vocabulary(Path(__file__).parents[1] / 'config/vocabulary.json')
    path = tmp_path / 'train.csv'
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    fields = list(rows[0])
    rows[0]['ctc_feasible'] = 'false'
    def write():
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    write()
    first = ManifestTierDataset(tmp_path, 'train.csv', vocab, 'train')
    assert len(first) == 7 and first.excluded == 1
    rows[1]['training_label'] = 'B'
    write()
    second = ManifestTierDataset(tmp_path, 'train.csv', vocab, 'train')
    assert first.identity != second.identity
    rows[1]['training_label'] = '\u2603'
    write()
    with pytest.raises(KeyError, match='Unknown character'):
        ManifestTierDataset(tmp_path, 'train.csv', vocab, 'train')
    rows[1]['training_label'] = 'A'
    rows[1]['resized_width'] = '100'
    write()
    with pytest.raises(ValueError, match='dimensions differ'):
        ManifestTierDataset(tmp_path, 'train.csv', vocab, 'train')
