import copy
import random
from pathlib import Path

import pytest
import torch

from data.width_sampler import WidthBudgetBatchSampler
from data.width_sampler import resized_widths
from data.vocabulary import TalimVocabulary
from training import train_crnn as trainer


def test_sampler_contract():
    widths = [10 + i % 83 for i in range(500)]
    sampler = WidthBudgetBatchSampler(widths, 400, 8, seed=7)
    original = list(sampler)
    sampler.set_epoch(0)
    assert list(sampler) == original
    assert sorted(i for b in original for i in b) == list(range(500))
    assert all(len(b) <= 8 and max(widths[i] for i in b) * len(b) <= 400 for b in original)
    sampler.set_epoch(1)
    assert list(sampler) != original
    sampler.set_epoch(0, 9)
    assert list(sampler) == original[9:]
    assert len(sampler) == len(original) - 9
    sampler.set_epoch(0, len(original))
    assert list(sampler) == []
    with pytest.raises(ValueError, match='No samples were dropped'):
        WidthBudgetBatchSampler([401], 400, 8)


def test_atomic_checkpoint_and_resume(tmp_path, monkeypatch):
    trainer.DATASET_ROOT = trainer.TRAIN_PATH = trainer.VALIDATION_PATH = trainer.TEST_PATH = tmp_path
    torch.manual_seed(1)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    sampler = WidthBudgetBatchSampler([10] * 20, 40, 4)
    batches = list(sampler)
    def step():
        optimizer.zero_grad()
        model(torch.ones(2, 2)).square().mean().backward()
        optimizer.step()
    step()
    scheduler.step(0.7)
    scheduler.step(0.8)
    vocab = TalimVocabulary.for_tiers(['A'])
    progress = dict(next_batch=2, global_step=2, loss_sum=1., loss_batches=2)
    path = tmp_path / 'last.pt'
    def save():
        trainer.save_checkpoint(path, model, optimizer, scheduler, 0, .7, vocab, ['A'], [],
                                dict(epochs_without_improvement=1), scaler, progress, {'version': 1})
    save()
    saved = torch.load(path, weights_only=True)
    required = {'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict',
                'random_state', 'torch_rng_state', 'cuda_rng_states', 'history', 'char_to_idx',
                'idx_to_char', 'blank_index', 'config', 'next_batch', 'global_step', 'sampler_state'}
    assert required <= saved.keys()
    assert saved['next_batch'] == saved['global_step'] == 2
    previous = path.read_bytes()
    def interrupted(*args):
        raise OSError('interrupted write')
    with monkeypatch.context() as patch:
        patch.setattr(trainer.torch, 'save', interrupted)
        with pytest.raises(OSError):
            save()
    assert path.read_bytes() == previous
    step()
    expected = copy.deepcopy(model.state_dict())
    model.load_state_dict(saved['model_state_dict'])
    optimizer.load_state_dict(saved['optimizer_state_dict'])
    scheduler.load_state_dict(saved['scheduler_state_dict'])
    scaler.load_state_dict(saved['scaler_state_dict'])
    assert scheduler.state_dict() == saved['scheduler_state_dict']
    assert scaler.state_dict() == saved['scaler_state_dict']
    assert optimizer.state_dict()['state']
    sampler.set_epoch(saved['epoch'], saved['next_batch'])
    assert list(sampler) == batches[2:]
    progress = saved['progress']
    step()
    progress['global_step'] += 1
    assert progress['global_step'] == 3
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])


def test_width_cache_matches_resize_and_invalidates(tmp_path):
    import cv2
    import numpy as np
    from types import SimpleNamespace
    from data.collate import resize_preserve_aspect_ratio
    image = tmp_path / 'image.png'
    cv2.imwrite(str(image), np.zeros((31, 57), dtype=np.uint8))
    metadata = tmp_path / 'metadata.csv'
    metadata.write_text('identity')
    class Dataset:
        dataset_root = tmp_path
        metadata_path = metadata
        samples = [dict(image_path=image, label_path=tmp_path / 'label.txt', tier_id='1')]
        def __len__(self):
            return 1
    cache = tmp_path / 'widths.json'
    widths, identity = resized_widths(Dataset(), cache)
    assert widths == [resize_preserve_aspect_ratio(torch.zeros(1, 31, 57)).shape[-1]]
    assert resized_widths(Dataset(), cache) == (widths, identity)
    cv2.imwrite(str(image), np.zeros((31, 77), dtype=np.uint8))
    assert resized_widths(Dataset(), cache)[0] != widths


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware unavailable')
def test_cuda_scaler_roundtrip():
    model = torch.nn.Linear(2, 2).cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler('cuda')
    with torch.autocast('cuda'):
        loss = model(torch.ones(2, 2, device='cuda')).float().square().mean()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
    scaler.step(optimizer)
    scaler.update()
    restored = torch.amp.GradScaler('cuda')
    restored.load_state_dict(scaler.state_dict())
    assert restored.state_dict() == scaler.state_dict()
