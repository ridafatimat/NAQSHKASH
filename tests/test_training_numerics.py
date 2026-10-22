import json
import csv
from pathlib import Path

import pytest
import torch

from data.encoder import CTCLabelEncoder
from models.crnn import CRNN
from tests.test_manifest_training import fixture_dataset
from training import manifest_training as training


@pytest.mark.parametrize('configured_scale', [4096., 8192.])
def test_fresh_amp_scale_and_saved_scaler_precedence(tmp_path, monkeypatch, configured_scale):
    # CUDA scaler construction/state restoration is lazy: no GPU allocation here.
    # Only availability is mocked; GradScaler and its state loading are real.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    args = training.parser().parse_args(['--dataset-root', str(tmp_path), '--output-root', str(tmp_path),
                                         '--amp-init-scale', str(configured_scale)])
    assert training.parser().get_default('amp_init_scale') == 4096.
    fresh = training.create_grad_scaler(args, torch.device('cuda'))
    assert fresh.is_enabled() and fresh.get_scale() == configured_scale
    state = fresh.state_dict()
    assert state['growth_factor'] == 2. and state['backoff_factor'] == .5
    assert state['growth_interval'] == 2000
    # A checkpoint must restore scale AND all dynamic growth/backoff state.
    state.update(scale=1024., growth_factor=3., backoff_factor=.25, growth_interval=123, _growth_tracker=17)
    torch.save({'scaler_state_dict': state}, tmp_path / 'scaler.pt')
    resumed = training.create_grad_scaler(args, torch.device('cuda'))
    checkpoint = torch.load(tmp_path / 'scaler.pt', weights_only=True)
    resumed.load_state_dict(checkpoint['scaler_state_dict'])
    assert resumed.get_scale() == 1024.
    assert resumed.state_dict() == state


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf'])
def test_invalid_amp_init_scale_rejected(tmp_path, value):
    with pytest.raises(SystemExit) as error:
        training.main(['--dataset-root', str(tmp_path), '--output-root', str(tmp_path),
                       '--total-steps', '1', '--amp-init-scale=' + value])
    assert error.value.code == 2


def tiny_batch():
    return dict(images=torch.ones(2, 1, 64, 16), image_widths=torch.tensor([16, 12]),
                label_texts=['AA', 'B'], tier_names=['upper', 'count'])


def encoded_batch():
    return dict(targets=torch.tensor([1, 1, 2]), target_lengths=torch.tensor([2, 1]))


def test_runtime_ctc_repeats_and_actual_widths():
    batch = tiny_batch()
    lengths, required = training.runtime_ctc_lengths(CRNN, batch, encoded_batch(), 4)
    assert lengths.tolist() == [4, 3]
    assert required.tolist() == [3, 1]
    assert lengths.device.type == 'cpu' and lengths.dtype == torch.long
    batch['image_widths'][0] = 8  # AA needs three timesteps, not merely two
    with pytest.raises(ValueError, match='Runtime CTC infeasible'):
        training.runtime_ctc_lengths(CRNN, batch, encoded_batch(), 4)
    batch = tiny_batch()
    with pytest.raises(ValueError, match='logits time dimension'):
        training.runtime_ctc_lengths(CRNN, batch, encoded_batch(), 3)


def test_ctc_receives_fp32_even_under_autocast():
    logits = torch.randn(4, 2, 4, dtype=torch.bfloat16, requires_grad=True)
    encoded = encoded_batch()
    criterion = torch.nn.CTCLoss(zero_infinity=True)
    observed = []
    def spy(log_probs, targets, lengths, target_lengths):
        observed.append(log_probs.dtype)
        assert targets.dtype == torch.long
        assert lengths.dtype == target_lengths.dtype == torch.long
        assert lengths.device.type == target_lengths.device.type == 'cpu'
        return criterion(log_probs, targets, lengths, target_lengths)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss, log_probs = training.fp32_ctc_loss(logits, encoded['targets'], torch.tensor([4, 3]),
                                                encoded['target_lengths'], spy)
    assert observed == [torch.float32] and log_probs.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_separate_counters_and_only_first_five_details():
    state = dict(global_step=7, epoch=2, next_batch=9, skipped_batches=0)
    batch = tiny_batch()
    logits = torch.zeros(4, 2, 4)
    lines = []
    for index in range(7):
        kind = 'loss' if index % 2 == 0 else 'gradient'
        with pytest.warns(UserWarning, match='Non-finite loss' if kind == 'loss' else 'Non-finite gradient norm'):
            training.record_numerical_failure(
                state, kind, batch, torch.tensor([4, 3]), torch.tensor([2, 1]), torch.tensor([3, 1]),
                torch.tensor(float('nan') if kind == 'loss' else 2.), logits, logits.float().log_softmax(2),
                65536., torch.tensor(float('inf')) if kind == 'gradient' else None, emit=lines.append)
    assert state['skipped_batches'] == 7
    assert state['nonfinite_loss_batches'] == 4
    assert state['nonfinite_gradient_batches'] == 3
    assert state['numerical_failure_details'] == len(lines) == 5
    first = json.loads(lines[0])
    assert first['event'] == 'nonfinite_loss' and first['loss'] == 'nan'
    assert first['global_step'] == 7 and first['batch_index'] == 9
    assert first['ctc_required_timesteps_min_max'] == [1, 3]
    assert first['image_width_min_max'] == [12, 16]
    assert first['logits_finite'] and first['log_probs_finite']
    assert first['tier_distribution'] == {'upper': 1, 'count': 1}
    assert json.loads(lines[1])['gradient_norm'] == 'inf'


@pytest.mark.parametrize('kind', ['loss', 'gradient'])
def test_injected_failure_skips_update_and_keeps_threshold(tmp_path, monkeypatch, kind):
    fixture_dataset(tmp_path / 'data')
    monkeypatch.setattr(training.signal, 'signal', lambda *_: None)
    original_model = training.CRNN
    initial = {}
    def model_factory(*args, **kwargs):
        model = original_model(*args, **kwargs)
        initial.update({name: param.detach().clone() for name, param in model.named_parameters()})
        if kind == 'gradient':
            model.classifier.weight.register_hook(lambda grad: grad * float('nan'))
        return model
    monkeypatch.setattr(training, 'CRNN', model_factory)
    if kind == 'loss':
        original_loss = training.fp32_ctc_loss
        def invalid_loss(*args, **kwargs):
            loss, log_probs = original_loss(*args, **kwargs)
            return loss * float('nan'), log_probs
        monkeypatch.setattr(training, 'fp32_ctc_loss', invalid_loss)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    output = tmp_path / 'run'
    try:
        with pytest.warns(UserWarning, match='Non-finite'), pytest.raises(RuntimeError, match='threshold reached'):
            training.main(['--dataset-root', str(tmp_path / 'data'), '--train-manifest', 'train.csv',
                           '--val-manifest', 'validation.csv', '--output-root', str(output), '--total-steps', '1',
                           '--num-workers', '0', '--max-batch-size', '2', '--no-amp', '--max-nonfinite-batches', '1'])
    finally:
        torch.set_num_threads(previous_threads)
    checkpoint = torch.load(output / 'last.pt', weights_only=True)
    assert checkpoint['global_step'] == 0 and checkpoint['next_batch'] == 1
    assert checkpoint['scheduler_state_dict']['last_epoch'] == 0
    assert checkpoint['optimizer_state_dict']['state'] == {}
    assert checkpoint['nonfinite_loss_batches'] == int(kind == 'loss')
    assert checkpoint['nonfinite_gradient_batches'] == int(kind == 'gradient')
    for name, original in initial.items():
        assert torch.equal(checkpoint['model_state_dict'][name], original), name
    details = [json.loads(line) for line in (output / 'numerical_failures.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(details) == 1 and details[0]['event'] == 'nonfinite_' + kind
    assert details[0]['log_probs_dtype'] == 'torch.float32'
    assert training.parser().get_default('max_nonfinite_batches') == 10


def test_resume_prediagnostic_v3_preserves_csv(tmp_path, monkeypatch):
    fixture_dataset(tmp_path / 'data')
    monkeypatch.setattr(training.signal, 'signal', lambda *_: None)
    output = tmp_path / 'run'
    args = ['--dataset-root', str(tmp_path / 'data'), '--train-manifest', 'train.csv',
            '--val-manifest', 'validation.csv', '--output-root', str(output), '--total-steps', '2',
            '--num-workers', '0', '--max-batch-size', '2', '--no-amp',
            '--log-every-steps', '1', '--stop-after-steps', '1']
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        training.main(args)
        checkpoint = torch.load(output / 'last.pt', weights_only=True)
        for key in ('nonfinite_loss_batches', 'nonfinite_gradient_batches', 'numerical_failure_details'):
            checkpoint.pop(key)
            checkpoint['progress'].pop(key)
        checkpoint['progress']['skipped_batches'] = checkpoint['skipped_batches'] = 1
        torch.save(checkpoint, output / 'last.pt')
        csv_path = output / 'metrics.csv'
        with csv_path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        fields = [key for key in rows[0] if key not in ('nonfinite_loss_batches', 'nonfinite_gradient_batches')]
        with csv_path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        with pytest.warns(UserWarning, match='Historical skipped batches'):
            training.main(args + ['--resume', 'auto'])
    finally:
        torch.set_num_threads(previous_threads)
    resumed = torch.load(output / 'last.pt', weights_only=True)
    assert resumed['global_step'] == 2 and resumed['skipped_batches'] == 1
    assert resumed['nonfinite_loss_batches'] == resumed['nonfinite_gradient_batches'] == 0
    with csv_path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == fields
        rows = list(reader)
    assert len(rows) == 2 and [row['global_step'] for row in rows] == ['1', '2']
    assert all(None not in row for row in rows)
