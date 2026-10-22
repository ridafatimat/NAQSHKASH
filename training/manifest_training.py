"""Step-budgeted Stage 1 training on approved manifests (checkpoint schema v3)."""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import signal
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.collate import talim_collate_fn
from data.encoder import CTCLabelEncoder
from data.manifest_dataset import ManifestTierDataset
from data.vocabulary import TalimVocabulary
from data.width_sampler import WidthBudgetBatchSampler
from models.crnn import CRNN


def lr_factor(step, total, warmup, minimum=0.001):
    if warmup and step < warmup:
        return (step + 1) / warmup
    position = min(1., max(0., (step - warmup) / max(1, total - warmup)))
    return minimum + (1 - minimum) * (1 + math.cos(math.pi * position)) / 2


def load_vocabulary(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    mapping = data['char_to_idx']
    if len(mapping) != 65 or set(mapping.values()) != set(range(1, 66)):
        raise ValueError('Expected fixed 65-symbol vocabulary with blank=0')
    vocabulary = TalimVocabulary.for_tiers(mapping)
    vocabulary.char_to_index = mapping
    vocabulary.index_to_char = {index: char for char, index in mapping.items()}
    vocabulary.index_to_char[0] = vocabulary.blank_token
    digest = hashlib.sha256(json.dumps(mapping, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
    return vocabulary, digest


def runtime_ctc_lengths(model, batch, encoded, time_steps):
    """Validate actual collated widths and encoded targets, including repeats."""
    widths = batch['image_widths']
    targets, target_lengths = encoded['targets'], encoded['target_lengths']
    if any(t.dtype != torch.long or t.device.type != 'cpu'
           for t in (widths, targets, target_lengths)):
        raise ValueError('CTC widths/encoded targets/lengths must be CPU int64 before transfer')
    lengths = model.output_lengths(widths)
    if widths.numel() != target_lengths.numel() or widths.numel() != batch['images'].shape[0]:
        raise ValueError('CTC batch/length counts differ')
    if bool((widths < 4).any()) or bool((widths > batch['images'].shape[-1]).any()):
        raise ValueError('Invalid actual collated image widths')
    if int(target_lengths.sum()) != targets.numel() or bool((target_lengths <= 0).any()):
        raise ValueError('Invalid flattened CTC target lengths')
    required = []
    offset = 0
    for count in target_lengths.tolist():
        label = targets[offset:offset + count]
        required.append(count + int((label[1:] == label[:-1]).sum()))
        offset += count
    required = torch.tensor(required, dtype=torch.long)
    if bool((lengths < required).any()) or bool((lengths > time_steps).any()):
        raise ValueError('Runtime CTC infeasible: output length < required timesteps or > logits time dimension')
    return lengths, required


def fp32_ctc_loss(logits, targets, input_lengths, target_lengths, criterion):
    """Only the model forward uses autocast; CTC explicitly receives float32."""
    with torch.autocast(device_type=logits.device.type, enabled=False):
        log_probs = logits.float().log_softmax(2)
        assert log_probs.dtype == torch.float32
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
    return loss, log_probs


def record_numerical_failure(state, kind, batch, lengths, target_lengths, required,
                             loss, logits, log_probs, amp_scale, norm=None,
                             detail_limit=5, emit=print):
    """Always count/warn; materialize tensor diagnostics only for first failures."""
    state['skipped_batches'] += 1
    key = {'loss': 'nonfinite_loss_batches', 'gradient': 'nonfinite_gradient_batches'}[kind]
    state[key] = state.get(key, 0) + 1
    failure_count = state.get('numerical_failure_details', 0)
    warnings.warn(f'Non-finite {"loss" if kind == "loss" else "gradient norm"}; '
                  f'optimizer and scheduler skipped; global_step={state["global_step"]} '
                  f'epoch={state["epoch"]} batch_index={state["next_batch"]} '
                  f'{key}={state[key]} skipped_batches={state["skipped_batches"]}')
    if failure_count >= detail_limit:
        return
    state['numerical_failure_details'] = failure_count + 1
    def bounds(tensor):
        return [int(tensor.min()), int(tensor.max())]
    def scalar(tensor):
        value = float(tensor.detach().cpu())
        return value if math.isfinite(value) else str(value)
    payload = dict(event='nonfinite_' + kind, global_step=state['global_step'],
                   epoch=state['epoch'], batch_index=state['next_batch'],
                   batch_size=len(batch['label_texts']), image_width_min_max=bounds(batch['image_widths']),
                   input_lengths_min_max=bounds(lengths), target_lengths_min_max=bounds(target_lengths),
                   ctc_required_timesteps_min_max=bounds(required), loss=scalar(loss),
                   amp_scale=amp_scale, logits_finite=bool(torch.isfinite(logits).all()),
                   log_probs_finite=bool(torch.isfinite(log_probs).all()),
                   log_probs_dtype=str(log_probs.dtype), input_lengths_device=str(lengths.device),
                   input_lengths_dtype=str(lengths.dtype), target_lengths_device=str(target_lengths.device),
                   target_lengths_dtype=str(target_lengths.dtype),
                   gradient_norm=scalar(norm) if norm is not None else None,
                   tier_distribution=dict(Counter(batch.get('tier_names', ['unknown'] * len(batch['label_texts'])))))
    emit(json.dumps(payload, allow_nan=False))


def create_grad_scaler(args, device):
    """Fresh-run scale only; resume load_state_dict overrides this initialization."""
    return torch.amp.GradScaler('cuda', init_scale=args.amp_init_scale,
                                enabled=args.amp and device.type == 'cuda')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--train-manifest', default='manifests/stratified_train_150k.csv')
    p.add_argument('--val-manifest', default='manifests/stratified_validation_10k.csv')
    p.add_argument('--vocabulary', type=Path, default=Path(__file__).parent.parent / 'config/vocabulary.json')
    p.add_argument('--resume')
    p.add_argument('--total-steps', type=int)
    p.add_argument('--benchmark-steps', type=int)
    p.add_argument('--warmup-steps', type=int, default=500)
    p.add_argument('--amp-init-scale', type=float, default=4096.,
                   help='Fresh AMP run initial scale; saved scaler state takes precedence on resume')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--prefetch-factor', type=int, default=2)
    for name in ('amp', 'pin-memory', 'persistent-workers'):
        p.add_argument('--' + name, dest=name.replace('-', '_'), action='store_true', default=True)
        p.add_argument('--no-' + name, dest=name.replace('-', '_'), action='store_false')
    p.add_argument('--width-budget', type=int, default=16384)
    p.add_argument('--max-batch-size', type=int, default=32)
    p.add_argument('--checkpoint-every-steps', type=int, default=1000)
    p.add_argument('--validate-every-steps', type=int, default=2000)
    p.add_argument('--log-every-steps', type=int, default=100)
    p.add_argument('--max-runtime-minutes', type=float, default=660)
    p.add_argument('--shutdown-margin-seconds', type=float, default=120)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--stop-after-steps', type=int)
    p.add_argument('--train-limit', type=int)
    p.add_argument('--val-limit', type=int)
    p.add_argument('--max-nonfinite-batches', type=int, default=10)
    p.add_argument('--failure-detail-limit', type=int, default=5)
    return p


def main(argv=None):
    from training.train_crnn import evaluate, print_evaluation
    p = parser()
    a = p.parse_args(argv)
    if not math.isfinite(a.amp_init_scale) or a.amp_init_scale <= 0:
        p.error('amp-init-scale must be positive and finite')
    if a.benchmark_steps:
        if a.resume:
            p.error('Benchmark uses a separate fresh output directory')
        a.total_steps = a.benchmark_steps
    if not a.total_steps or a.total_steps < 1:
        p.error('Supply --total-steps after benchmarking, or --benchmark-steps 200')
    for key in ('width_budget', 'max_batch_size', 'prefetch_factor', 'checkpoint_every_steps',
                'validate_every_steps', 'log_every_steps', 'max_nonfinite_batches'):
        if getattr(a, key) < 1:
            p.error(f'{key} must be positive')
    if a.num_workers < 0 or a.warmup_steps < 0 or a.learning_rate <= 0 or a.failure_detail_limit < 0:
        p.error('Invalid worker, warmup or learning-rate setting')
    for key in ('train_limit', 'val_limit', 'stop_after_steps'):
        if getattr(a, key) is not None and getattr(a, key) < 1:
            p.error(f'{key} must be positive')
    if a.max_runtime_minutes * 60 <= a.shutdown_margin_seconds:
        p.error('Runtime must exceed shutdown safety margin')
    started = time.perf_counter()
    deadline = started + a.max_runtime_minutes * 60 - a.shutdown_margin_seconds
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocabulary, vocab_hash = load_vocabulary(a.vocabulary)
    train = ManifestTierDataset(a.dataset_root, a.train_manifest, vocabulary, 'train', a.train_limit)
    val = ManifestTierDataset(a.dataset_root, a.val_manifest, vocabulary, 'validation', a.val_limit)
    sampler = WidthBudgetBatchSampler(train.widths, a.width_budget, a.max_batch_size, a.seed)
    def loader(dataset, batch_sampler):
        options = dict(num_workers=a.num_workers, pin_memory=a.pin_memory,
                       collate_fn=talim_collate_fn, batch_sampler=batch_sampler,
                       generator=torch.Generator().manual_seed(a.seed))
        if a.num_workers:
            options.update(persistent_workers=a.persistent_workers, prefetch_factor=a.prefetch_factor)
        return DataLoader(dataset, **options)
    train_loader = loader(train, sampler)
    val_loader = loader(val, WidthBudgetBatchSampler(val.widths, a.width_budget, a.max_batch_size, a.seed))
    model = CRNN(len(vocabulary)).to(device)
    encoder = CTCLabelEncoder(vocabulary)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, a.total_steps, a.warmup_steps))
    scaler = create_grad_scaler(a, device)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    critical = ('total_steps', 'warmup_steps', 'width_budget', 'max_batch_size', 'seed',
                'learning_rate', 'amp', 'train_limit', 'val_limit', 'validate_every_steps')
    identities = dict(train=train.identity, validation=val.identity)
    state = dict(epoch=0, next_batch=0, global_step=0, best_val_cer=float('inf'),
                 skipped_batches=0, stale_validations=0, nonfinite_loss_batches=0,
                 nonfinite_gradient_batches=0, numerical_failure_details=0)
    a.output_root.mkdir(parents=True, exist_ok=True)
    resume = a.output_root / 'last.pt' if a.resume == 'auto' else Path(a.resume) if a.resume else None
    if resume:
        checkpoint = torch.load(resume, map_location='cpu', weights_only=True)
        if checkpoint.get('format_version') != 3:
            raise ValueError('Resume requires manifest checkpoint schema v3')
        if checkpoint['vocabulary_hash'] != vocab_hash:
            raise ValueError('Resume vocabulary hash mismatch')
        if checkpoint['manifest_identity'] != identities:
            raise ValueError('Resume manifest identity mismatch')
        if checkpoint['sampler_state'] != dict(version=sampler.VERSION, seed=a.seed, pool_size=sampler.pool_size):
            raise ValueError('Resume sampler version or settings mismatch')
        for key in critical:
            if checkpoint['config'][key] != config[key]:
                raise ValueError(f'Resume configuration mismatch: {key}')
        if checkpoint['amp_active'] != scaler.is_enabled():
            raise ValueError('Resume requires the same active AMP mode')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        state.update(checkpoint['progress'])
        if state['skipped_batches'] and 'nonfinite_loss_batches' not in checkpoint['progress']:
            warnings.warn('Historical skipped batches predate split diagnostics and cannot be classified retroactively')
        random.setstate(checkpoint['python_rng_state'])
        nr = checkpoint['numpy_rng_state']
        np.random.set_state((nr[0], np.array(nr[1], dtype=np.uint32), nr[2], nr[3], nr[4]))
        torch.set_rng_state(checkpoint['torch_rng_state'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_states'])
        print(f'Resume: {state}; LR={optimizer.param_groups[0]["lr"]}', flush=True)
        del checkpoint
    elif any(a.output_root.glob('*.pt')) or (a.output_root / 'metrics.csv').exists():
        raise ValueError('Output already used; resume or choose a new output directory')
    print(f'Device={device}; AMP active={scaler.is_enabled()}; '
          f'width_budget={a.width_budget}; max_batch_size={a.max_batch_size}; '
          'CTC log_probs=float32; lengths=CPU int64; zero_infinity=True', flush=True)
    print(f'PyTorch={torch.__version__}; CUDA={torch.version.cuda}; '
          f'cuDNN={torch.backends.cudnn.version()}; AMP initial/restored scale={scaler.get_scale()}', flush=True)
    stop_requested = [False]
    def request_stop(*_):
        stop_requested[0] = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)

    def save(name):
        nr = np.random.get_state()
        checkpoint = dict(format_version=3, model_state_dict=model.state_dict(),
                          optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                          scaler_state_dict=scaler.state_dict(), progress=dict(state), **state,
                          random_seed=a.seed, python_rng_state=random.getstate(),
                          numpy_rng_state=(nr[0], nr[1].tolist(), nr[2], nr[3], nr[4]),
                          torch_rng_state=torch.get_rng_state(),
                          cuda_rng_states=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                          config=config, char_to_idx=vocabulary.char_to_index,
                          idx_to_char=vocabulary.index_to_char, vocabulary_hash=vocab_hash,
                          manifest_identity=identities, amp_active=scaler.is_enabled(),
                          sampler_state=dict(version=sampler.VERSION, seed=a.seed, pool_size=sampler.pool_size))
        path = a.output_root / name
        temporary = path.with_suffix('.pt.tmp')
        with temporary.open('wb') as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    fields = ['global_step', 'epoch', 'batch_index', 'train_loss', 'learning_rate', 'average_batch_size', 'optimizer_steps_per_second',
              'samples_per_second', 'data_wait_percent', 'compute_percent', 'gpu_peak_memory_mb',
              'val_cer', 'val_exact', 'upper_cer', 'upper_exact', 'count_cer', 'count_exact',
              'lower_cer', 'lower_exact', 'elapsed_minutes', 'timestamp', 'skipped_batches',
              'nonfinite_loss_batches', 'nonfinite_gradient_batches']
    metrics_path = a.output_root / 'metrics.csv'
    exists = metrics_path.exists() and metrics_path.stat().st_size > 0
    # Preserve pre-diagnostic CSV history without appending wider rows to its header.
    csv_fields = fields
    if exists:
        with metrics_path.open(encoding='utf-8', newline='') as previous:
            csv_fields = next(csv.reader(previous))
        if csv_fields not in (fields, fields[:-2]):
            raise ValueError('Unexpected metrics CSV schema')
    invocation_step = state['global_step']
    window = dict(wait=0., compute=0., samples=0, batches=0)
    losses = torch.zeros((), device=device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    window_start = time.perf_counter()
    last_reported_skips = state['skipped_batches']
    with metrics_path.open('a', encoding='utf-8', newline='') as stream, \
            (a.output_root / 'numerical_failures.jsonl').open('a', encoding='utf-8') as failure_stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        def report(validation=None):
            nonlocal losses, window_start, last_reported_skips
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - window_start
            # Window boundary synchronization accounts for outstanding CUDA work.
            compute = max(window['compute'], elapsed - window['wait'])
            measured = max(1e-9, window['wait'] + compute)
            row = dict(global_step=state['global_step'], epoch=state['epoch'], batch_index=state['next_batch'],
                       train_loss=float(losses.cpu()) / max(1, window['batches']),
                       learning_rate=optimizer.param_groups[0]['lr'],
                       average_batch_size=window['samples'] / max(1, window['batches']),
                       optimizer_steps_per_second=window['batches'] / max(elapsed, 1e-9),
                       samples_per_second=window['samples'] / max(elapsed, 1e-9),
                       data_wait_percent=100 * window['wait'] / measured, compute_percent=100 * compute / measured,
                       gpu_peak_memory_mb=torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else 0,
                       elapsed_minutes=(time.perf_counter() - started) / 60,
                       timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), skipped_batches=state['skipped_batches'],
                       nonfinite_loss_batches=state['nonfinite_loss_batches'],
                       nonfinite_gradient_batches=state['nonfinite_gradient_batches'])
            if validation:
                for key in ('train_loss', 'average_batch_size', 'optimizer_steps_per_second',
                            'samples_per_second', 'data_wait_percent', 'compute_percent'):
                    row[key] = ''
                row.update(val_cer=validation['cer'], val_exact=validation['exact_accuracy'])
                for tier, values in validation['per_tier'].items():
                    row[tier + '_cer'] = values['cer']
                    row[tier + '_exact'] = values['exact_accuracy']
            if not window['batches']:
                row['train_loss'] = ''
            writer.writerow(row)
            stream.flush()
            print(json.dumps(row), flush=True)
            last_reported_skips = state['skipped_batches']
            losses.zero_()
            window.update(wait=0., compute=0., samples=0, batches=0)
            window_start = time.perf_counter()
        def failure(kind, batch, lengths, encoded, required, loss, logits, log_probs, amp_scale, norm=None):
            def emit(line):
                print(line, flush=True)
                failure_stream.write(line + '\n')
                failure_stream.flush()
            record_numerical_failure(state, kind, batch, lengths, encoded['target_lengths'], required,
                                     loss, logits, log_probs, amp_scale, norm, a.failure_detail_limit, emit)
            if state['skipped_batches'] >= a.max_nonfinite_batches:
                raise RuntimeError('Non-finite batch threshold reached')
        try:
            while state['global_step'] < a.total_steps and not stop_requested[0]:
                sampler.set_epoch(state['epoch'], state['next_batch'])
                iterator = iter(train_loader)
                model.train()
                while state['next_batch'] < len(sampler.batches):
                    if stop_requested[0] or time.perf_counter() >= deadline:
                        stop_requested[0] = True
                        break
                    tick = time.perf_counter()
                    batch = next(iterator)
                    window['wait'] += time.perf_counter() - tick
                    tick = time.perf_counter()
                    images = batch['images'].to(device, non_blocking=True)
                    encoded = encoder.encode_batch(batch['label_texts'])
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                        logits = model(images)
                    lengths, required = runtime_ctc_lengths(model, batch, encoded, logits.shape[0])
                    loss, log_probs = fp32_ctc_loss(logits, encoded['targets'].to(device, non_blocking=True),
                                                   lengths, encoded['target_lengths'], criterion)
                    state['next_batch'] += 1
                    if not bool(torch.isfinite(loss)):
                        failure('loss', batch, lengths, encoded, required, loss, logits, log_probs, scaler.get_scale())
                        # No backward/unscale occurred, so there is no scaler inf-check to update.
                        window['compute'] += time.perf_counter() - tick
                        continue
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    finite_grad = bool(torch.isfinite(norm))
                    if finite_grad:
                        scaler.step(optimizer)
                    elif state['numerical_failure_details'] < a.failure_detail_limit:
                        # Capture the pre-update scale only on a detailed failure.
                        failed_scale = scaler.get_scale()
                    scaler.update()
                    if not finite_grad:
                        failure('gradient', batch, lengths, encoded, required, loss, logits, log_probs,
                                failed_scale if state['numerical_failure_details'] < a.failure_detail_limit else None, norm)
                        window['compute'] += time.perf_counter() - tick
                        continue
                    scheduler.step()
                    state['global_step'] += 1
                    losses += loss.detach()
                    window['batches'] += 1
                    window['samples'] += len(batch['label_texts'])
                    window['compute'] += time.perf_counter() - tick
                    step = state['global_step']
                    validation = None
                    validate = not a.benchmark_steps and step % a.validate_every_steps == 0
                    if step % a.log_every_steps == 0 or validate:
                        report()
                    if validate and time.perf_counter() < deadline:
                        class GuardedValidation:
                            def __len__(self):
                                return len(val_loader)
                            def __iter__(self):
                                for batch in val_loader:
                                    if stop_requested[0] or time.perf_counter() >= deadline:
                                        raise TimeoutError('Validation stopped for clean shutdown')
                                    yield batch
                        try:
                            validation = evaluate(model, GuardedValidation(), encoder, criterion, vocabulary, device,
                                                  'fast validation', validate_ctc_runtime=True)
                        except TimeoutError:
                            stop_requested[0] = True
                            break
                        print_evaluation('Fast validation', validation)
                        if validation['cer'] < state['best_val_cer']:
                            state.update(best_val_cer=validation['cer'], stale_validations=0)
                            save('best.pt')
                        else:
                            state['stale_validations'] += 1
                        report(validation)
                        save('last.pt')
                        window_start = time.perf_counter()
                        model.train()
                        if state['stale_validations'] >= 6:
                            stop_requested[0] = True
                    if step % a.checkpoint_every_steps == 0:
                        if window['batches']:
                            report()
                        save('last.pt')
                        window_start = time.perf_counter()
                    if step >= a.total_steps or (a.stop_after_steps and step - invocation_step >= a.stop_after_steps):
                        stop_requested[0] = True
                    if stop_requested[0]:
                        break
                if state['next_batch'] == len(sampler.batches):
                    state.update(epoch=state['epoch'] + 1, next_batch=0)
        finally:
            if window['batches'] or state['skipped_batches'] != last_reported_skips:
                report()
            save('last.pt')
    print(f'Clean stop at optimizer step {state["global_step"]}', flush=True)


if __name__ == '__main__':
    main()
