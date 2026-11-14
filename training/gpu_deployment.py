"""Small, side-effect-free helpers shared by the Kaggle and Colab notebooks."""
from pathlib import Path
import shlex
import sys


def verify_layout(project_root, dataset_root):
    project, dataset = Path(project_root), Path(dataset_root)
    required = [project / name for name in ('training', 'models', 'data', 'tests')]
    required += [project / 'training/train_crnn.py']
    for split in ('train', 'validation', 'test'):
        required += [dataset / split / name for name in ('images', 'labels', 'metadata.csv')]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError('Check PROJECT_ROOT / DATASET_ROOT. Missing:\n' + '\n'.join(missing))


def commands(dataset_root, output_root, width_budget=16384, max_batch_size=32, workers=2):
    """Full commands are returned for display only, never executed here."""
    root = Path(output_root)
    base = [sys.executable, '-m', 'training.train_crnn', '--dataset-root', str(dataset_root),
            '--width-budget', str(width_budget), '--max-batch-size', str(max_batch_size),
            '--amp', '--num-workers', str(workers)]
    smoke_root = root / 'resume_test'
    smoke = base + ['--output-root', str(smoke_root), '--epochs', '2',
                    '--train-limit', '4096', '--val-limit', '256', '--skip-test',
                    '--checkpoint-every-batches', '10', '--stop-after-batches', '20']
    full = base + ['--output-root', str(root / 'full_run'), '--epochs', '30',
                   '--checkpoint-every-batches', '1000']
    return {
        'preflight': base + ['--output-root', str(root / 'quick_probe'),
                            '--train-limit', '4096', '--val-limit', '256', '--preflight-only'],
        'interrupt': smoke,
        'resume_test': smoke + ['--resume', str(smoke_root / 'last.pt')],
        # Mandatory before full training: no TRAIN limit, so outlier widths are covered.
        'full_width_probe': base + ['--output-root', str(root / 'full_width_probe'),
                                   '--val-limit', '256', '--preflight-only'],
        'full': full,
        'resume': full + ['--resume', str(root / 'full_run/last.pt')],
    }


def display_command(command):
    return ' '.join(shlex.quote(str(part)) for part in command)


def inspect_checkpoints(directory):
    import torch
    directory = Path(directory)
    last = None
    for name in ('last.pt', 'best.pt'):
        path = directory / name
        if not path.exists():
            print(f'{name}: absent (best.pt appears only after validation improvement)')
            continue
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        print(f'{name}: {path.stat().st_size / 2**20:.1f} MiB')
        print('keys:', sorted(checkpoint))
        print({key: checkpoint.get(key) for key in ('epoch', 'next_batch', 'global_step')})
        print('AMP requested:', checkpoint['config'].get('amp'),
              'scaler state:', checkpoint.get('scaler_state_dict'))
        if name == 'last.pt':
            # Return only small diagnostics, rather than retaining model/Adam tensors in notebook RAM.
            last = {key: checkpoint[key] for key in ('epoch', 'next_batch', 'global_step')}
            last['scaler_active'] = bool(checkpoint.get('scaler_state_dict'))
    return last
