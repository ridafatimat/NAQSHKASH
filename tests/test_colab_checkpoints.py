import json
from pathlib import Path
import subprocess
import sys
import re
import zipfile

import pytest

from training.colab_checkpoints import backup, mirror_checkpoint, restore, quick_dataset


def test_mirror_bundle_is_consistent_and_restorable(tmp_path):
    active, persistent = tmp_path / 'active', tmp_path / 'drive'
    active.mkdir()
    (active / 'last.pt').write_bytes(b'last1')
    (active / 'best.pt').write_bytes(b'best1')
    mirror_checkpoint(active / 'best.pt', persistent)
    mirror_checkpoint(active / 'last.pt', persistent)
    (active / 'best.pt').write_bytes(b'best2')
    mirror_checkpoint(active / 'best.pt', persistent)
    # A runtime loss before the corresponding last save retains the old pair.
    restored = tmp_path / 'restored'
    restore(persistent / 'checkpoints.zip', restored)
    assert (restored / 'last.pt').read_bytes() == b'last1'
    assert (restored / 'best.pt').read_bytes() == b'best1'
    with pytest.raises(FileExistsError):
        restore(persistent / 'checkpoints.zip', restored)
    (active / 'last.pt').write_bytes(b'last2')
    mirror_checkpoint(active / 'last.pt', persistent)
    restore(persistent / 'checkpoints.zip', tmp_path / 'new')
    assert (tmp_path / 'new/best.pt').read_bytes() == b'best2'
    assert (persistent / 'last.pt').read_bytes() == b'last2'
    assert not list(persistent.glob('*.tmp'))


def test_mirror_failure_keeps_previous_backup_and_local(tmp_path, monkeypatch):
    active = tmp_path / 'active'
    active.mkdir()
    (active / 'last.pt').write_bytes(b'valid')
    persistent = tmp_path / 'drive'
    mirror_checkpoint(active / 'last.pt', persistent)
    previous = (persistent / 'checkpoints.zip').read_bytes()
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(zipfile.ZipFile, 'write', fail)
    with pytest.warns(RuntimeWarning, match='PERSISTENCE FAILED'):
        mirror_checkpoint(active / 'last.pt', persistent)
    assert (persistent / 'checkpoints.zip').read_bytes() == previous
    assert (active / 'last.pt').read_bytes() == b'valid'
    with pytest.raises(FileNotFoundError):
        backup(tmp_path, tmp_path / 'missing.zip')


def test_restore_rejects_unsafe_archive(tmp_path):
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as output:
        output.writestr('last.pt', b'ok')
        output.writestr('../escape', b'bad')
    with pytest.raises(ValueError):
        restore(archive, tmp_path / 'output')


def test_quick_dataset_does_not_read_unselected_rows(tmp_path):
    (tmp_path / 'i.png').touch()
    (tmp_path / 'l.txt').write_text(' A')
    (tmp_path / 'metadata.csv').write_text(
        'tier_id,row_id,parent_sample_id,tier_name,image_path,label_path\n'
        '1,1,1,upper,i.png,l.txt\n2,2,2,upper,MISSING,MISSING\n')
    dataset = quick_dataset(tmp_path, 1)
    assert len(dataset) == 1
    with pytest.raises(FileNotFoundError):
        quick_dataset(tmp_path, 2)


def test_colab_commands_match_real_cli_and_full_is_display_only():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / 'colab/NAQSHKASH_Stage1_Training.ipynb').read_text())
    sources = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    for source in sources:
        compile(source, '<colab>', 'exec')
    namespace = dict(sys=sys, Path=Path, PROJECT_ROOT=root, DATASET_ROOT=Path('/dataset'),
                     OUTPUT_ROOT=Path('/output'), PERSISTENT_CKPT_ROOT=Path('/drive'),
                     WIDTH_BUDGET=16384, MAX_BATCH_SIZE=32, WORKERS=2)
    exec(next(s for s in sources if 'def command(' in s), namespace)
    help_text = subprocess.check_output([sys.executable, '-m', 'training.train_crnn', '--help'], text=True)
    flags = set(re.findall(r'--[a-z-]+', help_text))
    for cmd in namespace['plan'].values():
        assert {v for v in cmd if v.startswith('--')} <= flags
    for name in ('full', 'resume'):
        assert '--train-limit' not in namespace['plan'][name]
        assert '--skip-test' not in namespace['plan'][name]
    text = '\n'.join(sources)
    assert "run_short('full')" not in text and "run_short('resume')" not in text
    assert '--checkpoint-mirror-dir' in namespace['plan']['full']
