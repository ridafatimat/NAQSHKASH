import json
from pathlib import Path
import re
import subprocess
import sys

import pytest
import torch

from training.gpu_deployment import commands, display_command, inspect_checkpoints, verify_layout

ROOT = Path(__file__).resolve().parents[1]


def test_commands_match_cli_and_keep_full_separate():
    help_text = subprocess.check_output([sys.executable, '-m', 'training.train_crnn', '--help'], text=True)
    flags = set(re.findall(r'--[a-z-]+', help_text))
    plan = commands('/input/stage 1', '/working/run')
    for command in plan.values():
        assert {item for item in command if item.startswith('--')} <= flags
        assert '--amp' in command
    assert '--preflight-only' in plan['preflight']
    assert '--train-limit' not in plan['full_width_probe']
    assert '--preflight-only' in plan['full_width_probe']
    for name in ('full', 'resume'):
        assert '--train-limit' not in plan[name]
        assert '--val-limit' not in plan[name]
        assert '--skip-test' not in plan[name]
        assert '--stop-after-batches' not in plan[name]
        assert plan[name][plan[name].index('--epochs') + 1] == '30'
    assert plan['resume_test'][:-2] == plan['interrupt']
    assert plan['resume_test'][-2:] == ['--resume', str(Path('/working/run') / 'resume_test/last.pt')]
    assert "'/input/stage 1'" in display_command(plan['preflight'])


def test_layout_and_checkpoint_inspection(tmp_path, capsys):
    with pytest.raises(FileNotFoundError, match='PROJECT_ROOT'):
        verify_layout(tmp_path, tmp_path / 'dataset')
    for name in ('training', 'models', 'data', 'tests'):
        (tmp_path / name).mkdir()
    (tmp_path / 'training/train_crnn.py').touch()
    for split in ('train', 'validation', 'test'):
        for name in ('images', 'labels'):
            (tmp_path / 'dataset' / split / name).mkdir(parents=True)
        (tmp_path / 'dataset' / split / 'metadata.csv').touch()
    verify_layout(tmp_path, tmp_path / 'dataset')
    assert inspect_checkpoints(tmp_path) is None
    torch.save(dict(epoch=0, next_batch=20, global_step=20, config={'amp': True},
                    scaler_state_dict={'scale': 65536.}), tmp_path / 'last.pt')
    report = inspect_checkpoints(tmp_path)
    assert report == dict(epoch=0, next_batch=20, global_step=20, scaler_active=True)
    output = capsys.readouterr().out
    assert 'best.pt: absent' in output and 'keys:' in output and 'MiB' in output


@pytest.mark.parametrize('filename', ['NAQSHKASH_Stage1_Kaggle.ipynb'])
def test_notebook_code_compiles_and_full_training_is_display_only(filename):
    notebook = json.loads((ROOT / 'colab' / filename).read_text(encoding='utf-8'))
    source = ''
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        text = ''.join(cell['source'])
        compile(text, f'{filename}:{index}', 'exec')
        assert cell['execution_count'] is None and cell['outputs'] == []
        source += text + '\n'
    for name in ('preflight', 'interrupt', 'resume_test', 'full_width_probe'):
        assert f"run_short('{name}')" in source
    assert "run_short('full')" not in source and "run_short('resume')" not in source
    assert "assert name in ('preflight', 'interrupt', 'resume_test', 'full_width_probe')" in source
    assert "CUDA_VISIBLE_DEVICES" in source
    assert "assert torch.cuda.is_available()" in source
    assert "after['global_step'] == before['global_step'] + 20" in source
