"""Read-only Stage 1 compatibility and archive inventory; writes only Stage 2 reports."""
import argparse
import ctypes
import hashlib
import json
import os
import platform
import sys
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE1 = HERE.parents[1]
sys.path.insert(0, str(STAGE1))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def archive_inventory(path):
    path = Path(path)
    if not path.is_file():
        return {'path': str(path), 'status': 'missing'}
    with zipfile.ZipFile(path) as archive:
        files = [i for i in archive.infolist() if not i.is_dir()]
        result = dict(path=str(path), sha256=sha256(path), files=len(files),
                      extensions=dict(Counter(Path(i.filename).suffix.lower() for i in files)),
                      members=[dict(name=i.filename, bytes=i.file_size) for i in files])
        # Read actual small text artifacts, never extract or execute archive code.
        result['text_previews'] = {}
        for item in files:
            if Path(item.filename).suffix.lower() in {'.json', '.csv', '.txt', '.md'} and item.file_size < 100000:
                result['text_previews'][item.filename] = archive.read(item).decode('utf-8-sig', errors='replace')[:12000]
        return result


def hardware(torch):
    info = dict(platform=platform.platform(), python=sys.version, cpu=platform.processor(),
                logical_cpus=os.cpu_count(), torch=torch.__version__, cuda_available=torch.cuda.is_available())
    if os.name == 'nt':
        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ('total', 'available', 'page_total', 'page_available',
                                                       'virtual_total', 'virtual_available', 'extended')]
        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            info.update(ram_total_bytes=memory.total, ram_available_bytes=memory.available)
    info['gpus'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def check_checkpoint(path, vocabulary_path):
    import torch
    from models.crnn import CRNN
    from training.manifest_training import load_vocabulary
    from data.collate import talim_collate_fn
    from data.ctc_decode import greedy_ctc_decode
    vocabulary, digest = load_vocabulary(vocabulary_path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['char_to_idx'] != vocabulary.char_to_index:
        raise ValueError('Checkpoint vocabulary mapping mismatch')
    if checkpoint.get('vocabulary_hash', digest) != digest:
        raise ValueError('Checkpoint vocabulary hash mismatch')
    model = CRNN(num_classes=66, hidden_size=256, num_lstm_layers=2)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    sample = dict(image=torch.ones(1, 32, 64), label_text='A', sample_id='probe')
    batch = talim_collate_fn([sample])
    with torch.inference_mode():
        logits = model(batch['images'])
    assert tuple(logits.shape) == (32, 1, 66)
    assert greedy_ctc_decode([0, 8, 8, 0, 8], vocabulary) == 'AA'
    return dict(path=str(Path(path).resolve()), sha256=sha256(path), strict_load=True,
                vocabulary_hash=digest, characters=65, classes=66, blank=0,
                hidden_size=256, bidirectional=True, lstm_layers=2, input_height=64,
                normalization='grayscale float32 / 255; white right padding',
                resize='Stage 1 talim_collate_fn: bilinear, align_corners=False, round(W*64/H)',
                target_policy='Stage 1 tier dataset removes leading spaces, preserves raw label and internal spaces',
                decoding='Stage 1 greedy CTC collapse; truncate to true output width',
                probe_logits_shape=list(logits.shape), global_step=checkpoint.get('global_step'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--archives', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, default=HERE / 'reports/results/local_inspection_current.json')
    a = p.parse_args()
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    result = dict(hardware=hardware(torch),
                  checkpoint=check_checkpoint(a.checkpoint, STAGE1 / 'config/vocabulary.json'),
                  archives=[archive_inventory(path) for path in a.archives])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'archives'}, indent=2))
    print('Archive inventories:', [(r['path'], r.get('files', r.get('status'))) for r in result['archives']])


if __name__ == '__main__':
    main()
