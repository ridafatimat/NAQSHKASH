"""Stage 2F I/O. Historical integration is read-only; no import starts a job."""
import csv
import hashlib
import importlib.util
import json
import sys
import zipfile
import atexit
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(r'C:\NAQSHKASH')
DATA = Path(r'C:\FYP Dataset Generation\datasets')
D = PROJECT / 'stage2/stage2d_experiments'
E = PROJECT / 'stage2/stage2e_finetuning'
TOOLS = Path(r'C:\FYP Dataset Generation\dataset_tools')
CHECKPOINT = E / 'checkpoints/selected_model/best.pt'
CHECKPOINT_SHA = '27f2da50795e2fca8f327afefdce10645f3e60e5320060a8cee93994fabdcdf1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_archives = {}


def close_archives():
    for archive in _archives.values():
        archive.close()
    _archives.clear()


atexit.register(close_archives)


def asset_bytes(path):
    """An archive reference is absolute archive path :: member, never extracted."""
    if '::' in path:
        archive, member = path.split('::', 1)
        if archive not in _archives:
            _archives[archive] = zipfile.ZipFile(archive)
        return _archives[archive].read(member)
    return Path(path).read_bytes()


def text_asset(path):
    return asset_bytes(path).decode('utf-8-sig').replace('\r\r\n', '\n').replace('\r\n', '\n').replace('\r', '\n')


def new_output(path):
    p = Path(path)
    if not p.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Experiment outputs must stay inside Stage 2F')
    p.mkdir(parents=True, exist_ok=False)
    return p


def verify_integrity():
    lock = load_json(ROOT / 'configs/historical_lock.json')
    bad = [p for p, h in lock.items() if not Path(p).is_file() or digest(p) != h]
    if bad:
        raise ValueError('Historical input drift: ' + repr(bad))
    return len(lock)
