"""Isolated experiment paths and read-only access to the frozen Stage2F API."""
import csv
import hashlib
import json
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
LEGACY = Path(r'C:\NAQSHKASH\stage2\stage2f_negative_rejection')
BASELINE = LEGACY / 'metrics/20261106_154728_812_dev_baseline'
sys.dont_write_bytecode = True
sys.path.insert(0, str(LEGACY / 'src'))
from common import digest, load_json, read_csv, write_json, write_csv, asset_bytes, text_asset, PROJECT, D, E, CHECKPOINT
from talim_validator import validate_tiers, rules
from rejection_pipeline import decide


def allowed_characters(role):
    if role == 'upper':
        return set(rules.EXCLUSIVE_UP_SYMBOLS) | {' '}
    if role == 'lower':
        return set(rules.EXCLUSIVE_DOWN_SYMBOLS) | {' '}
    if role == 'count':
        return set(''.join(rules.encode_count(n) for n in range(1, 260))) | {' '}
    return set()


def new_output(path):
    p = Path(path).resolve()
    if not p.is_relative_to(WORKSPACE.resolve()):
        raise ValueError('Experiment outputs must stay in the isolated tier-aware workspace')
    p.mkdir(parents=True, exist_ok=False)
    return p


def serialize_csv(path, rows):
    rows = list(rows)
    fields = sorted({k for row in rows for k in row})
    flat = [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows]
    write_csv(path, flat, fields or ['image_id'])


def verify_integrity():
    protected = load_json(WORKSPACE / 'configs/historical_lock.json')
    changed = [p for p, h in protected.items() if not Path(p).is_file() or digest(p) != h]
    if changed:
        raise ValueError('Historical input drift: ' + repr(changed))
    frozen = load_json(WORKSPACE / 'configs/experiment_lock.json')
    changed = [p for p, h in frozen.items() if digest(WORKSPACE / p) != h]
    if changed:
        raise ValueError('Tier-aware protocol/code drift: ' + repr(changed))
    return dict(historical_files=len(protected), experiment_files=len(frozen), unchanged=True)


def load_registry():
    path = WORKSPACE / 'manifests/verified_role_ambiguities.csv'
    summary = load_json(WORKSPACE / 'reports/glyph_audit_summary.json')
    if digest(path) != summary['mapping_sha256']:
        raise ValueError('Verified ambiguity mapping drift')
    rows = read_csv(path)
    for row in rows:
        if row['status'] != 'verified_exact_in_scope':
            raise ValueError('Unverified mapping entry')
        row['verified_sizes'] = json.loads(row['verified_sizes'])
    return rows
