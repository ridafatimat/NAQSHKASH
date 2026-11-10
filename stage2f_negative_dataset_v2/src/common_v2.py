"""Read-only canonical integration and isolated dataset-v2 I/O. No model imports."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(r'C:\NAQSHKASH')
TOOLS = Path(r'C:\FYP Dataset Generation\dataset_tools')
DATA = Path(r'C:\FYP Dataset Generation\datasets')
FONT = Path(r'C:\FYP Dataset Generation\Assets\Talamat.ttf')
sys.dont_write_bytecode = True
sys.path.insert(0, str(TOOLS))
import talim_rules as rules
from talim.generator import generate_talim
from talim.renderer import render_talim_text, _cell_metrics
from validate_datasets import check_sample
from PIL import Image, ImageFont

ROLES = ('upper', 'count_line', 'lower')
RULES = {'wrong_tier': 'CANON_SYMBOL_DIRECTION', 'invalid_count_token': 'CANON_COUNT_ENCODING',
         'missing_count': 'CANON_REQUIRED_SYMBOL_COUNT'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):h.update(b)
    return h.hexdigest()


def hash_object(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, obj):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)


def write_csv(path, rows):
    rows=list(rows);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['image_id'])
        w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def new_output(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):raise ValueError('Outputs must stay in dataset-v2 workspace')
    path.mkdir(parents=True,exist_ok=False)
    return path


def displayed_rows(sample):
    return [dict(zip(ROLES,rules.canonical_row(row['runs']))) for row in sample['logical_rows']]


def source_specs():
    return [dict(name='stage1_100000',path=r'C:\FYP Dataset Generation\Final dataset\talim_100000.zip',prefix='datasets/talim_100000/'),
            dict(name='general_10000',path=str(DATA/'talim_10000.zip'),prefix='talim_10000/'),
            dict(name='positive_edges',path=str(DATA/'talim_edge_cases_positive_v1.zip'),prefix='talim_edge_cases_positive_v1/')]


def verify_historical():
    lock=read_json(ROOT/'configs/historical_lock.json')
    changed=[p for p,h in lock.items() if not Path(p).is_file() or digest(p)!=h]
    if changed:raise ValueError('Historical drift: '+repr(changed))
    return len(lock)


def verify_design():
    lock=read_json(ROOT/'configs/design_lock.json')
    changed=[p for p,h in lock.items() if digest(ROOT/p)!=h]
    if changed:raise ValueError('Frozen generation design drift: '+repr(changed))
    return len(lock)
