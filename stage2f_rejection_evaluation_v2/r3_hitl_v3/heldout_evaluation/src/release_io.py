"""Authorized adapter utilities; never import working pipeline modules."""
import hashlib,json,sys
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
RELEASE=HERE.parent/'frozen_release'
sys.dont_write_bytecode=True
sys.path.insert(0,str(RELEASE))
from verify_freeze import verify,load,sha,canonical


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def output_path(path):
    p=Path(path).resolve()
    if not p.is_relative_to((HERE/'runs').resolve()) or p==(HERE/'runs').resolve():raise ValueError('Outputs must be new directories within heldout_evaluation/runs')
    return p


def check_completion(path,kind=None):
    path=output_path(path);done=load(path/'completion.json')
    if not done.get('complete') or (kind and done.get('kind')!=kind):raise ValueError('Wrong/incomplete completed run')
    for name,h in done['outputs'].items():
        p=(path/name).resolve()
        if not p.is_relative_to(path) or sha(p)!=h:raise ValueError('Completed run output drift: '+name)
    if done['release_sha256']!=sha(RELEASE/'release_manifest.json'):raise ValueError('Run uses another frozen release')
    return done
