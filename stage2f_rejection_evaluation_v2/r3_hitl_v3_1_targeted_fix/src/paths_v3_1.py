"""Read-only integration. Importing this module never creates a recognizer."""
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
PROJECT=Path(r'C:\NAQSHKASH')
WORKSPACE=PROJECT/'stage2/stage2f_rejection_evaluation_v2'
R3=WORKSPACE/'r3_hitl_v3'
D1=PROJECT/'stage2/stage2d_experiments'
sys.dont_write_bytecode=True
sys.path.insert(0,str(R3/'src'))
from policy_v3 import evidence,validate_aligned_row
from integration import load_json,digest,content_hash,write_json,write_csv
ID='R3_HITL_V3_1'


def config():
    c=load_json(HERE/'configs/policy_v3_1.json')
    if set(c)!={'policy_id','implementation_revision','r3_run','offset_tolerance_columns','fallback_threshold','minimum_pitch_observations','raster_uncertainty_pixels','read_only_sha256'}:raise ValueError('Unknown configuration field')
    if c['implementation_revision']!='targeted_pitch_fix_1':raise ValueError('Unknown targeted revision')
    if c['policy_id']!=ID or c['offset_tolerance_columns']!=0.5 or c['fallback_threshold']!=128 or c['minimum_pitch_observations']!=3 or c['raster_uncertainty_pixels']!=1:
        raise ValueError('Unsupported fixed measurement configuration')
    return c


def verify_locks(c):
    for path,h in c['read_only_sha256'].items():
        if digest(path)!=h:raise ValueError('Read-only input drift: '+path)


def own_hashes():
    return {p.relative_to(HERE).as_posix():digest(p) for folder in ('src','tests','configs','reports') for p in sorted((HERE/folder).glob('*')) if p.is_file()} | {'README.md':digest(HERE/'README.md')}
