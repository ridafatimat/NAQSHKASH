"""Isolated adapters for the existing frozen D1/E1, T1 and F1 v2 APIs."""
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(r'C:\NAQSHKASH')
F1=PROJECT/'stage2/stage2f_negative_rejection'
T1=PROJECT/'stage2/stage2f_tier_aware_decoding'
V2=PROJECT/'stage2/stage2f_negative_dataset_v2'
CHECKPOINT=PROJECT/'stage2/stage2e_finetuning/checkpoints/selected_model/best.pt'
CHECKPOINT_SHA='27f2da50795e2fca8f327afefdce10645f3e60e5320060a8cee93994fabdcdf1'
sys.dont_write_bytecode=True
# The T1 runtime internally imports the original F1 `common` module. Keeping
# its own module names avoids accidentally resolving this workspace's evaluator.
sys.path.insert(0,str(T1/'src'))
import runtime as tier_runtime
from contextual_pipeline import process as contextual_process
from measurements import page_errors,assess_corrections
from talim_validator import validate_tiers,rules
sys.path.insert(0,str(ROOT/'src'))  # keep the new evaluator name distinct from legacy evaluate_rejection.py


def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)


def write_csv(path,rows):
    rows=list(rows);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted({k for r in rows for k in r}) or ['image_id']
    with path.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def new_output(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):raise ValueError('Outputs must stay inside the new evaluation workspace')
    path.mkdir(parents=True,exist_ok=False);return path


def verify_integrity():
    historical=load_json(ROOT/'configs/historical_lock.json')
    bad=[p for p,h in historical.items() if not Path(p).is_file() or digest(p)!=h]
    if bad:raise ValueError('Historical input drift: '+repr(bad))
    frozen=load_json(ROOT/'configs/experiment_lock.json')
    bad=[p for p,h in frozen.items() if digest(ROOT/p)!=h]
    if bad:raise ValueError('Evaluation code/config drift: '+repr(bad))
    return dict(historical_files=len(historical),experiment_files=len(frozen),unchanged=True)


def runtime_rendering_context(row):
    """Whitelisted provenance ONLY: no validity, source glyphs, rules or truth."""
    layout=row.get('layout',{})
    profile=load_json(T1/'configs/rendering_profile.json')
    if (layout.get('font_sha256')!=profile['font_sha256'] or layout.get('font_size_px')!=48
            or layout.get('convention')!='existing Talamat grayscale integer lt grid; matched parent layout is frozen'):
        return {}
    recipe=row.get('augmentation_recipe')
    if recipe is not None:
        ops=recipe.get('operations',[])
        if (recipe.get('padding')!=0 or len(ops)!=1 or ops[0].get('type')!='brightness'
                or ops[0].get('factor') not in (0.92,0.96) or recipe.get('output_format')!='PNG'):
            return {}
    return dict(profile,source='frozen_dataset_v2_renderer_contract_and_layout',transformed_variant=recipe is not None)


def apply_candidate(observation,rendering_context,candidate):
    if candidate not in ('R0','R1'):raise ValueError('Candidate must be R0 or R1')
    return contextual_process(observation,dict(enabled=candidate=='R1'),tier_runtime.load_registry(),rendering_context)


def make_pixel_pipeline(device):
    # Lazy: importing/testing evaluators never loads torch or a model.
    spec=importlib.util.spec_from_file_location('rejection_v2_original_pixel_pipeline',F1/'src/evaluate_baseline.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module.FrozenPipeline(device)


def pipeline_exception(exc):
    return dict(preprocessing_status='pipeline_error',segmentation_accepted=False,segmentation_reason='unlocalized D1/E1 exception',
        segmentation_rows=[],diagnostics={},ocr_status='error',decoded_rows=[],tiers=[],confidence={},
        pipeline_exception=dict(type=type(exc).__name__,message=str(exc),stage='D1_E1_observe_unlocalized'))
