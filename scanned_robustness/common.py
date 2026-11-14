"""Stage2B shared paths, immutable-source helpers and deterministic metrics."""
import csv,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
POSITIVE=PROJECT/'stage2/positive_edge_cases'
sys.path.insert(0,str(PROJECT))
sys.path.append(str(POSITIVE))
from evaluate import edit_counts,aggregate
from training.manifest_training import load_vocabulary
CHECKPOINT=POSITIVE/'checkpoints/stage1/best.pt'
VOCABULARY=PROJECT/'config/vocabulary.json'
DATA=Path(r'C:\FYP Dataset Generation\datasets')
STAGE1_ZIP=Path(r'C:\FYP Dataset Generation\Final dataset\talim_100000.zip')
DATASETS={
 'general':dict(scanned=str(DATA/'talim_10000_scanned.zip'),clean=str(DATA/'talim_10000.zip'),prefix='talim_10000_scanned/',clean_prefix='talim_10000/'),
 'positive':dict(scanned=str(DATA/'talim_edge_cases_positive_scanned_v1.zip'),clean=str(DATA/'talim_edge_cases_positive_v1.zip'),prefix='talim_edge_cases_positive_scanned_v1/',clean_prefix='talim_edge_cases_positive_v1/')}

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def semantic(data):
 rows=[[(r['symbol'],int(r['count']),r['direction']) for r in row['runs']] for row in data['logical_rows']]
 return hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()

def write_csv(path,rows,fields=None):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]));writer.writeheader();writer.writerows(rows)

def read_csv(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def eligible(stage1_splits,positive_split=None,identity_verified=True):
 splits=set(stage1_splits)
 if splits.intersection({'train','validation'}):return 'excluded_seen_or_validation'
 if positive_split is not None:return 'positive_frozen_test' if positive_split=='test' else 'excluded_stage2a_'+positive_split
 if splits=={'test'}:return 'general_stage1_test' if identity_verified else 'general_test_structure_match_exploratory'
 return 'general_unassigned_exploratory'

def metrics(rows):
 return aggregate(rows) if rows else dict(samples=0,characters=0,exact_matches=0,cer=None,exact_accuracy=None,substitutions=0,deletions=0,insertions=0)

def protected_snapshot():
 result={}
 for p in PROJECT.rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(PROJECT)
  if p.is_relative_to(HERE) or {'.git','__pycache__','.pytest_cache'}.intersection(rel.parts):continue
  result[str(rel)]=sha256(p)
 return result
