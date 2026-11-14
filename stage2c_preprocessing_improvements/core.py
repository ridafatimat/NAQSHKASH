"""Local Stage2C paths and read-only reference integration."""
import csv,hashlib,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
B=PROJECT/'stage2/scanned_robustness'
A=PROJECT/'stage2/positive_edge_cases'
sys.path.insert(0,str(PROJECT));sys.path.append(str(B));sys.path.append(str(A))
CHECKPOINT=A/'checkpoints/stage1/best.pt'
VOCABULARY=PROJECT/'config/vocabulary.json'
from modes import preprocess as baseline_preprocess,blind_tiers as baseline_tiers
from evaluate import edit_counts,aggregate
from training.manifest_training import load_vocabulary
DATA=Path(r'C:\FYP Dataset Generation\datasets')
SOURCE_CONFIG={
 'positive':dict(clean=str(DATA/'talim_edge_cases_positive_v1.zip'),scanned=str(DATA/'talim_edge_cases_positive_scanned_v1.zip'),prefix='talim_edge_cases_positive_v1/'),
 'general':dict(clean=str(DATA/'talim_10000.zip'),scanned=str(DATA/'talim_10000_scanned.zip'),prefix='talim_10000/'),
 'normal':dict(clean=r'C:\FYP Dataset Generation\Final dataset\talim_100000.zip',scanned='',prefix='datasets/talim_100000/')}
CANDIDATES={
 'C0':dict(normalize=False,robust=False,geometry=False,conservative=False),
 'C1':dict(normalize=True,robust=False,geometry=False,conservative=False),
 'C2':dict(normalize=False,robust=True,geometry=False,conservative=False),
 'C3':dict(normalize=False,robust=False,geometry=True,conservative=False),
 'C4':dict(normalize=True,robust=True,geometry=True,conservative=False),
 'C5':dict(normalize=True,robust=True,geometry=True,conservative=True)}

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def read_csv(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write_csv(path,rows,fields=None):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)

def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')

def load_local(name,path):
 spec=importlib.util.spec_from_file_location(name,HERE/path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def protected():
 return {str(p.relative_to(PROJECT)):digest(p) for p in PROJECT.rglob('*') if p.is_file() and not p.is_relative_to(HERE) and not {'.git','__pycache__','.pytest_cache'}.intersection(p.relative_to(PROJECT).parts)}

def verify_lock(code=False):
 lock=json.loads((HERE/'protocol/lock.json').read_text())
 for file,expected in lock['locked_files'].items():
  if digest(file)!=expected:raise ValueError('Frozen source/protocol changed: '+file)
 if code:
  for file,expected in json.loads((HERE/'protocol/software_lock.json').read_text()).items():
   if digest(file)!=expected:raise ValueError('Experiment code changed: '+file)
 return lock
