"""Stage 2D read-only integration; writes stay under this module."""
import csv,hashlib,importlib.util,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
C=ROOT.parent/'preprocessing_improvements'
sys.path.insert(0,str(C))
import core as old
from pipeline import Processor
import cv2,numpy as np
cv2.setNumThreads(1)
reference=old.load_local('stage2d_offline_reference','evaluation/reference.py')
def read(path):return old.read_csv(path)
def dump(path,value):old.write_json(ROOT/path,value)
def csvout(path,rows,fields=None):old.write_csv(ROOT/path,rows,fields)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
class Sources:
 def __init__(self):
  self.z={d:{k:zipfile.ZipFile(v) for k,v in [('clean',s['clean']),('scanned',s['scanned'])] if v} for d,s in old.SOURCE_CONFIG.items()};self.cache={}
 def get(self,p):
  d=p['dataset'];k=(d,p['parent_id']);s=old.SOURCE_CONFIG[d]
  if k not in self.cache:
   raw=self.z[d]['clean'].read(s['prefix']+p['source_image'])
   assert hashlib.sha256(raw).hexdigest()==p['source_sha256']
   clean=cv2.imdecode(np.frombuffer(raw,np.uint8),0)
   data=json.loads(self.z[d]['clean'].read(s['prefix']+p['source_json']))
   labels,entries=reference.clean_reference(clean,data['logical_rows'])
   self.cache[k]=(labels,entries)
  labels,entries=self.cache[k]
  raw=self.z[d][p['kind']].read(p['image_member']);assert hashlib.sha256(raw).hexdigest()==p['image_sha256']
  image=cv2.imdecode(np.frombuffer(raw,np.uint8),0)
  ref=labels if p['kind']=='clean' else reference.augmented_reference(labels,json.loads(p['parameters']),image.shape)
  return image,ref,entries
def pages():return read(C/'manifests/pages.csv')
def key(p):return (p['dataset'],p['case_id'])
def snapshot():
 inv={}
 for p in old.PROJECT.rglob('*'):
  if p.is_file() and not p.is_relative_to(ROOT) and not {'.git','__pycache__','.pytest_cache'}.intersection(p.parts):inv[str(p)]=old.digest(p)
 return inv
def check_history(expected):
 bad=[p for p,h in expected.items() if not Path(p).is_file() or old.digest(p)!=h]
 return dict(checked=len(expected),changed=bad,passed=not bad)

