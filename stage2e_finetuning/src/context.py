"""Isolated Stage 2E paths and read-only historical integration."""
import csv,hashlib,json,sys,zipfile,random,time
from pathlib import Path
E=Path(__file__).resolve().parents[1]
PROJECT=E.parents[1];D=E.parent/'stage2d_experiments';C=E.parent/'preprocessing_improvements'
# The requested src/training.py must not shadow the existing namespace package.
import types,importlib.machinery
if 'training' not in sys.modules:
 original_training=types.ModuleType('training')
 original_training.__path__=[str(PROJECT/'training')]
 original_training.__spec__=importlib.machinery.ModuleSpec('training',loader=None,is_package=True)
 sys.modules['training']=original_training
sys.path.insert(0,str(D))
import dcore as dc
import cv2,numpy as np,torch
from models.crnn import CRNN
from data.collate import talim_collate_fn,resize_preserve_aspect_ratio
from data.ctc_decode import greedy_ctc_decode
from training.manifest_training import load_vocabulary,runtime_ctc_lengths,fp32_ctc_loss
from data.encoder import CTCLabelEncoder
SYN=Path(r'C:\FYP Dataset Generation\NAQSHKASH_Stage1_Kaggle_Training')
CHECKPOINT=dc.old.CHECKPOINT;VOCAB=dc.old.VOCABULARY
EXPECTED='22450cbddd42968149739ea0293a6f4ad105f593e82387d707ecb6418a6bac0f'
digest=dc.old.digest
def read(path):return dc.read(path)
def dump(path,data):
 p=E/path;p.parent.mkdir(parents=True,exist_ok=True);dc.old.write_json(p,data)
def write(path,rows):dc.old.write_csv(E/path,rows)
def seed(value=42):
 random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.set_num_threads(4);cv2.setNumThreads(1);torch.use_deterministic_algorithms(True)
def model_load(path=CHECKPOINT,device='cpu'):
 state=torch.load(path,map_location='cpu',weights_only=True);v,h=load_vocabulary(VOCAB)
 assert state['char_to_idx']==v.char_to_index and state['vocabulary_hash']==h
 if Path(path)==CHECKPOINT:assert digest(path)==EXPECTED
 m=CRNN(66);m.load_state_dict(state['model_state_dict'],strict=True);m.to(device)
 return m,v,state
def tensor(image):return torch.from_numpy(image.astype(np.float32)/255).unsqueeze(0)
def input_hash(image):
 im=resize_preserve_aspect_ratio(tensor(image))
 return hashlib.sha256(str(tuple(im.shape)).encode()+im.numpy().tobytes()).hexdigest()
def row_from_image(path,truth,**fields):
 image=cv2.imread(str(path),0)
 if image is None:raise ValueError('Unreadable image: '+str(path))
 v,_=load_vocabulary(VOCAB);encoded=v.encode(truth);assert truth and v.decode(encoded)==truth
 height,width=image.shape;resized=max(1,round(width*64/height));required=len(encoded)+sum(a==b for a,b in zip(encoded,encoded[1:]))
 assert resized>=4 and required<=resized//4
 return dict(path=str(path),truth=truth,width=width,height=height,resized_width=resized,ctc_required=required,input_hash=input_hash(image),file_sha256=digest(path),**fields)
def protect_snapshot():
 return {str(p):digest(p) for p in PROJECT.rglob('*') if p.is_file() and not p.is_relative_to(E) and not {'.git','__pycache__','.pytest_cache'}.intersection(p.parts)}
def verify():
 lock=json.loads((E/'configs/input_lock.json').read_text())
 bad=[p for p,h in lock.items() if not Path(p).is_file() or digest(p)!=h]
 if bad:raise ValueError('Protected input drift: '+str(bad))
 return len(lock)

