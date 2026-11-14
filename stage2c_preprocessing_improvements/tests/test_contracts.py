"""Stage2C contract/reference tests, no tuning and no model training."""
import hashlib,inspect,json,sys,unittest,zipfile
from pathlib import Path
import cv2,numpy as np
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from core import *
from pipeline import process_image,Processor,engine,photo
ref=load_local('stage2c_test_reference','evaluation/reference.py')

class Stage2CTests(unittest.TestCase):
 def test_pixels_and_config_only(self):
  self.assertEqual(list(inspect.signature(process_image).parameters),['image','configuration'])
  self.assertNotIn('truth',inspect.getsource(engine.propose));self.assertNotIn('predict',inspect.getsource(engine.propose))
 def test_blank_safe_rejection_and_source_immutability(self):
  source=np.full((100,150),255,np.uint8);copy=source.copy()
  for c in CANDIDATES.values():
   result=process_image(source,c)
   if c['robust']:self.assertFalse(result['accepted'])
  self.assertTrue(np.array_equal(source,copy))
 def test_normalization_white_identity_and_grey_background(self):
  image=np.full((120,160),255,np.uint8);image[40:70,40:100]=0
  self.assertTrue(np.array_equal(photo.flatten(image),image))
  grey=(image.astype(float)*.75).astype(np.uint8);flat=photo.flatten(grey)
  self.assertGreater(np.median(flat),245);self.assertLess(np.median(flat[45:65,45:95]),20)
 def test_source_parents_are_isolated_and_unexposed(self):
  rows=read_csv(HERE/'manifests/parents.csv');keys=[(r['dataset'],r['parent_id']) for r in rows]
  self.assertEqual(len(keys),len(set(keys)));self.assertEqual(len(keys),478)
  exposed={(r['dataset'],r['parent_id']) for r in read_csv(B/'manifests/evaluation_pages.csv')}
  self.assertFalse(set(keys)&exposed)
  self.assertEqual(sum(r['dataset']=='positive' and r['partition']=='validation' for r in rows),46)
 def test_source_and_manifest_locks(self):
  verify_lock();self.assertEqual(digest(CHECKPOINT),'22450cbddd42968149739ea0293a6f4ad105f593e82387d707ecb6418a6bac0f')
 def test_missing_ink_not_claimed_correct(self):
  labels=np.zeros((40,80),np.uint16);labels[10:20,10:20]=1;labels[25:35,10:20]=2
  entries=[dict(id=1,row=0,role='upper',truth='A',raw='A',leading_offset=0),dict(id=2,row=0,role='count',truth='a',raw='a',leading_offset=0)]
  result=dict(accepted=True,image=np.full((40,80),255,np.uint8),matrix=np.eye(3),rows=[[dict(row=0,role='upper',box=[0,10,80,15]),dict(row=0,role='count',box=[0,25,80,40])]])
  self.assertFalse(ref.assess(result,labels,entries)['correct'])
 def test_neighbour_contamination_is_rejected(self):
  labels=np.zeros((50,80),np.uint16);labels[5:10,10:20]=1;labels[25:35,10:20]=2
  entries=[dict(id=1,row=0,role='upper',truth='A'),dict(id=2,row=0,role='count',truth='a')]
  result=dict(accepted=True,image=np.full((50,80),255,np.uint8),matrix=np.eye(3),rows=[[dict(row=0,role='upper',box=[0,0,80,35]),dict(row=0,role='count',box=[0,20,80,40])]])
  self.assertFalse(ref.assess(result,labels,entries)['correct'])
 def test_annotation_only_rotation_and_perspective_dimensions(self):
  labels=np.zeros((100,160),np.uint16);labels[20:50,30:80]=1
  recipe=dict(padding=12,operations=[dict(type='rotation',degrees=2.)])
  from PIL import Image,ImageOps
  shape=np.asarray(ImageOps.expand(Image.fromarray(labels),12).rotate(2.,expand=True)).shape
  out=ref.augmented_reference(labels,recipe,shape);self.assertEqual(out.shape,shape);self.assertIn(1,np.unique(out))
  for axis in ('horizontal','vertical'):
   recipe=dict(padding=12,operations=[dict(type='perspective',axis=axis,inset=.04,reverse=False)])
   out=ref.augmented_reference(labels,recipe,(124,184));self.assertEqual(out.shape,(124,184))
 def test_reference_slots_cover_every_source_ink_pixel(self):
  page=next(r for r in read_csv(HERE/'manifests/pages.csv') if r['kind']=='clean' and r['dataset']=='positive')
  c=SOURCE_CONFIG['positive']
  with zipfile.ZipFile(c['clean']) as z:
   image=cv2.imdecode(np.frombuffer(z.read(page['image_member']),np.uint8),0);data=json.loads(z.read(c['prefix']+page['source_json']))
  labels,entries=ref.clean_reference(image,data['logical_rows'])
  self.assertFalse(np.any((image<128)&(labels==0)));self.assertEqual(len(entries),int(labels.max()))
 def test_c0_is_exact_stage2b_blind_proposal(self):
  page=next(r for r in read_csv(HERE/'manifests/pages.csv') if r['kind']=='clean' and r['dataset']=='positive')
  with zipfile.ZipFile(SOURCE_CONFIG['positive']['clean']) as z:image=cv2.imdecode(np.frombuffer(z.read(page['image_member']),np.uint8),0)
  base=baseline_preprocess(image)
  try:
   expected=[[ (role,crop) for role,crop in baseline_tiers(r.image)] for r in base.rows]
  except ValueError:
   self.assertFalse(process_image(image,CANDIDATES['C0'])['accepted']);return
  actual=process_image(image,CANDIDATES['C0']);self.assertTrue(actual['accepted'])
  self.assertEqual(len(actual['rows']),len(expected))
  for row,gt in zip(actual['rows'],expected):
   for tier,(name,crop) in zip(row,gt):
    x0,y0,x1,y1=tier['box'];self.assertEqual(tier['role'],name);self.assertTrue(np.array_equal(actual['image'][y0:y1,x0:x1],crop))
 def test_metrics_keep_spaces_and_edits(self):
  d=edit_counts('A  B','A B');self.assertEqual(d['deletions'],1)
  self.assertEqual(aggregate([dict(truth='A  B',prediction='A B',**d)])['cer'],.25)
 def test_checkpoint_vocabulary_and_decoder_are_original(self):
  import torch
  from models.crnn import CRNN
  from data.ctc_decode import greedy_ctc_decode
  torch.set_num_threads(4);state=torch.load(CHECKPOINT,map_location='cpu',weights_only=True);v,h=load_vocabulary(VOCABULARY)
  model=CRNN(66);model.load_state_dict(state['model_state_dict'],strict=True)
  self.assertEqual(state['char_to_idx'],v.char_to_index);self.assertEqual(h,state['vocabulary_hash']);self.assertEqual(greedy_ctc_decode([0,8,8,0,8],v),'AA')

if __name__=='__main__':unittest.main(verbosity=2)
