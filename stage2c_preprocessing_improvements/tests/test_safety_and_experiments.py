"""Additional Stage2C safety and completed-experiment audit tests."""
import inspect,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from core import *
from pipeline import process_image,engine,geo,photo
ref=load_local('stage2c_safety_reference','evaluation/reference.py')

class SafetyTests(unittest.TestCase):
 def test_no_annotation_keyword_is_accepted(self):
  with self.assertRaises(TypeError):process_image(np.full((64,64),255,np.uint8),CANDIDATES['C5'],expected_rows=1)
 def test_hash_drift_fails_closed_without_changing_a_file(self):
  with patch('core.digest',return_value='tampered'):
   with self.assertRaisesRegex(ValueError,'Frozen source'):verify_lock()
 def test_software_is_locked(self):verify_lock(code=True)
 def test_all_candidates_reject_blank(self):
  for name,config in CANDIDATES.items():self.assertFalse(process_image(np.full((80,120),255,np.uint8),config)['accepted'],name)
 def test_geometry_has_no_unsupported_blank_warp(self):
  image=np.full((100,180),255,np.uint8);corrected,M,details=geo.correct(image)
  self.assertTrue(np.array_equal(image,corrected));self.assertTrue(np.allclose(M,np.eye(3)));self.assertFalse(details['perspective_applied'])
 def test_high_contrast_single_dot_survives_noise_filter(self):
  image=np.full((80,100),255,np.uint8);image[40,50]=0
  output,_=photo.conservative_denoise(image);self.assertEqual(output[40,50],0)
 def test_adaptive_boundaries_do_not_overlap(self):
  image=np.full((110,160),255,np.uint8)
  cv2.circle(image,(40,15),4,0,-1);cv2.circle(image,(34,28),4,0,-1);cv2.circle(image,(46,28),4,0,-1)
  cv2.ellipse(image,(50,64),(22,14),-15,0,360,0,3)
  rows,details=engine.propose(image)
  boxes=[t['box'] for row in rows for t in row]
  self.assertTrue(all(a[3]<=b[1] for a,b in zip(boxes,boxes[1:])))
  source=image.copy();process_image(image,CANDIDATES['C2']);self.assertTrue(np.array_equal(image,source))
 def test_partial_canvas_clipping_is_not_a_false_pass(self):
  labels=np.zeros((40,80),np.uint16);labels[10:20,10:20]=1
  result=dict(accepted=True,image=np.full((40,80),255,np.uint8),matrix=np.array([[1.,0.,65.],[0.,1.,0.],[0.,0.,1.]]),rows=[[dict(row=0,role='count',box=[0,0,80,40])]])
  measured=ref.assess(result,labels,[dict(id=1,row=0,role='count',truth='a')])
  self.assertFalse(measured['correct']);self.assertLess(measured['tiers'][0]['retention'],.98)
 def test_reference_rejects_shape_mismatch(self):
  with self.assertRaisesRegex(ValueError,'dimensions'):ref.augmented_reference(np.zeros((50,80),np.uint16),dict(padding=12,operations=[]),(10,10))
 def test_rotated_reference_never_relabels_ink(self):
  from PIL import Image,ImageOps
  labels=np.zeros((80,100),np.uint16);labels[20:30,20:40]=1;labels[45:60,30:60]=2
  for degrees in (-4.,4.):
   shape=np.asarray(ImageOps.expand(Image.fromarray(labels),12).rotate(degrees,expand=True)).shape
   out=ref.augmented_reference(labels,dict(padding=12,operations=[dict(type='rotation',degrees=degrees)]),shape)
   self.assertEqual(set(np.unique(out)),{0,1,2})
 def test_c5_is_frozen_conservative_variant(self):
  configs=json.loads((HERE/'protocol/candidates.json').read_text())
  self.assertTrue(configs['C5']['conservative']);self.assertFalse(configs['C4']['conservative'])
  self.assertEqual(set(configs),{'C0','C1','C2','C3','C4','C5'})
 def test_each_selected_parent_keeps_all_existing_variants(self):
  parents=read_csv(HERE/'manifests/parents.csv');pages=read_csv(HERE/'manifests/pages.csv')
  for parent in parents:
   subset=[r for r in pages if r['dataset']==parent['dataset'] and r['parent_id']==parent['parent_id']]
   self.assertEqual({r['partition'] for r in subset},{parent['partition']})
   self.assertEqual(sum(r['kind']=='clean' for r in subset),1)
   self.assertEqual(sum(r['kind']=='scanned' for r in subset),0 if parent['dataset']=='normal' else 3)

class CompletedExperimentTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.root=HERE/'reports/experiments_v1'
  cls.complete=json.loads((cls.root/'completion.json').read_text())
  cls.pages=read_csv(cls.root/'pages.csv');cls.predictions=read_csv(cls.root/'predictions.csv')
 def test_all_candidates_used_identical_pages(self):
  planned={(r['dataset'],r['case_id']) for r in read_csv(HERE/'manifests/pages.csv')}
  self.assertEqual(len(planned),1612)
  for candidate in CANDIDATES:
   actual=[(r['dataset'],r['case_id']) for r in self.pages if r['candidate']==candidate]
   self.assertEqual(set(actual),planned);self.assertEqual(len(actual),len(planned))
 def test_selection_happened_before_validation(self):
  selection=json.loads((self.root/'selection_before_validation.json').read_text())
  self.assertEqual(selection['selection_inputs_partition'],'tuning');self.assertFalse(selection['validation_started'])
  self.assertEqual(selection['completed_page_evaluations'],1128*6)
  self.assertEqual(selection['protocol_sha256'],digest(HERE/'protocol/lock.json'))
 def test_rejected_or_wrong_crops_do_not_enter_ocr(self):
  success={(r['candidate'],r['dataset'],r['case_id']) for r in self.pages if r['correct_segmentation']=='True' and r['ctc_failures']=='0'}
  self.assertTrue(all((r['candidate'],r['dataset'],r['case_id']) in success for r in self.predictions))
 def test_predictions_keep_raw_label_offsets(self):
  self.assertTrue(all(int(r['leading_offset'])>=0 and float(r['retention'])>=.98 and float(r['contamination'])<=.02 for r in self.predictions))
  self.assertTrue(any(int(r['leading_offset'])>0 for r in self.predictions))
 def test_all_page_outcomes_are_accounted(self):
  for row in self.pages:
   self.assertTrue(row['proposal'] in ('True','False'));self.assertTrue(row['correct_segmentation'] in ('True','False'))
   if row['correct_segmentation']=='True':self.assertEqual(row['proposal'],'True')
   else:self.assertEqual(row['ocr_tiers'],'0')
 def test_no_model_state_change_and_uncached_replay(self):
  self.assertTrue(self.complete['model_state_unchanged']);self.assertEqual(len(self.complete['determinism_replays']),6)
  self.assertTrue(all(r['uncached_prediction_equal'] for r in self.complete['determinism_replays']))
 def test_software_and_protected_project_are_unchanged(self):
  verify_lock(code=True);before=json.loads((HERE/'protocol/protected_before.json').read_text());self.assertEqual(protected(),before)

if __name__=='__main__':unittest.main(verbosity=2)
