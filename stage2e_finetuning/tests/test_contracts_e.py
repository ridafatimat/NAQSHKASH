import sys,unittest,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from context import *
from datasets import FrozenTiers,schedule
from metrics import measurements,aggregate
from checkpoint_utils import forward_training,set_finetuning_mode
from selection import eligible
class ContractTests(unittest.TestCase):
 def test_original_checkpoint_vocabulary_and_architecture(self):
  model,v,state=model_load();self.assertEqual(digest(CHECKPOINT),EXPECTED);self.assertEqual(len(v.char_to_index),65);self.assertEqual(v.blank_index,0);self.assertEqual(model.rnn.hidden_size,256);self.assertEqual(model.rnn.num_layers,2)
 def test_ctc_encoder_preserves_meaningful_spaces(self):
  v,_=load_vocabulary(VOCAB);e=CTCLabelEncoder(v).encode_batch(['aa  a'])
  self.assertEqual(v.decode(e['targets'].tolist()),'aa  a');self.assertEqual(e['target_lengths'].item(),5)
  required=5+sum(a==b for a,b in zip('aa  a','aa  a'[1:]));self.assertEqual(required,7)
 def test_original_normalization_and_width(self):
  r=read(E/'manifests/positive_train.csv')[0];d=FrozenTiers([r]);batch=talim_collate_fn([d[0]])
  self.assertEqual(batch['images'].shape[2],64);self.assertEqual(int(batch['image_widths'][0]),int(r['resized_width']))
  self.assertGreaterEqual(float(batch['images'].min()),0);self.assertLessEqual(float(batch['images'].max()),1)
 def test_parent_disjoint_and_no_test_training(self):
  train=read(E/'manifests/positive_train.csv');dev=read(E/'manifests/positive_development.csv');replay=read(E/'manifests/synthetic_replay_train.csv')
  self.assertTrue({r['parent_id'] for r in train}.isdisjoint({r['parent_id'] for r in dev}))
  self.assertTrue(all(r['split']=='train' for r in train+replay))
  exposed={(r['dataset'],r['parent_id']) for r in dc.read(dc.old.B/'manifests/evaluation_pages.csv')}
  self.assertFalse({('positive',r['parent_id']) for r in train}&exposed)
 def test_no_normalized_pixel_leakage(self):
  tr=read(E/'manifests/positive_train.csv')+read(E/'manifests/synthetic_replay_train.csv')
  ev=read(E/'manifests/evaluation.csv')
  self.assertFalse({r['input_hash'] for r in tr}&{r['input_hash'] for r in ev})
 def test_all_targets_ctc_feasible_and_hash_valid(self):
  v,_=load_vocabulary(VOCAB)
  rows=read(E/'manifests/positive_train.csv')+read(E/'manifests/synthetic_replay_train.csv')+read(E/'manifests/evaluation.csv')
  for r in rows:
   ids=v.encode(r['truth']);self.assertTrue(ids);self.assertEqual(v.decode(ids),r['truth'])
   required=len(ids)+sum(a==b for a,b in zip(ids,ids[1:]))
   self.assertEqual(required,int(r['ctc_required']));self.assertLessEqual(required,int(r['resized_width'])//4)
 def test_mixed_replay_balanced_and_deterministic(self):
  p=read(E/'manifests/positive_train.csv');r=read(E/'manifests/synthetic_replay_train.csv')
  one=schedule(p,r,20,True,42);two=schedule(p,r,20,True,42);self.assertEqual(one,two)
  self.assertTrue(all(sum(s=='positive' for s,i in b)==2 and sum(s=='replay' for s,i in b)==2 for b in one))
 def test_pixel_only_inputs_cannot_receive_annotation_counts(self):
  r=read(E/'manifests/positive_train.csv')[0];item=FrozenTiers([r])[0]
  self.assertEqual(set(item),{'image','label_text','sample_id','tier_name'})
 def test_checkpointed_forward_preserves_architecture(self):
  seed();m,_,_=model_load();m.eval();x=torch.ones(1,1,64,64)
  direct=m(x);checkpointed=forward_training(m,x,checkpoint_width=0)
  self.assertTrue(torch.equal(direct,checkpointed))
 def test_frozen_batchnorm_running_statistics(self):
  m,_,_=model_load();set_finetuning_mode(m)
  self.assertTrue(m.training);self.assertTrue(all(not b.training for b in m.modules() if isinstance(b,torch.nn.BatchNorm2d)))
  self.assertTrue(all(b.weight.requires_grad for b in m.modules() if isinstance(b,torch.nn.BatchNorm2d)))
 def test_metrics_keep_spaces_and_diagnostic_separate(self):
  row=dict(parent_id='p',**measurements('A  B','A B'))
  m=aggregate([row]);self.assertEqual(m['cer'],.25);self.assertEqual(m['exact_accuracy'],0);self.assertEqual(m['space_normalized_diagnostic_cer'],0)
 def test_model_selection_rejects_clean_regression(self):
  b=dict(positive_development=dict(cer=.1,macro_parent_cer=.1),clean_validation=dict(cer=.001,exact_accuracy=.99),long_positive=dict(cer=.2))
  c=json.loads(json.dumps(b));c['positive_development'].update(cer=.09,macro_parent_cer=.09)
  self.assertTrue(eligible(c,b));c['clean_validation']['cer']=.01;self.assertFalse(eligible(c,b))
 def test_model_selection_rejects_parent_instability(self):
  b=dict(positive_development=dict(cer=.1,macro_parent_cer=.1),clean_validation=dict(cer=.001,exact_accuracy=.99),long_positive=dict(cer=.2))
  c=json.loads(json.dumps(b));c['positive_development'].update(cer=.05,macro_parent_cer=.11)
  self.assertFalse(eligible(c,b))
 def test_replay_has_full_supported_vocabulary(self):
  v,_=load_vocabulary(VOCAB);rows=read(E/'manifests/synthetic_replay_train.csv')
  self.assertEqual(set(''.join(r['truth'] for r in rows)),set(v.char_to_index))
if __name__=='__main__':unittest.main()

