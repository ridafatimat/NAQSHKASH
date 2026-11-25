"""Checks against actually executed optimization and paired final artifacts."""
import sys,unittest,json,math,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from context import *
from checkpoint_utils import set_finetuning_mode,forward_training
from datasets import FrozenTiers
from selection import eligible,rank
class ExecutedTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):seed()
 def test_preflight_real_updates_and_safe_reload(self):
  p=json.loads((E/'reports/preflight.json').read_text());self.assertTrue(p['passed']);self.assertFalse(p['selected'])
  self.assertTrue(all(r['weight_updated'] and math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in p['records']))
  self.assertTrue(p['reloaded_logits_equal']);self.assertGreater(max(int(r['resized_width']) for r in p['records']),30000)
 def test_actual_experiments_loaded_original_not_preflight(self):
  for name in ['e1','e2']:
   initial=json.loads((E/'reports'/f'{name}_initialization.json').read_text())
   self.assertTrue(initial['all_parameter_tensors_equal']);self.assertEqual(initial['original_sha256'],EXPECTED);self.assertEqual(initial['original_checkpoint'],str(CHECKPOINT))
 def test_training_usage_and_mixture_are_actual(self):
  for name in ['e1','e2']:
   c=json.loads((E/'reports'/f'{name}_completion.json').read_text());rows=read(E/'metrics'/f'{name}_training_usage.csv')
   self.assertEqual(len(rows),c['sample_exposures']);self.assertEqual(c['sample_exposures'],4*c['steps'])
   self.assertGreater(c['steps'],0);self.assertFalse(c['failures']);self.assertTrue(c['original_unchanged'])
   if name=='e2':self.assertEqual(c['positive_exposures'],c['replay_exposures'])
   else:self.assertEqual(c['replay_exposures'],0)
 def test_optimizer_losses_gradients_finite(self):
  for name in ['e1','e2']:
   rows=json.loads((E/'metrics'/f'{name}_training_history.json').read_text())
   self.assertTrue(all(math.isfinite(r['train_loss']) and math.isfinite(r['gradient_norm']) for r in rows))
 def test_paired_full_evaluation_inputs_identical(self):
  rows=[read(E/'metrics'/f'{name}_per_sample.csv') for name in ['baseline','e1','e2']]
  self.assertEqual(len(rows[0]),2493)
  self.assertTrue(all([(r['id'],r['input_hash'],r['truth']) for r in x]==[(r['id'],r['input_hash'],r['truth']) for r in rows[0]] for x in rows[1:]))
 def test_fresh_baseline_reproduces_stage2d(self):
  r=json.loads((E/'reports/baseline_reproduction_check.json').read_text());self.assertTrue(r['exact_stage2d_predictions_reproduced']);self.assertEqual(r['checked_tiers'],2430)
 def test_selected_checkpoint_and_lineage_compatible(self):
  r=json.loads((E/'reports/selected_checkpoint.json').read_text());m,v,c=model_load(Path(r['selected_checkpoint']))
  self.assertEqual(digest(r['selected_checkpoint']),r['sha256']);self.assertEqual(len(v.char_to_index),65);self.assertTrue(Path(r['selected_checkpoint']).is_relative_to(E))
  self.assertEqual(r['original_sha256'],EXPECTED);self.assertFalse(r['independent_test']);self.assertEqual(r['untouched_test_evaluations'],0)
 def test_safe_weights_only_optimizer_rng_reload(self):
  p=E/'checkpoints/preflight/not_selected.pt';c=torch.load(p,map_location='cpu',weights_only=True);m,v,_=model_load(p)
  optimizer=torch.optim.AdamW(m.parameters(),lr=1e-5);optimizer.load_state_dict(c['optimizer_state_dict'])
  self.assertTrue(optimizer.state);self.assertEqual(c['step'],3)
  torch.set_rng_state(c['torch_rng']);a=torch.rand(4);torch.set_rng_state(c['torch_rng']);b=torch.rand(4);self.assertTrue(torch.equal(a,b))
  n=c['numpy_rng'];np.random.set_state((n['name'],np.asarray(n['keys'],np.uint32),n['position'],n['has_gauss'],n['cached_gauss']))
 def test_checkpoint_resume_next_update_deterministic(self):
  # Same saved optimizer/weights/RNG produce the same actual next micro-update.
  checkpoint=torch.load(E/'checkpoints/preflight/not_selected.pt',map_location='cpu',weights_only=True)
  sample=FrozenTiers([read(E/'manifests/positive_train.csv')[0]])[0];batch=talim_collate_fn([sample]);outputs=[]
  for repetition in range(2):
   m,v,_=model_load(E/'checkpoints/preflight/not_selected.pt');o=torch.optim.AdamW(m.parameters(),lr=1e-5);o.load_state_dict(copy.deepcopy(checkpoint['optimizer_state_dict']));torch.set_rng_state(checkpoint['torch_rng']);set_finetuning_mode(m)
   enc=CTCLabelEncoder(v).encode_batch(batch['label_texts']);o.zero_grad();logits=forward_training(m,batch['images']);lengths,req=runtime_ctc_lengths(m,batch,enc,logits.shape[0])
   loss,_=fp32_ctc_loss(logits,enc['targets'],lengths,enc['target_lengths'],torch.nn.CTCLoss(blank=0,zero_infinity=False));loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);o.step();outputs.append({k:t.detach().clone() for k,t in m.state_dict().items()})
  self.assertTrue(all(torch.equal(outputs[0][k],outputs[1][k]) for k in outputs[0]))
 def test_actual_selected_model_satisfies_predefined_gates(self):
  selection=json.loads((E/'reports/selected_checkpoint.json').read_text())
  summaries={n:json.loads((E/'metrics'/f'{n}_summary.json').read_text()) for n in ['baseline','e1','e2']}
  passing=[n for n in ['e1','e2'] if eligible(summaries[n],summaries['baseline'])]
  if not passing:self.assertEqual(selection['selected'],'E0')
  else:
   name={'E1':'e1','E2':'e2'}[selection['selected']]
   self.assertIn(name,passing);self.assertEqual(rank(summaries[name]),min(rank(summaries[n]) for n in passing))
 def test_actual_weights_changed_and_batchnorm_buffers_preserved(self):
  original=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)['model_state_dict']
  for name in ['e1','e2']:
   path=json.loads((E/'reports'/f'{name}_completion.json').read_text())['candidate_checkpoint']
   state=torch.load(path,map_location='cpu',weights_only=True)['model_state_dict']
   self.assertTrue(any(not torch.equal(t,original[k]) for k,t in state.items()))
   for k,t in state.items():
    if k.endswith(('running_mean','running_var','num_batches_tracked')):self.assertTrue(torch.equal(t,original[k]))
 def test_no_silent_ctc_infeasible_scoring(self):
  for name in ['baseline','e1','e2']:
   for r in read(E/'metrics'/f'{name}_per_sample.csv'):
    self.assertLessEqual(int(r['ctc_required']),int(r['resized_width'])//4)
    self.assertEqual(len(r['truth']),int(r['characters']))
 def test_historical_files_unchanged(self):
  r=json.loads((E/'reports/protected_file_hash_check.json').read_text());self.assertTrue(r['passed']);self.assertFalse(r['changed'])
  self.assertEqual(digest(CHECKPOINT),EXPECTED)
 def test_frozen_source_and_input_locks(self):
  verify()
  for f,h in json.loads((E/'configs/software_lock.json').read_text()).items():self.assertEqual(digest(f),h)
if __name__=='__main__':unittest.main()

