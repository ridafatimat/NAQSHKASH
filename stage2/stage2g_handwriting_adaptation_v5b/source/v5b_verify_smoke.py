import json,collections,gc,time
import v5b_engine as e
base=e.ROOT/'local_run/smoke_tests';foldset=sorted(base.glob('folds_*'))[-1];finalset=sorted(base.glob('final_*'))[-1]
initial=e.torch.load(e.c.E/'checkpoints/selected_model/best.pt',map_location='cpu',weights_only=True)['model_state_dict'];folds=[]
for path in sorted(foldset.glob('*/completion.json')):
 done=json.loads(path.read_text());cfg=json.loads((path.parent/'run_config.json').read_text());assert done['steps']==20 and cfg['smoke']
 logs=[json.loads(line) for file in path.parent.glob('attempt_*/training.jsonl') for line in file.read_text().splitlines()];updates=[r for r in logs if r['event']=='update'];assert len(updates)==20
 for r in updates:
  assert collections.Counter(v['domain'] for v in r['exposures'])==cfg['effective_batch']
  assert all(v['writer_id']!=cfg['heldout_writer'] and v['writer_id']!='W05' for v in r['exposures'] if v['domain'].startswith('handwritten'))
 saved=e.torch.load(done['best_checkpoint'],map_location='cpu',weights_only=True)
 changes={group:any(not e.torch.equal(t,initial[k]) for k,t in saved['model_state_dict'].items() if k.startswith(group+'.') and not any(word in k for word in ['running_mean','running_var','num_batches_tracked'])) for group in ['cnn','rnn','classifier']}
 assert changes['cnn'] and changes['rnn'];assert changes['classifier']==('classifier' in cfg['groups'])
 assert all(int(s['step'])==20 for s in saved['optimizer_state_dict']['state'].values());assert cfg['initial_sha256']==e.config()['frozen_e1_sha256']
 folds.append(dict(configuration=path.parent.name,steps=20,heldout_writer=cfg['heldout_writer'],seconds_per_update=done['seconds_per_update'],groups=cfg['groups'],parameter_groups_changed=changes,heldout_training_leakage=False,W05_accessed=False,checkpoint_sha256=e.sha(done['best_checkpoint']),optimizer_step_20_verified=True));del saved;gc.collect()
assert len(folds)==4
done=json.loads((finalset/'training/completion.json').read_text());assert done['steps']==20 and done['smoke'];evaluation=json.loads((finalset/'evaluation_once.json').read_text());assert not evaluation['W05_used'] and evaluation['smoke'];assert evaluation['full_14']['raw']['overall']['tiers']==14 and evaluation['subset_13']['raw']['overall']['tiers']==13
assert {r['writer_id'] for r in e.read(finalset/'evaluation/candidate_once_raw.csv')}=={'W01'}
assert json.loads((finalset/'decision.json').read_text())['operational_default']=='frozen_E1'
resume=json.loads((e.ROOT/'local_run/resume_verification.json').read_text());assert resume['passed']
proof=dict(passed=True,fold_runs=folds,final_run=dict(steps=20,seconds_per_update=done['seconds_per_update'],W05_accessed=False,full_and_subset_paths_verified=True,synthetic_gate_and_frozen_fallback_paths_verified=True,checkpoint_sha256=e.sha(done['best_checkpoint'])),resume_verification=resume,outputs_deleted=False,only_bounded_smoke_runs_launched=True,non_smoke_outputs_present=(e.ROOT/'local_run/outputs').exists(),verified_unix=time.time())
assert not proof['non_smoke_outputs_present'];e.dump(e.ROOT/'local_run/smoke_test_verification.json',proof);print('BOTH SCRIPTS VERIFIED; 4 fold combinations x20, final x20, resume and gates passed; ready to delete temporary outputs')
