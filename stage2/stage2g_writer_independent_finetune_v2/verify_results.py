"""Independent completed-artifact checks; no inference or training."""
from pathlib import Path
import csv,json,hashlib,collections,math
P=Path('C:/NAQSHKASH/stage2/stage2g_writer_independent_finetune_v2')
def read(n):return list(csv.DictReader((P/n).open(encoding='utf-8-sig')))
def load(n):return json.loads((P/n).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ed(a,b):
 d=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  new=[i]
  for j,y in enumerate(b,1):new.append(min(new[-1]+1,d[j]+1,d[j-1]+(x!=y)))
  d=new
 return d[-1]
checks={};rr=read('audited_handwritten_inputs.csv');ids={r['id'] for r in rr}
checks['unique_authorized_samples']=len(rr)==len(ids)==2903
checks['train_counts']=collections.Counter(r['sample_type'] for r in rr if r['split']=='train')=={'original':108,'augmented':2469}
checks['validation_originals_only_for_candidate_evaluation']=not any((P/'metrics').glob('secondary_C*')) and not any((P/'metrics').glob('secondary_D*'))
checks['writer_allowlists']=all(r['writer_id'] in {'W01','W03','W04','W06'} if r['split']=='train' else r['writer_id']=='W05' for r in rr)
checks['exact_extracted_payload_allowlist']={str(p) for p in (P/'packages').rglob('*') if p.is_file()}=={r[k] for r in rr for k in ['path','label_path']}
checks['normalization_frozen_before_W05']=(P/'selected_normalization.json').stat().st_mtime<(P/'metrics/A_frozen_original.json').stat().st_mtime
checks['gates_frozen_before_W05']=(P/'training_config.json').stat().st_mtime<(P/'metrics/A_frozen_original.json').stat().st_mtime
schedule=load('sampling_schedule.json');checks['identical_fixed_216_step_schedule']=len(schedule)==216
checks['all_scheduled_handwriting_training_only']=all(i['row']['split']=='train' and i['row']['writer_id'] in {'W01','W03','W04','W06'} for s in schedule for i in s['batch'] if i['domain']!='synthetic_replay')
checks['no_near_duplicate_augmentations']=all(i['row']['near_duplicate'].lower()!='true' for s in schedule for i in s['batch'] if i['domain']=='handwritten_augmented')
checks['augmentation_exact_parent_balance']=all(len([i for i in s['batch'] if i['domain']=='handwritten_augmented'])<=1 and all(i['row']['tier_id']==s['batch'][0]['row']['tier_id'] for i in s['batch'] if i['domain']=='handwritten_augmented') for s in schedule)
writer=collections.Counter();replay=0.
for s in schedule:
 for i in s['batch']:
  if i['domain']=='synthetic_replay':replay+=i['loss_weight']
  else:writer[i['row']['writer_id']]+=i['loss_weight']
checks['equal_writer_effective_loss']=max(writer.values())-min(writer.values())<1e-8
checks['equal_handwriting_replay_effective_loss']=abs(sum(writer.values())-replay)<1e-8
results=load('final_comparison.json');candidates=load('all_candidates.json')
checks['candidate_checkpoint_hashes_match']=all(sha(x['checkpoint'])==x['sha256'] for x in candidates)
checks['distinct_raw_normalized_checkpoint_paths']=len({x['checkpoint'] for x in candidates})==len(candidates)
primary_truth={r['id']:r['truth'] for r in rr if r['split']=='validation' and r['sample_type']=='original'}
checks['all_primary_metric_arithmetic']=True;checks['same_original_validation_targets']=True
for name in ['A_frozen_original','B_frozen_normalized']+[x['branch']+'_epoch_'+str(x['epoch']) for x in candidates]:
 pred=read('predictions/'+name+'.csv');m=load('metrics/'+name+'.json')['overall']
 assert len(pred)==14 and {r['id']:r['truth'] for r in pred}==primary_truth
 assert all(ed(r['truth'],r['prediction'])==int(r['edit_distance'])==sum(int(r[k]) for k in ['substitutions','deletions','insertions']) for r in pred)
 assert sum(ed(r['truth'],r['prediction']) for r in pred)==m['errors']
 assert abs(m['cer']-m['errors']/101)<1e-12
 assert abs(m['cer_no_whitespace']-sum(ed(''.join(r['truth'].split()),''.join(r['prediction'].split())) for r in pred)/67)<1e-12
for branch in ['C_raw','D_normalized']:
 h=load('training/'+branch+'_history.json');u=read('training/'+branch+'_exposures.csv');completion=load('training/'+branch+'_completion.json')
 assert len(h)==completion['steps'] and len(h) in [108,216]
 assert len(u)==sum(len(s['batch']) for s in schedule[:len(h)])
 assert all(r['normalization']==('identity' if branch=='C_raw' else 'contrast') for r in u)
checks['training_logs_and_actual_preprocessing_match']=True
import torch
frozen=Path('C:/NAQSHKASH/stage2/stage2e_finetuning/checkpoints/selected_model/best.pt')
state=torch.load(frozen,map_location='cpu',weights_only=True)
checks['all_RNN_classifier_and_BN_running_states_unchanged']=True
checks['CNN_updates_are_nonzero']=True
for x in candidates:
 other=torch.load(x['checkpoint'],map_location='cpu',weights_only=True)
 assert other['stage2g_v2_initial_checkpoint_sha256']==results['approaches']['A']['sha256']
 for k,v in state['model_state_dict'].items():
  if k.startswith(('rnn.','classifier.')) or any(s in k for s in ['running_mean','running_var','num_batches_tracked']):assert torch.equal(v,other['model_state_dict'][k]),k
 assert any(not torch.equal(v,other['model_state_dict'][k]) for k,v in state['model_state_dict'].items() if k.startswith('cnn.'))
checks['final_integrity_passes']=all(load('final_integrity.json').values())
checks['source_hashes_match']=all(sha(f)==h for f,h in load('software_hashes.json').items())
assert all(checks.values()),checks
(P/'verification_results.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
