"""Bounded conservative fine-tuning from the original checkpoint, with replay."""
from context import *
from datasets import FrozenTiers,schedule
from checkpoint_utils import save_checkpoint,set_finetuning_mode,forward_training
from evaluation import evaluate
from selection import eligible,rank
import argparse,math
def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',choices=['e1','e2'],required=True);p.add_argument('--resume',type=Path);p.add_argument('--device',default='auto');a=p.parse_args()
 config=json.loads((E/'configs'/('e1_positive_only.yaml' if a.experiment=='e1' else 'e2_mixed_replay.yaml')).read_text())
 seed(config['seed']);verify()
 software=json.loads((E/'configs/software_lock.json').read_text())
 for f,h in software.items():assert digest(f)==h,f
 device='cuda' if a.device=='auto' and torch.cuda.is_available() else 'cpu' if a.device=='auto' else a.device
 if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested CUDA is unavailable')
 m,v,initial=model_load(device=device)
 assert digest(CHECKPOINT)==EXPECTED
 optimizer=torch.optim.AdamW(m.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
 encoder=CTCLabelEncoder(v);criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False)
 positive=FrozenTiers(E/'manifests/positive_train.csv');replay=FrozenTiers(E/'manifests/synthetic_replay_train.csv')
 batches=schedule(positive.rows,replay.rows,config['max_steps'],config['mixed_replay'],config['seed'])
 dev=read(E/'manifests/selection_development.csv')
 baseline=json.loads((E/'metrics/baseline_selection_summary.json').read_text())
 folder=E/'checkpoints'/a.experiment;start_step=0;history=[];used=[];best=None;bestpath=None;stale=0;start=time.perf_counter();optimizer_seconds=0.
 if a.resume:
  resumed=torch.load(a.resume,map_location='cpu',weights_only=True)
  assert resumed['original_checkpoint_sha256']==EXPECTED and resumed['config']==config
  m.load_state_dict(resumed['model_state_dict']);optimizer.load_state_dict(resumed['optimizer_state_dict']);start_step=resumed['step']
  history=resumed['extra']['history'];used=resumed['extra']['used'];best=resumed['extra']['best'];bestpath=resumed['extra']['bestpath'];stale=resumed['extra']['stale']
  random.setstate(resumed['python_rng']);n=resumed['numpy_rng'];np.random.set_state((n['name'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gauss']));torch.set_rng_state(resumed['torch_rng'])
  if device=='cuda' and resumed.get('cuda_rng'):torch.cuda.set_rng_state_all(resumed['cuda_rng'])
 else:
  if folder.exists():raise RuntimeError('Existing run directory; use explicit resume or a new experiment version')
  folder.mkdir()
  dump('reports/'+a.experiment+'_initialization.json',dict(original_checkpoint=str(CHECKPOINT),original_sha256=digest(CHECKPOINT),all_parameter_tensors_equal=all(torch.equal(t.cpu(),initial['model_state_dict'][n]) for n,t in m.state_dict().items()),optimizer='fresh AdamW',device=device,config=config))
 numerical_failures=[];status='completed_budget';final_step=start_step
 for index in range(start_step,config['max_steps']):
  set_finetuning_mode(m);optimizer.zero_grad(set_to_none=True);t0=time.perf_counter();losses=[];gradient=None
  lr=config['learning_rate']*(.1+.9*(1+math.cos(math.pi*index/config['max_steps']))/2)
  for group in optimizer.param_groups:group['lr']=lr
  # Microbatch1 avoids padding waste; four examples form each effective update.
  try:
   for source,i in batches[index]:
    data=positive if source=='positive' else replay;row=data.rows[i];batch=talim_collate_fn([data[i]]);encoded=encoder.encode_batch(batch['label_texts'])
    logits=forward_training(m,batch['images'].to(device),config['checkpoint_width'])
    lengths,required=runtime_ctc_lengths(m,batch,encoded,logits.shape[0])
    loss,_=fp32_ctc_loss(logits,encoded['targets'].to(device),lengths,encoded['target_lengths'],criterion)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite CTC loss')
    (loss/4).backward();losses.append(float(loss.detach()))
   gradient=torch.nn.utils.clip_grad_norm_(m.parameters(),config['gradient_clip'])
   if not torch.isfinite(gradient):raise FloatingPointError('Nonfinite gradient norm')
   optimizer.step()
  except (RuntimeError,FloatingPointError) as exc:
   optimizer.zero_grad(set_to_none=True);status='stopped_failure'
   numerical_failures.append(dict(step=index+1,error=str(exc),sample_ids=[(s,(positive if s=='positive' else replay).rows[i]['id']) for s,i in batches[index]]))
   dump('reports/'+a.experiment+'_failure.json',dict(failures=numerical_failures,recovery='No sample skipped, label truncated or original checkpoint changed. Resume last complete checkpoint. Microbatch already1; CNN checkpointing active for wide inputs. If allocator/OOM persists, use portable CUDA package with identical manifests.'))
   if device=='cuda':torch.cuda.empty_cache()
   break
  elapsed=time.perf_counter()-t0;optimizer_seconds+=elapsed;final_step=index+1
  used.extend(dict(step=final_step,source=s,id=(positive if s=='positive' else replay).rows[i]['id'],parent_id=(positive if s=='positive' else replay).rows[i]['parent_id']) for s,i in batches[index])
  history.append(dict(step=final_step,train_loss=sum(losses)/4,learning_rate=lr,gradient_norm=float(gradient),optimizer_seconds=elapsed,effective_batch_size=4,microbatch_size=1))
  if final_step%8==0:print(a.experiment,'step',final_step,'/',config['max_steps'],'loss',round(history[-1]['train_loss'],5),'optimizer_seconds',round(optimizer_seconds),flush=True)
  if final_step%config['evaluate_every']==0 or final_step==config['max_steps']:
   summary,preds=evaluate(m,v,dev,device=device)
   good=eligible(summary,baseline)
   path=folder/f"step_{final_step:05}.pt"
   isbest=good and (best is None or rank(summary)<rank(best))
   if isbest:best=summary;bestpath=str(path);stale=0
   else:stale+=1
   history[-1].update(development=summary,eligible=good,best=isbest)
   save_checkpoint(path,m,optimizer,config,final_step,v,dict(history=history,used=used,best=best,bestpath=bestpath,stale=stale))
   write('metrics/'+a.experiment+'_training_usage.csv',used)
   dump('metrics/'+a.experiment+'_training_history.json',history)
   dump('metrics/'+a.experiment+f'_step_{final_step:05}_development.json',summary)
   dump('checkpoints/'+a.experiment+'/latest.json',dict(path=str(path),step=final_step))
   print('VALIDATION',a.experiment,final_step,'positiveCER',summary['positive_development']['cer'],'cleanCER',summary['clean_validation']['cer'],'safe_improvement',good,'stale',stale,flush=True)
   if stale>=config['patience']:
    status='early_stopped';break
 # A non-improving candidate is evaluated honestly; E0 remains selectable.
 if bestpath is None and final_step:
  latest=json.loads((folder/'latest.json').read_text()) if (folder/'latest.json').exists() else None
  bestpath=latest['path'] if latest else None
 completed=dict(experiment=a.experiment,status=status,steps=final_step,configured_max_steps=config['max_steps'],sample_exposures=len(used),positive_exposures=sum(r['source']=='positive' for r in used),replay_exposures=sum(r['source']=='replay' for r in used),distinct_positive_tiers=len({r['id'] for r in used if r['source']=='positive'}),distinct_replay_tiers=len({r['id'] for r in used if r['source']=='replay'}),distinct_training_parents=len({r['parent_id'] for r in used}),optimizer_seconds=optimizer_seconds,wall_seconds=time.perf_counter()-start,candidate_checkpoint=bestpath,safe_development_improvement_found=best is not None,original_sha256=digest(CHECKPOINT),original_unchanged=digest(CHECKPOINT)==EXPECTED,failures=numerical_failures,config=config)
 dump('reports/'+a.experiment+'_completion.json',completed)
 print('EXPERIMENT COMPLETE',json.dumps(completed),flush=True)
if __name__=='__main__':main()

