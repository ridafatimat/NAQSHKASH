"""Freeze training/selection protocol after E0 and measured optimization preflight."""
from context import *
from metrics import aggregate
from collections import Counter
import math
def main():
 assert json.loads((E/'reports/preflight.json').read_text())['passed']
 if (E/'configs/software_lock.json').exists():raise RuntimeError('Training protocol already frozen')
 preds=read(E/'metrics/baseline_per_sample.csv');ids={r['id'] for r in read(E/'manifests/selection_development.csv')};subset=[r for r in preds if r['id'] in ids];summary={}
 for cohort in sorted({r['cohort'] for r in subset}):
  g=[r for r in subset if r['cohort']==cohort];summary[cohort]=dict(**aggregate(g),ctc_loss=sum(float(r['loss']) for r in g)/len(g))
 for name,fn in [('long_positive',lambda r:r['cohort']=='positive_development' and len(r['truth'])>=65),('short_positive',lambda r:r['cohort']=='positive_development' and len(r['truth'])<=4)]:
  g=[r for r in subset if fn(r)];summary[name]=dict(**aggregate(g),ctc_loss=sum(float(r['loss']) for r in g)/len(g) if g else None)
 dump('metrics/baseline_selection_summary.json',summary)
 pre=json.loads((E/'reports/preflight.json').read_text());ordinary=pre['records'][:2];wide=pre['records'][2]
 n=len(read(E/'manifests/positive_train.csv'));steps=math.ceil(n/4)
 # At most one E1 pass, same update/exposure budget E2. No validation outcome chooses budget.
 estimate_seconds=(sum(r['seconds'] for r in ordinary)/2)*(n-12)+wide['seconds']*12
 if estimate_seconds>1800:steps=min(128,steps)
 common=dict(seed=42,learning_rate=1e-5,optimizer='AdamW',weight_decay=1e-4,scheduler='cosine, floor10%, step-based',gradient_clip=1.,microbatch_size=1,effective_batch_size=4,max_steps=steps,evaluate_every=max(1,math.ceil(steps/4)),patience=2,checkpoint_width=2048,batchnorm_running_statistics='frozen; affine trainable',model='original CRNN unchanged',original_sha256=EXPECTED,selection='space-sensitive positive CER improves; macro-parent CER does not worsen; clean CER increase <=0.001 absolute, clean exact drop <=0.01; long CER increase <=0.005. Apply all clean cohorts when available. E0 fallback if no safe improvement.',claims='bounded local CPU fine-tuning study, exposed parent-disjoint development, no independent test')
 for name,mixed in [('baseline',None),('e1_positive_only',False),('e2_mixed_replay',True)]:
  dump('configs/'+name+'.yaml',dict(common,mixed_replay=mixed,train=False if name=='baseline' else True))
 dump('configs/training_design.json',dict(config=common,source_tiers=n,preflight_estimated_one_pass_seconds=estimate_seconds,max_positive_exposures_per_E1=steps*4,max_positive_exposures_per_E2=steps*2,max_replay_exposures_per_E2=steps*2,selection_subset_sha256=digest(E/'manifests/selection_development.csv'),full_evaluation_sha256=digest(E/'manifests/evaluation.csv'),independent_test_authorized=False,negative_handwritten_data=False,config_format='JSON is a valid YAML 1.2 subset; read without additional dependency'))
 files=list((E/'src').glob('*.py'))+list((E/'configs').glob('*.yaml'))+[E/'configs/training_design.json']
 dump('configs/software_lock.json',{str(p):digest(p) for p in files})
 print('TRAINING PROTOCOL FROZEN',steps,'max updates per run; preflight estimate',round(estimate_seconds),'seconds/one pass',flush=True)
if __name__=='__main__':main()

