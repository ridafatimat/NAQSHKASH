"""Paired frozen-input recognition evaluation; no segmentation selection."""
from context import *
from datasets import FrozenTiers
from metrics import measurements,aggregate
from collections import defaultdict
def evaluate(model,vocab,rows,output=None,device='cpu',progress=False):
 model.eval();dataset=FrozenTiers(rows);cache={};predictions=[];criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False);encoder=CTCLabelEncoder(vocab);begin=time.perf_counter()
 for i,r in enumerate(dataset.rows):
  before=time.perf_counter();key=(r['input_hash'],r['truth'])
  if key in cache:prediction,loss=cache[key];cached=True
  else:
   batch=talim_collate_fn([dataset[i]]);encoded=encoder.encode_batch(batch['label_texts'])
   with torch.inference_mode():
    logits=model(batch['images'].to(device));lengths,required=runtime_ctc_lengths(model,batch,encoded,logits.shape[0])
    loss,_=fp32_ctc_loss(logits,encoded['targets'].to(device),lengths,encoded['target_lengths'],criterion)
    assert torch.isfinite(loss) and torch.isfinite(logits).all()
    ids=logits.argmax(2)[:int(lengths[0]),0].tolist();prediction=greedy_ctc_decode(ids,vocab)
   loss=float(loss);cache[key]=(prediction,loss);cached=False
  predictions.append(dict(id=r['id'],cohort=r['cohort'],parent_id=r['parent_id'],case_id=r['case_id'],kind=r['kind'],augmentation=r['augmentation'],role=r['role'],truth=r['truth'],prediction=prediction,input_hash=r['input_hash'],resized_width=r['resized_width'],ctc_required=r['ctc_required'],loss=loss,cached=cached,seconds=time.perf_counter()-before,**measurements(r['truth'],prediction)))
  if progress and (i+1)%100==0:print('Inference',i+1,'/',len(dataset),'seconds',round(time.perf_counter()-begin),flush=True)
 summaries={}
 for cohort in sorted({r['cohort'] for r in predictions}):
  g=[r for r in predictions if r['cohort']==cohort];summaries[cohort]=dict(**aggregate(g),ctc_loss=sum(r['loss'] for r in g)/len(g))
 for key,fn in [('long_positive',lambda r:r['cohort']=='positive_development' and len(r['truth'])>=65),('short_positive',lambda r:r['cohort']=='positive_development' and len(r['truth'])<=4)]:
  g=[r for r in predictions if fn(r)];summaries[key]=dict(**aggregate(g),ctc_loss=sum(r['loss'] for r in g)/len(g) if g else None)
 summaries['runtime']=dict(seconds=time.perf_counter()-begin,unique_model_inputs=len(cache),cached_tiers=len(predictions)-len(cache),device=device,average_uncached_seconds=sum(r['seconds'] for r in predictions if not r['cached'])/len(cache))
 if output:
  write('metrics/'+output+'_per_sample.csv',predictions);dump('metrics/'+output+'_summary.json',summaries)
 return summaries,predictions
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,default=CHECKPOINT);p.add_argument('--name',default='baseline');a=p.parse_args()
 if (E/'metrics'/(a.name+'_per_sample.csv')).exists():raise FileExistsError('Preserve existing evaluation: '+a.name)
 seed();verify();model,v,state=model_load(a.checkpoint);summary,preds=evaluate(model,v,read(E/'manifests/evaluation.csv'),a.name,progress=True)
 dump('reports/'+a.name+'_lineage.json',dict(checkpoint=str(a.checkpoint),sha256=digest(a.checkpoint),original_sha256=EXPECTED,input_manifest_sha256=digest(E/'manifests/evaluation.csv'),decoder_sha256=digest(PROJECT/'data/ctc_decode.py')))
 if a.name=='baseline':
  (E/'reports/baseline_evaluation.md').write_text('# Original checkpoint E0 baseline\n\n'+json.dumps(summary,indent=2)+'\n\nThese are fixed parent-disjoint exposed development/regression inputs, not an independent test. Full page-level segmentation rejections/wrong acceptances are retained in manifests/page_outcomes.csv; CER is conditional on independently correct frozen D1 crops. No hard evaluable tier is dropped or truncated. Space-normalized CER is diagnostic only.\n',encoding='utf-8')
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

