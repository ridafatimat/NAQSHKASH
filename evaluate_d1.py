"""Frozen D1 evaluation; baseline OCR reused only for identical pixel hashes."""
from dcore import *
from pipeline_d1 import prepare,process_prepared
from preservation import audit
import csv,time
def main():
 outdir=ROOT/'reports/evaluation_v1'
 if outdir.exists():raise RuntimeError('Preserve previous evaluation; output already exists')
 selection=json.loads((ROOT/'configs/selection_lock.json').read_text())
 for p,h in selection['locked_files'].items():assert old.digest(p)==h,p
 old.verify_lock(code=True)
 history=check_history(json.loads((ROOT/'configs/protected_before.json').read_text()))
 assert history['passed'],history
 dump('reports/pre_evaluation_hash_check.json',history)
 parents={}
 for p in pages():
  k=(p['dataset'],p['parent_id'])
  assert k not in parents or parents[k]==p['partition']
  parents[k]=p['partition']
 outdir.mkdir()
 config=json.loads((ROOT/'configs/D1.json').read_text())
 import torch
 from models.crnn import CRNN
 from data.collate import talim_collate_fn
 from data.ctc_decode import greedy_ctc_decode
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
 vocab,vh=old.load_vocabulary(old.VOCABULARY);cp=torch.load(old.CHECKPOINT,map_location='cpu',weights_only=True)
 assert cp['char_to_idx']==vocab.char_to_index and cp['vocabulary_hash']==vh
 model=CRNN(66);model.load_state_dict(cp['model_state_dict']);model.eval()
 basepages=read(C/'reports/experiments_v1/pages.csv');basepreds=read(C/'reports/experiments_v1/predictions.csv')
 bpred={}
 for r in basepreds:
  if r['candidate'] in ['C4','C5']:bpred.setdefault((r['candidate'],*key(r)),[]).append(r)
 # Exact input hashing, not IDs/targets, controls recognition cache.
 def cropkey(crop):return hashlib.sha256(str(crop.shape).encode()+crop.tobytes()).hexdigest()
 cache={};hits=0;misses=0;src=Sources();records=[];predictions=[];details=[];replay=None;begin=time.perf_counter()
 def predict(crop,reuse=True):
  nonlocal hits,misses
  k=cropkey(crop)
  if reuse and k in cache:hits+=1;return cache[k]
  batch=talim_collate_fn([dict(image=torch.from_numpy(crop.astype(np.float32)/255).unsqueeze(0),label_text='',sample_id='pixels_only')])
  with torch.inference_mode():
   logits=model(batch['images']);assert torch.isfinite(logits).all()
   ids=logits.argmax(2)[:int(model.output_lengths(batch['image_widths'])[0]),0].tolist()
  pred=greedy_ctc_decode(ids,vocab)
  if reuse:cache[k]=pred;misses+=1
  return pred
 for index,p in enumerate(pages()):
  image,ref,entries=src.get(p);processor=Processor(image);baseline={}
  for name in ['C4','C5']:
   b=processor.run(old.CANDIDATES[name]);baseline[name]=b
   a=reference.assess(b,ref,entries)
   saved=next(r for r in basepages if r['candidate']==name and key(r)==key(p))
   assert b['accepted']==(saved['proposal']=='True') and a['correct']==(saved['correct_segmentation']=='True')
   if a['correct']:
    for t in a['tiers']:
     ps=[r for r in bpred.get((name,*key(p)),[]) if int(r['row'])==t['row'] and r['role']==t['role']]
     if not ps:continue
     x0,y0,x1,y1=t['box'];crop=b['image'][y0:y1,x0:x1];k=cropkey(crop)
     if k in cache:assert cache[k]==ps[0]['prediction']
     cache[k]=ps[0]['prediction']
  # prepare is independent of old proposal success; image-only features used by D1.
  pixels,M,extra=prepare(image);output=process_prepared(pixels,M,config,extra);a=reference.assess(output,ref,entries);components=audit(output,ref)
  rec={k:p[k] for k in ['partition','dataset','scope','parent_id','case_id','kind','augmentation']}
  rec.update(candidate='D1',proposal=output['accepted'],correct_segmentation=a['correct'],structure_correct=a['structure'],geometry_correct=a['geometry'],reason=a['reason'],row_count=len(output['rows']),expected_row_count=1+max(e['row'] for e in entries),inferred_row_count=output['diagnostics'].get('inferred_rows',len(output['rows'])),ocr_tiers=0,characters=0,substitutions=0,deletions=0,insertions=0,exact_tiers=0,page_exact=False,ctc_failures=0,**components)
  pending=[]
  if a['correct']:
   for t in a['tiers']:
    x0,y0,x1,y1=t['box'];crop=output['image'][y0:y1,x0:x1];truth=t['truth'];assert vocab.decode(vocab.encode(truth))==truth
    width=round(crop.shape[1]*64/crop.shape[0]);steps=int(model.output_lengths(torch.tensor([width]))[0])
    required=len(truth)+sum(x==y for x,y in zip(truth,truth[1:]))
    if required>steps:rec['ctc_failures']+=1;continue
    pred=predict(crop);counts=old.edit_counts(truth,pred)
    pending.append(dict(candidate='D1',**{k:p[k] for k in ['partition','dataset','scope','parent_id','case_id','kind','augmentation']},row=t['row'],role=t['role'],truth=truth,prediction=pred,leading_offset=t['leading_offset'],retention=t['retention'],contamination=t['contamination'],crop_sha256=cropkey(crop),resized_width=width,ctc_steps=steps,ctc_required=required,**counts))
    if replay is None:replay=(crop.copy(),pred)
   if not rec['ctc_failures']:
    predictions.extend(pending);rec.update(ocr_tiers=len(pending),characters=sum(len(t['truth']) for t in pending),exact_tiers=sum(t['truth']==t['prediction'] for t in pending),page_exact=all(t['truth']==t['prediction'] for t in pending))
    for name in ['substitutions','deletions','insertions']:rec[name]=sum(t[name] for t in pending)
  if p['dataset']=='positive' and p['partition']=='tuning' and a['correct'] and not rec['ctc_failures'] and components['components_lost']==0 and components['components_weak_content']==0:
   folder=ROOT/'derived_crops'/p['case_id'];folder.mkdir(parents=True,exist_ok=True)
   for t in a['tiers']:
    x0,y0,x1,y1=t['box'];assert cv2.imwrite(str(folder/f"row{t['row']}_{t['role']}.png"),output['image'][y0:y1,x0:x1])
  records.append(rec)
  details.append(dict(**{k:p[k] for k in ['partition','dataset','parent_id','case_id','kind']},accepted=output['accepted'],rows=output['rows'],matrix=output['matrix'].tolist(),diagnostics=output['diagnostics'],assessment=a,components=components))
  if (index+1)%40==0:print('D1',index+1,'/',len(pages()),'model misses',misses,'cache hits',hits,'seconds',round(time.perf_counter()-begin),flush=True)
 assert predict(replay[0],False)==replay[1]
 assert all(torch.equal(t,cp['model_state_dict'][name]) for name,t in model.state_dict().items())
 csvout('reports/evaluation_v1/pages.csv',records);csvout('reports/evaluation_v1/predictions.csv',predictions)
 dump('reports/evaluation_v1/diagnostics.json',details)
 dump('reports/evaluation_v1/completion.json',dict(pages=len(records),tiers=len(predictions),model_inputs=misses,exact_pixel_cache_hits=hits,uncached_replay_equal=True,weights_unchanged=True,checkpoint_sha256=old.digest(old.CHECKPOINT),vocabulary_hash=vh,elapsed_seconds=time.perf_counter()-begin,selection_unchanged=all(old.digest(p)==h for p,h in selection['locked_files'].items()),independent_test_claim=False))
 print('D1 EVALUATION COMPLETE',len(records),'pages',len(predictions),'tiers',flush=True)
if __name__=='__main__':main()

