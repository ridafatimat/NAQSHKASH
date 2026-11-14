"""Controlled C0-C5 comparisons. Selection is locked before validation/regression."""
import argparse,csv,hashlib,json,sys,time,zipfile
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import *
from pipeline import Processor
import cv2,numpy as np,torch
from models.crnn import CRNN
from data.collate import talim_collate_fn
from data.ctc_decode import greedy_ctc_decode
reference_module=load_local('stage2c_run_reference','evaluation/reference.py')


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'reports/experiments_v1');a=p.parse_args()
 if a.output.exists():p.error('Output exists; preserve previous experiments')
 lock=verify_lock(code=True);a.output.mkdir(parents=True);(a.output/'examples').mkdir()
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(42);torch.use_deterministic_algorithms(True);cv2.setNumThreads(1)
 vocab,vocab_hash=load_vocabulary(VOCABULARY);checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
 assert checkpoint['char_to_idx']==vocab.char_to_index and checkpoint['vocabulary_hash']==vocab_hash
 model=CRNN(66);model.load_state_dict(checkpoint['model_state_dict'],strict=True);model.eval()
 pages=read_csv(HERE/'manifests/pages.csv');configs=json.loads((HERE/'protocol/candidates.json').read_text())
 archives={d:dict(clean=zipfile.ZipFile(c['clean']),**({'scanned':zipfile.ZipFile(c['scanned'])} if c['scanned'] else {})) for d,c in SOURCE_CONFIG.items()}
 caches={};cache_hits=0;replays={};page_records=[];prediction_records=[];geometry_records=[];examples=[];example_groups=Counter();source_key=None;start=time.perf_counter();ordinal=0
 pagefields=['candidate','partition','dataset','scope','parent_id','case_id','kind','augmentation','proposal','correct_segmentation','structure_correct','geometry_correct','reason','row_count','tier_count','ocr_tiers','characters','substitutions','deletions','insertions','exact_tiers','page_exact','ctc_failures','geometry_applied','role_margin','seconds','ordinal']
 predfields=['candidate','partition','dataset','scope','parent_id','case_id','kind','augmentation','row','role','truth','prediction','leading_offset','retention','contamination','substitutions','deletions','insertions']
 pf=(a.output/'pages.csv').open('w',newline='',encoding='utf-8');pw=csv.DictWriter(pf,fieldnames=pagefields);pw.writeheader()
 tf=(a.output/'predictions.csv').open('w',newline='',encoding='utf-8');tw=csv.DictWriter(tf,fieldnames=predfields);tw.writeheader()
 def predict(image,cache=True):
  nonlocal cache_hits
  key=hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
  if cache and key in caches:cache_hits+=1;return caches[key]
  batch=talim_collate_fn([dict(image=torch.from_numpy(image.astype(np.float32)/255).unsqueeze(0),label_text='',sample_id='pixels_only')])
  with torch.inference_mode():
   logits=model(batch['images']);assert bool(torch.isfinite(logits).all())
   ids=logits.argmax(2)[:int(model.output_lengths(batch['image_widths'])[0]),0].tolist()
  prediction=greedy_ctc_decode(ids,vocab)
  if cache:caches[key]=prediction
  return prediction
 def gallery(page,image,results,assessments,entries):
  base=assessments['C0'];new=assessments['C5'];res=results['C5']
  event='recovered' if not base['correct'] and new['correct'] else 'regression' if base['correct'] and not new['correct'] else 'wrong_acceptance' if res['accepted'] and not new['correct'] else 'rejection' if not res['accepted'] else 'correct'
  group=(page['partition'],page['dataset'],page['kind'],event)
  if example_groups[group]>=1:return
  example_groups[group]+=1
  folder=a.output/'examples'/f"{page['dataset']}_{page['case_id']}_{event}";folder.mkdir()
  cv2.imwrite(str(folder/'source.png'),image)
  panels=[]
  for name in ('C0','C2','C5'):
   output=results[name];annotated=cv2.cvtColor(output['image'],cv2.COLOR_GRAY2BGR)
   color=(0,160,0) if assessments[name]['correct'] else (0,0,220)
   for row in output['rows']:
    for tier in row:
     x0,y0,x1,y1=tier['box'];cv2.rectangle(annotated,(x0,y0),(x1-1,y1-1),color,1);cv2.putText(annotated,tier['role'],(x0,max(10,y0-2)),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
   cv2.imwrite(str(folder/(name+'_full.png')),annotated)
   detail=annotated[:,:min(900,annotated.shape[1])];cv2.imwrite(str(folder/(name+'_detail.png')),detail)
   s=min(450/detail.shape[1],500/detail.shape[0]);thumb=cv2.resize(detail,None,fx=s,fy=s)
   panel=np.full((550,470,3),255,np.uint8);panel[35:35+thumb.shape[0],10:10+thumb.shape[1]]=thumb
   cv2.putText(panel,name+' '+('correct' if assessments[name]['correct'] else assessments[name]['reason'][:25]),(8,20),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1);panels.append(panel)
  cv2.imwrite(str(folder/'comparison.png'),np.hstack(panels))
  examples.append(dict(partition=page['partition'],dataset=page['dataset'],case_id=page['case_id'],kind=page['kind'],augmentation=page['augmentation'],event=event,path=str(folder.relative_to(a.output)),expected=[{k:e[k] for k in ('row','role','truth')} for e in entries]))
 for partition in ('tuning','validation','clean_regression'):
  subset=[r for r in pages if r['partition']==partition]
  if partition=='validation':
   tune=[r for r in page_records if r['partition']=='tuning' and r['kind']=='scanned']
   eligible_sets=[{r['dataset']+'/'+r['case_id'] for r in tune if r['candidate']==c and r['correct_segmentation'] and r['ocr_tiers']>0} for c in configs]
   common=set.intersection(*eligible_sets)
   ranking=[]
   for c in configs:
    group=[r for r in tune if r['candidate']==c]
    preds=[r for r in prediction_records if r['candidate']==c and r['partition']=='tuning' and r['kind']=='scanned' and r['dataset']+'/'+r['case_id'] in common]
    common_cer=aggregate(preds)['cer'] if preds else None
    ranking.append(dict(candidate=c,correct_pages=sum(r['correct_segmentation'] for r in group),attempted=len(group),wrong_acceptances=sum(r['proposal'] and not r['correct_segmentation'] for r in group),common_cer=common_cer))
   ranking.sort(key=lambda r:(-r['correct_pages'],r['wrong_acceptances'],r['common_cer'] if r['common_cer'] is not None else 1e10,r['candidate']))
   selection=dict(selected=ranking[0]['candidate'],ranking=ranking,selection_inputs_partition='tuning',common_scanned_pages=len(common),completed_page_evaluations=len(page_records),validation_started=False,
    protocol_sha256=digest(HERE/'protocol/lock.json'),software_lock_sha256=digest(HERE/'protocol/software_lock.json'))
   write_json(a.output/'selection_before_validation.json',selection)
   print('FROZEN SELECTION BEFORE VALIDATION',json.dumps(selection),flush=True)
  for index,page in enumerate(subset):
   key=(page['dataset'],page['parent_id']);config=SOURCE_CONFIG[key[0]]
   if key!=source_key:
    raw=archives[key[0]]['clean'].read(config['prefix']+page['source_image']);assert hashlib.sha256(raw).hexdigest()==page['source_sha256']
    clean_image=cv2.imdecode(np.frombuffer(raw,np.uint8),0);data=json.loads(archives[key[0]]['clean'].read(config['prefix']+page['source_json']))
    labels,entries=reference_module.clean_reference(clean_image,data['logical_rows']);source_key=key
   raw=archives[key[0]][page['kind']].read(page['image_member']);assert hashlib.sha256(raw).hexdigest()==page['image_sha256']
   image=cv2.imdecode(np.frombuffer(raw,np.uint8),0);assert image is not None
   reference=labels if page['kind']=='clean' else reference_module.augmented_reference(labels,json.loads(page['parameters']),image.shape)
   processor=Processor(image);results={};assessments={}
   for candidate,configuration in configs.items():
    ordinal+=1;begin=time.perf_counter();result=processor.run(configuration);assessment=reference_module.assess(result,reference,entries)
    results[candidate]=result;assessments[candidate]=assessment
    row=dict(candidate=candidate,partition=partition,dataset=page['dataset'],scope=page['scope'],parent_id=page['parent_id'],case_id=page['case_id'],kind=page['kind'],augmentation=page['augmentation'],
      proposal=result['accepted'],correct_segmentation=assessment['correct'],structure_correct=assessment['structure'],geometry_correct=assessment['geometry'],reason=assessment['reason'],row_count=len(result['rows']),tier_count=sum(len(r) for r in result['rows']),
      ocr_tiers=0,characters=0,substitutions=0,deletions=0,insertions=0,exact_tiers=0,page_exact=False,ctc_failures=0,geometry_applied=result['diagnostics'].get('perspective_applied',False),role_margin=result['diagnostics'].get('role_margin',''),seconds=0.,ordinal=ordinal)
    if assessment['correct']:
     pending=[]
     for checked in assessment['tiers']:
      x0,y0,x1,y1=checked['box'];crop=result['image'][y0:y1,x0:x1];truth=checked['truth'];vocab.encode(truth)
      width=max(1,round(crop.shape[1]*64/crop.shape[0]));required=len(truth)+sum(a==b for a,b in zip(truth,truth[1:]))
      if width<4 or required>width//4:row['ctc_failures']+=1;continue
      prediction=predict(crop);counts=edit_counts(truth,prediction)
      pending.append(dict(candidate=candidate,partition=partition,dataset=page['dataset'],scope=page['scope'],parent_id=page['parent_id'],case_id=page['case_id'],kind=page['kind'],augmentation=page['augmentation'],row=checked['row'],role=checked['role'],truth=truth,prediction=prediction,leading_offset=checked['leading_offset'],retention=checked['retention'],contamination=checked['contamination'],**counts))
      if candidate not in replays:replays[candidate]=(crop.copy(),prediction)
     if row['ctc_failures']==0:
      for pred in pending:tw.writerow(pred);prediction_records.append(pred)
      row.update(ocr_tiers=len(pending),characters=sum(len(r['truth']) for r in pending),exact_tiers=sum(r['truth']==r['prediction'] for r in pending),page_exact=all(r['truth']==r['prediction'] for r in pending))
      for name in ('substitutions','deletions','insertions'):row[name]=sum(r[name] for r in pending)
    row['seconds']=time.perf_counter()-begin;pw.writerow(row);page_records.append(row)
   gallery(page,image,results,assessments,entries)
   if (index+1)%10==0:
    pf.flush();tf.flush();print(partition,index+1,'/',len(subset),'OCR records',len(prediction_records),'unique inputs',len(caches),'elapsed',round(time.perf_counter()-start,1),flush=True)
  pf.flush();tf.flush()
  print('PHASE COMPLETE',partition,flush=True)
 checks=[]
 for candidate,(image,original) in replays.items():
  assert predict(image,cache=False)==original;checks.append(dict(candidate=candidate,uncached_prediction_equal=True))
 assert all(torch.equal(t,checkpoint['model_state_dict'][name]) for name,t in model.state_dict().items())
 pf.close();tf.close()
 for group in archives.values():
  for z in group.values():z.close()
 verify_lock(code=True)
 write_json(a.output/'completion.json',dict(page_count=len(pages),candidate_page_evaluations=len(page_records),prediction_count=len(prediction_records),model_state_unchanged=True,determinism_replays=checks,
    cache_hits=cache_hits,unique_model_inputs=len(caches),elapsed_seconds=time.perf_counter()-start,checkpoint_sha256=digest(CHECKPOINT),vocabulary_hash=vocab_hash,validation_selection_unchanged=True))
 print('ALL EXPERIMENTS COMPLETE',len(page_records),'page-candidates; seconds',round(time.perf_counter()-start,1),flush=True)

if __name__=='__main__':main()
