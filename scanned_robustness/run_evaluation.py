"""Frozen-model Stage2B evaluation, only after the full integrity/leakage audit."""
import argparse,hashlib,json,time,zipfile
from collections import Counter,defaultdict
from pathlib import Path
import cv2,numpy as np,torch
from common import *
from modes import preprocess,blind_segment,assisted_segment,align_after_segmentation,SegmentationFailure
from models.crnn import CRNN
from data.collate import talim_collate_fn
from data.ctc_decode import greedy_ctc_decode


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=HERE/'reports/evaluation_v1');p.add_argument('--device',choices=['cpu','auto','cuda'],default='cpu');a=p.parse_args()
 if a.output.exists():p.error('Output already exists; never replace earlier results')
 audit=json.loads((HERE/'reports/dataset_audit.json').read_text());assert audit['status']=='passed'
 pages=read_csv(HERE/'manifests/evaluation_pages.csv');parents={(r['dataset'],r['parent_id']):r for r in read_csv(HERE/'manifests/parents.csv')}
 assert len(pages)==audit['selected_pages']
 assert all(r['cohort'] in {'positive_frozen_test','general_stage1_test','general_unassigned_exploratory','general_test_structure_match_exploratory'} for r in pages)
 assert all(not set(r['stage1_splits'].split(';')).intersection({'train','validation'}) for r in pages)
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(42);torch.use_deterministic_algorithms(True);cv2.setNumThreads(1)
 device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device)
 vocabulary,vocab_hash=load_vocabulary(VOCABULARY)
 checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
 assert checkpoint['char_to_idx']==vocabulary.char_to_index and checkpoint['vocabulary_hash']==vocab_hash
 assert sha256(CHECKPOINT)=='22450cbddd42968149739ea0293a6f4ad105f593e82387d707ecb6418a6bac0f'
 model=CRNN(66).to(device);model.load_state_dict(checkpoint['model_state_dict'],strict=True);model.eval()
 a.output.mkdir(parents=True);(a.output/'examples').mkdir()
 caches={};cache_hits=0;example_keys=set();examples=[];checks={};prediction_rows=[];clean_rows=[];page_rows=[];clean_page_rows=[];source_cache={};start=time.perf_counter()
 archives={d:zipfile.ZipFile(c['scanned']) for d,c in DATASETS.items()};clean_archives={d:zipfile.ZipFile(c['clean']) for d,c in DATASETS.items()}
 def predict(image,use_cache=True):
  nonlocal cache_hits
  key=hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
  if use_cache and key in caches:cache_hits+=1;return caches[key]
  batch=talim_collate_fn([dict(image=torch.from_numpy(image.astype(np.float32)/255).unsqueeze(0),label_text='',sample_id='inference')])
  with torch.inference_mode():
   logits=model(batch['images'].to(device));assert torch.isfinite(logits).all()
   ids=logits.argmax(2)[:int(model.output_lengths(batch['image_widths'])[0]),0].cpu().tolist()
  text=greedy_ctc_decode(ids,vocabulary)
  if use_cache:caches[key]=text
  return text
 def extract(processed,data,mode):
  proposed=blind_segment(processed) if mode=='blind' else assisted_segment(processed,data['logical_rows'])
  return align_after_segmentation(proposed,data['logical_rows'],vocabulary)
 def example(page,mode,event,raw,description,truth,prediction='',tier_image=None):
  key=(page['cohort'],mode,page['augmentation'],event)
  if key in example_keys:return
  example_keys.add(key);stem=f"{page['dataset']}_{page['sample_id']}_{mode}_{event}"
  image_file=a.output/'examples'/(stem+('.png' if tier_image is not None else '.jpg'))
  if tier_image is not None:cv2.imwrite(str(image_file),tier_image)
  else:image_file.write_bytes(raw)
  examples.append(dict(dataset=page['dataset'],cohort=page['cohort'],parent_id=page['parent_id'],sample_id=page['sample_id'],mode=mode,
    augmentation=page['augmentation'],event=event,image_path=str(image_file.relative_to(a.output)),truth=truth,prediction=prediction,description=description))
 for index,page in enumerate(pages):
  dataset=page['dataset'];pid=page['parent_id'];config=DATASETS[dataset];parent=parents[(dataset,pid)];cache_key=(dataset,pid)
  if cache_key not in source_cache:
   clean_raw=clean_archives[dataset].read(config['clean_prefix']+parent['source_image']);assert hashlib.sha256(clean_raw).hexdigest()==page['source_sha256']
   data=json.loads(clean_archives[dataset].read(config['clean_prefix']+parent['source_json']))
   clean_image=cv2.imdecode(np.frombuffer(clean_raw,np.uint8),0)
   clean_modes={}
   try:clean_processed=preprocess(clean_image);clean_preprocess_error=''
   except Exception as exc:clean_preprocess_error=type(exc).__name__+':'+str(exc)
   for mode in ('blind','assisted'):
    status=dict(dataset=dataset,cohort=page['cohort'],parent_id=pid,mode=mode,status='success',reason='',tiers=0)
    try:
     if clean_preprocess_error:raise SegmentationFailure('preprocessing:'+clean_preprocess_error)
     tiers=extract(clean_processed,data,mode);records={}
     for tier in tiers:
      pred=predict(tier['image']);key=(tier['row_index'],tier['tier_name'])
      record=dict(dataset=dataset,cohort=page['cohort'],parent_id=pid,mode=mode,row_index=tier['row_index'],tier_name=tier['tier_name'],truth=tier['truth'],prediction=pred,**edit_counts(tier['truth'],pred))
      clean_rows.append(record);records[key]=record
     clean_modes[mode]=records;status['tiers']=len(tiers)
    except (ValueError,RuntimeError,cv2.error) as exc:
     status.update(status='failed',reason=str(exc));clean_modes[mode]={}
    clean_page_rows.append(status)
   source_cache[cache_key]=(data,clean_modes)
  data,clean_modes=source_cache[cache_key]
  raw=archives[dataset].read(page['image_member']);assert hashlib.sha256(raw).hexdigest()==page['augmented_sha256']
  image=cv2.imdecode(np.frombuffer(raw,np.uint8),0)
  try:processed=preprocess(image);preprocess_error=''
  except Exception as exc:preprocess_error=type(exc).__name__+':'+str(exc)
  for mode in ('blind','assisted'):
   status=dict(dataset=dataset,cohort=page['cohort'],sample_id=page['sample_id'],parent_id=pid,mode=mode,augmentation=page['augmentation'],
     operations=page['operations'],severity=page['severity'],status='success',reason='',failure_stage='',tiers=0,characters=0,detected_rows=0,expected_rows=len(data['logical_rows']))
   try:
    if preprocess_error:raise SegmentationFailure('preprocessing:'+preprocess_error)
    status['detected_rows']=len(processed.rows)
    tiers=extract(processed,data,mode)
    pending=[]
    for tier in tiers:
     pred=predict(tier['image']);truth=tier['truth'];clean_record=clean_modes[mode].get((tier['row_index'],tier['tier_name']))
     row=dict(dataset=dataset,cohort=page['cohort'],sample_id=page['sample_id'],parent_id=pid,mode=mode,augmentation=page['augmentation'],operations=page['operations'],severity=page['severity'],
       row_index=tier['row_index'],tier_name=tier['tier_name'],truth=truth,prediction=pred,raw_label=tier['raw_label'],leading_offset=tier['leading_offset'],resized_width=tier['resized_width'],
       paired_clean_available=clean_record is not None,clean_prediction='' if clean_record is None else clean_record['prediction'],**edit_counts(truth,pred))
     pending.append(row)
     if truth!=pred:example(page,mode,'recognition',raw,'Structurally aligned tier differs from label; residual crop/warp effects may contribute.',truth,pred,tier['image'])
     check_key=(page['cohort'],mode)
     if check_key not in checks:checks[check_key]=(tier['image'].copy(),pred)
    prediction_rows.extend(pending);status['tiers']=len(tiers);status['characters']=sum(len(t['truth']) for t in tiers)
   except (ValueError,RuntimeError,cv2.error) as exc:
    reason=str(exc);status.update(status='failed',reason=reason,failure_stage='preprocessing' if reason.startswith('preprocessing:') else 'alignment' if reason.startswith('alignment_') else 'ctc' if reason.startswith('ctc_') else 'segmentation')
    truth='\n\n'.join('\n'.join((r['upper'],r['count_line'],r['lower'])) for r in data['logical_rows'])
    example(page,mode,'segmentation',raw,reason,truth)
   page_rows.append(status)
  if (index+1)%10==0:print('evaluated pages',index+1,'/',len(pages),'tier predictions',len(prediction_rows),'elapsed_s',round(time.perf_counter()-start,1),flush=True)
 # Independently replay uncached model inference, one successful tier per cohort/mode.
 replay=[]
 for key,(image,previous) in checks.items():
  repeated=predict(image,use_cache=False);assert repeated==previous
  replay.append(dict(cohort=key[0],mode=key[1],prediction_equal=True))
 assert all(torch.equal(value.detach().cpu(),checkpoint['model_state_dict'][name]) for name,value in model.state_dict().items()),'Model state changed'
 write_csv(a.output/'predictions.csv',prediction_rows)
 write_csv(a.output/'page_processing.csv',page_rows)
 write_csv(a.output/'segmentation_failures.csv',[r for r in page_rows if r['status']=='failed'],fields=list(page_rows[0]))
 write_csv(a.output/'recognition_errors.csv',[r for r in prediction_rows if r['truth']!=r['prediction']],fields=list(prediction_rows[0]))
 write_csv(a.output/'clean_predictions.csv',clean_rows)
 write_csv(a.output/'clean_page_processing.csv',clean_page_rows)
 write_csv(a.output/'failure_examples.csv',examples,fields=list(examples[0]) if examples else ['sample_id','mode','description'])
 groups={}
 for cohort in sorted({r['cohort'] for r in pages}):
  groups[cohort]={}
  for mode in ('blind','assisted'):
   subset=[r for r in prediction_rows if r['cohort']==cohort and r['mode']==mode];page_subset=[r for r in page_rows if r['cohort']==cohort and r['mode']==mode]
   entry=dict(overall=metrics(subset),pages_attempted=len(page_subset),pages_processed=sum(r['status']=='success' for r in page_subset),pages_failed=sum(r['status']=='failed' for r in page_subset),
      parents_attempted=len({r['parent_id'] for r in page_subset}),failure_reasons=dict(Counter(r['reason'] for r in page_subset if r['status']=='failed')),
      clean=metrics([r for r in clean_rows if r['cohort']==cohort and r['mode']==mode]))
   entry['per_tier']={tier:metrics([r for r in subset if r['tier_name']==tier]) for tier in ('upper','count','lower')}
   for field in ('augmentation','severity'):
    entries={}
    for value in sorted({r[field] for r in page_subset}):
     records=[r for r in subset if r[field]==value];ps=[r for r in page_subset if r[field]==value]
     entries[value]=dict(metrics=metrics(records),pages_attempted=len(ps),pages_processed=sum(r['status']=='success' for r in ps))
    entry['per_'+field]=entries
   entry['per_operation']={}
   for op in sorted({op for r in page_subset for op in r['operations'].split(';')}):
    ps=[r for r in page_subset if op in r['operations'].split(';')];records=[r for r in subset if op in r['operations'].split(';')]
    entry['per_operation'][op]=dict(metrics=metrics(records),pages_attempted=len(ps),pages_processed=sum(r['status']=='success' for r in ps))
   paired=[r for r in subset if r['paired_clean_available']]
   paired_clean=[dict(truth=r['truth'],prediction=r['clean_prediction'],**edit_counts(r['truth'],r['clean_prediction'])) for r in paired]
   entry['paired_scanned']=metrics(paired);entry['paired_clean']=metrics(paired_clean)
   groups[cohort][mode]=entry
 result=dict(checkpoint=str(CHECKPOINT),checkpoint_sha256=sha256(CHECKPOINT),vocabulary_hash=vocab_hash,classes=66,blank=0,model_state_unchanged=True,
    device=str(device),batch_size=1,cpu_threads=4,deterministic_algorithms=True,inference_manifest_sha256=sha256(HERE/'manifests/evaluation_pages.csv'),
    mode_policy='Blind proposals use pixels only, then GT structural validation may reject but never retry/choose proposals. Assisted uses annotated row count and tier presence.',
    page_policy='Reject whole page from OCR metrics if any row/tier alignment, segmentation or CTC feasibility fails; retain separate failure accounting.',
    clean_policy='Same frozen model, same mode and preprocessing on exact clean parents. Paired metrics repeat clean predictions per successfully aligned scanned tier instance.',
    groups=groups,uncached_determinism_replays=replay,prediction_cache_hits=cache_hits,unique_model_inputs=len(caches),elapsed_seconds=time.perf_counter()-start)
 (a.output/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 flat=[]
 for cohort,modes in groups.items():
  for mode,entry in modes.items():flat.append(dict(cohort=cohort,mode=mode,pages_attempted=entry['pages_attempted'],pages_processed=entry['pages_processed'],pages_failed=entry['pages_failed'],**entry['overall']))
 write_csv(a.output/'metrics.csv',flat)
 for z in list(archives.values())+list(clean_archives.values()):z.close()
 print(json.dumps(flat,indent=2))

if __name__=='__main__':main()
