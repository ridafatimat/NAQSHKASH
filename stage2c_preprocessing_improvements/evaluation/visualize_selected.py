"""Before/after panels for the already selected candidate; no new inference."""
import json,sys,zipfile,hashlib
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import *
from pipeline import Processor
ref=load_local('stage2c_visual_reference','evaluation/reference.py')

def main():
 root=HERE/'reports/experiments_v1';winner=json.loads((root/'selection_before_validation.json').read_text())['selected']
 rows=read_csv(root/'pages.csv');planned=read_csv(HERE/'manifests/pages.csv');lookup={(r['candidate'],r['dataset'],r['case_id']):r for r in rows}
 chosen={}
 for page in sorted(planned,key=lambda r:hashlib.sha256(('visual:42:'+r['case_id']).encode()).hexdigest()):
  if page['partition']=='tuning' and int(page['width'])<8000:continue
  baseline=lookup[('C0',page['dataset'],page['case_id'])];current=lookup[(winner,page['dataset'],page['case_id'])]
  event='recovered' if baseline['correct_segmentation']=='False' and current['correct_segmentation']=='True' else 'regression' if baseline['correct_segmentation']=='True' and current['correct_segmentation']=='False' else 'wrong_acceptance' if current['proposal']=='True' and current['correct_segmentation']=='False' else 'rejection' if current['proposal']=='False' else 'correct'
  if int(page['width'])>=8000:event='long_dense_'+event
  group=(page['dataset'],page['kind'],event)
  if group not in chosen:chosen[group]=page
 # A fixed maximum preserves diversity of cohorts/events without implying exhaustive review.
 selected=list(chosen.items())[:20]
 for group,page in chosen.items():
  if group[2].startswith('long_dense_') and not any(g[2].startswith('long_dense_') for g,p in selected):selected.append((group,page))
 destination=HERE/'examples/selected_candidate';destination.mkdir(exist_ok=True)
 index=['# Stage2C selected-candidate before/after examples\n\nThese are offline evaluation overlays. Blue boxes are accepted proposals, not guaranteed correct segmentation. Per-page status is stated below. Full-resolution and left/right detail images are saved for dot/stroke inspection.\n']
 review=[];preds=read_csv(root/'predictions.csv')
 for group,page in selected:
  dataset,kind,event=group;c=SOURCE_CONFIG[dataset]
  with zipfile.ZipFile(c['clean']) as clean:
   source=cv2.imdecode(np.frombuffer(clean.read(c['prefix']+page['source_image']),np.uint8),0)
   data=json.loads(clean.read(c['prefix']+page['source_json']))
  if kind=='clean':image=source
  else:
   with zipfile.ZipFile(c['scanned']) as scanned:image=cv2.imdecode(np.frombuffer(scanned.read(page['image_member']),np.uint8),0)
  folder=destination/(dataset+'_'+page['case_id']);folder.mkdir(exist_ok=True)
  cv2.imwrite(str(folder/'source.png'),image);processor=Processor(image);panels=[]
  for candidate in ['C0',winner]:
   result=processor.run(CANDIDATES[candidate]);frame=cv2.cvtColor(result['image'],cv2.COLOR_GRAY2BGR)
   for row in result['rows']:
    for tier in row:
     x0,y0,x1,y1=tier['box'];cv2.rectangle(frame,(x0,y0),(x1-1,y1-1),(220,70,20),1);cv2.putText(frame,tier['role'],(x0,max(10,y0-2)),cv2.FONT_HERSHEY_SIMPLEX,.35,(220,70,20),1)
   cv2.imwrite(str(folder/(candidate+'_full.png')),frame)
   detail=frame[:,:min(850,frame.shape[1])]
   if frame.shape[1]>1000:detail=np.hstack([detail,np.full((len(frame),15,3),255,np.uint8),frame[:,-250:]])
   cv2.imwrite(str(folder/(candidate+'_detail.png')),detail)
   scale=min(530/detail.shape[1],530/detail.shape[0]);thumb=cv2.resize(detail,None,fx=scale,fy=scale)
   panel=np.full((570,550,3),255,np.uint8);panel[30:30+thumb.shape[0],10:10+thumb.shape[1]]=thumb
   record=lookup[(candidate,dataset,page['case_id'])];label=candidate+' correct='+record['correct_segmentation']
   cv2.putText(panel,label,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1);panels.append(panel)
  cv2.imwrite(str(folder/'comparison.png'),np.hstack(panels))
  record=lookup[(winner,dataset,page['case_id'])]
  observed=[dict(row=r['row'],role=r['role'],truth=r['truth'],prediction=r['prediction']) for r in preds if r['candidate']==winner and r['dataset']==dataset and r['case_id']==page['case_id']]
  review.append(dict(case_id=page['case_id'],dataset=dataset,partition=page['partition'],kind=kind,event=event,candidate=winner,reason=record['reason'],expected=data['logical_rows'],evaluated_predictions=observed,folder=str(folder.relative_to(HERE))))
  index.append(f"\n## {dataset} / {page['case_id']} / {event}\n\nPartition: {page['partition']}; augmentation: {page['augmentation']}. {winner} reason: {record['reason'] or 'reference geometry passed'}.\n\n![comparison]({folder.name}/comparison.png)\n")
 write_json(destination/'review.json',review);(destination/'index.md').write_text(''.join(index),encoding='utf-8')
 print('Selected candidate panels:',len(review),winner)

if __name__=='__main__':main()
