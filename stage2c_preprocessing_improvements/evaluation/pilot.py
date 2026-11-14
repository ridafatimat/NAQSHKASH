"""Tuning-only segmentation pilot; no validation access and no OCR training."""
import sys,json,zipfile,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import *
from pipeline import Processor
import cv2,numpy as np
ref=load_local('stage2c_pilot_reference','evaluation/reference.py')
verify_lock();cv2.setNumThreads(1)
software={str(p):digest(p) for p in HERE.rglob('*.py')}
write_json(HERE/'protocol/software_lock_pilot.json',software)
parents=read_csv(HERE/'manifests/parents.csv');chosen=set()
for d in ('positive','general'):
 group=[r for r in parents if r['dataset']==d and r['partition']=='tuning'][:10]
 chosen.update((r['dataset'],r['parent_id']) for r in group)
pages=[r for r in read_csv(HERE/'manifests/pages.csv') if (r['dataset'],r['parent_id']) in chosen]
assert all(r['partition']=='tuning' for r in pages)
archives={d:dict(clean=zipfile.ZipFile(c['clean']),scanned=zipfile.ZipFile(c['scanned'])) for d,c in SOURCE_CONFIG.items() if c['scanned']}
rows=[];source_cache={};start=time.perf_counter()
for page in pages:
 d=page['dataset'];pid=page['parent_id'];key=(d,pid);c=SOURCE_CONFIG[d]
 if key not in source_cache:
  data=json.loads(archives[d]['clean'].read(c['prefix']+page['source_json']));image=cv2.imdecode(np.frombuffer(archives[d]['clean'].read(c['prefix']+page['source_image']),np.uint8),0)
  source_cache[key]=ref.clean_reference(image,data['logical_rows'])
 labels,entries=source_cache[key]
 image=cv2.imdecode(np.frombuffer(archives[d][page['kind'] if page['kind']=='clean' else 'scanned'].read(page['image_member']),np.uint8),0)
 reference=labels if page['kind']=='clean' else ref.augmented_reference(labels,json.loads(page['parameters']),image.shape)
 processor=Processor(image)
 for candidate,config in CANDIDATES.items():
  result=processor.run(config);assessment=ref.assess(result,reference,entries)
  rows.append(dict(candidate=candidate,dataset=d,kind=page['kind'],case_id=page['case_id'],proposed=result['accepted'],correct=assessment['correct'],reason=assessment['reason']))
from collections import Counter
summary={}
for candidate in CANDIDATES:
 selected=[r for r in rows if r['candidate']==candidate];summary[candidate]=dict(pages=len(selected),proposed=sum(r['proposed'] for r in selected),correct=sum(r['correct'] for r in selected),reasons=dict(Counter(r['reason'] for r in selected if not r['correct'])))
write_csv(HERE/'reports/pilot_segmentation.csv',rows);write_json(HERE/'reports/pilot_segmentation.json',dict(tuning_only=True,parents=20,pages=len(pages),elapsed_seconds=time.perf_counter()-start,candidates=summary))
print(json.dumps(summary,indent=2))
