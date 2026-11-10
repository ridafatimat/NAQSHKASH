"""Pre-registered per-component validation diagnostic; no candidate reselection."""
import json,sys,zipfile
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import *
from pipeline import Processor
reference=load_local('stage2c_component_reference','evaluation/reference.py')

def main():
 verify_lock(code=True);cv2.setNumThreads(1)
 design=json.loads((HERE/'protocol/component_audit_design.json').read_text())
 assert design['registered_before_validation'] and design['manifest_sha256']==digest(HERE/'manifests/component_audit_pages.csv')
 result_root=HERE/'reports/experiments_v1';selection=json.loads((result_root/'selection_before_validation.json').read_text());selected=selection['selected']
 candidates=list(dict.fromkeys(['C0',selected]));cases=read_csv(HERE/'manifests/component_audit_pages.csv')
 archives={d:dict(clean=zipfile.ZipFile(c['clean']),**({'scanned':zipfile.ZipFile(c['scanned'])} if c['scanned'] else {})) for d,c in SOURCE_CONFIG.items()}
 records=[];component_rows=[];source_cache={}
 for i,page in enumerate(cases):
  d=page['dataset'];c=SOURCE_CONFIG[d];key=(d,page['parent_id'])
  if key not in source_cache:
   source=cv2.imdecode(np.frombuffer(archives[d]['clean'].read(c['prefix']+page['source_image']),np.uint8),0)
   data=json.loads(archives[d]['clean'].read(c['prefix']+page['source_json']));source_cache[key]=reference.clean_reference(source,data['logical_rows'])
  labels,entries=source_cache[key];image=cv2.imdecode(np.frombuffer(archives[d][page['kind']].read(page['image_member']),np.uint8),0)
  expected=labels if page['kind']=='clean' else reference.augmented_reference(labels,json.loads(page['parameters']),image.shape)
  # Components are counted on ideal transformed ink, not on a candidate's mask.
  components=[]
  for entry in entries:
   count,component_map,stats,_=cv2.connectedComponentsWithStats((expected==entry['id']).astype(np.uint8))
   for index in range(1,count):
    if stats[index,cv2.CC_STAT_AREA]<3:continue
    ys,xs=np.where(component_map==index);components.append((entry,np.column_stack([xs,ys]).astype(np.float32)))
  proc=Processor(image)
  for candidate in candidates:
   output=proc.run(CANDIDATES[candidate]);assessment=reference.assess(output,expected,entries)
   aligned=output['accepted'] and assessment['structure'];losses=0;evaluated=0;min_retention=1.
   if aligned:
    tiers={(t['row'],t['role']):t for row in output['rows'] for t in row}
    for component_index,(entry,points) in enumerate(components):
     tier=tiers[(entry['row'],entry['role'])];x0,y0,x1,y1=tier['box']
     transformed=cv2.perspectiveTransform(points[None],output['matrix'])[0]
     retention=float(((transformed[:,0]>=x0)&(transformed[:,0]<x1)&(transformed[:,1]>=y0)&(transformed[:,1]<y1)).mean())
     losses+=retention<.95;evaluated+=1;min_retention=min(min_retention,retention)
     if retention<.95:component_rows.append(dict(candidate=candidate,dataset=d,case_id=page['case_id'],kind=page['kind'],row=entry['row'],role=entry['role'],component=component_index,pixels=len(points),retention=retention))
   records.append(dict(candidate=candidate,partition=page['partition'],dataset=d,case_id=page['case_id'],kind=page['kind'],proposal=output['accepted'],structure_correct=aligned,aggregate_geometry_correct=assessment['correct'],components_evaluated=evaluated,components_with_loss=losses,min_component_retention=min_retention if evaluated else '',all_components_preserved=aligned and losses==0))
  if (i+1)%10==0:print('Component diagnostic',i+1,'/',len(cases),flush=True)
 write_csv(result_root/'component_audit_pages.csv',records)
 write_csv(result_root/'component_loss_examples.csv',component_rows,fields=['candidate','dataset','case_id','kind','row','role','component','pixels','retention'])
 summary=[]
 for candidate in candidates:
  group=[r for r in records if r['candidate']==candidate]
  summary.append(dict(candidate=candidate,pages=len(group),structurally_aligned_pages=sum(r['structure_correct'] for r in group),components_evaluated=sum(r['components_evaluated'] for r in group),components_with_loss=sum(r['components_with_loss'] for r in group),pages_all_components_preserved=sum(r['all_components_preserved'] for r in group),aggregate_pass_pages_with_component_loss=sum(r['aggregate_geometry_correct'] and r['components_with_loss']>0 for r in group)))
 write_json(result_root/'component_audit.json',dict(design=design,selected_before_audit=selected,candidates=summary,selection_unchanged=True))
 for group in archives.values():
  for z in group.values():z.close()
 print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
