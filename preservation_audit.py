"""Fixed historical 90-page component diagnostic, with C5 and D1 added."""
from dcore import *
from pipeline_d1 import process_image
from preservation import audit
def main():
 cases=read(C/'manifests/component_audit_pages.csv');src=Sources();config=json.loads((ROOT/'configs/D1.json').read_text());records=[]
 for i,p in enumerate(cases):
  image,ref,entries=src.get(p);processor=Processor(image)
  for name in ['C4','C5','D1']:
   out=process_image(image,config) if name=='D1' else processor.run(old.CANDIDATES[name])
   a=reference.assess(out,ref,entries)
   records.append(dict(candidate=name,**{k:p[k] for k in ['dataset','partition','parent_id','case_id','kind']},proposal=out['accepted'],correct=a['correct'],structure=a['structure'],reason=a['reason'],**audit(out,ref)))
  if (i+1)%30==0:print('Preservation diagnostic',i+1,'/',len(cases),flush=True)
 csvout('reports/preservation_audit.csv',records);summary=[]
 for name in ['C4','C5','D1']:
  group=[r for r in records if r['candidate']==name]
  aligned=[r for r in group if r['structure']]
  summary.append(dict(candidate=name,pages=len(group),proposals=sum(r['proposal'] for r in group),structurally_aligned=len(aligned),components_in_aligned=sum(r['components'] for r in aligned),components_lost_in_aligned=sum(r['components_lost'] for r in aligned),components_with_weak_content_in_aligned=sum(r['components_weak_content'] for r in aligned),aligned_pages_preserving_all_components=sum(r['components_lost']==0 for r in aligned),reference_correct_pages_with_component_loss=sum(r['correct'] and r['components_lost']>0 for r in group)))
 dump('reports/preservation_summary.json',dict(scope='Original frozen 90-page diagnostic, descriptive only; all candidates frozen beforehand',rows=summary,geometry_threshold=.95,minimum_reference_component_pixels=3,content_support='Otsu foreground within 1px of mapped ideal-ink pixels; approximation, not human verification'))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

