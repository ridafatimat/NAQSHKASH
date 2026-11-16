"""Pixel-only trace replay; annotations are used afterwards by offline assessor."""
from dcore import *
from diagnose import exhaustive,dp_reference
from collections import Counter
from types import ModuleType
import time
def instrument():
 source=(C/'segmentation/engine.py').read_text()
 source=source.replace("component_scale=scale)","component_scale=scale,bands=bands,features=features,path=path,solutions=solutions)")
 module=ModuleType('diagnostic_engine');exec(compile(source,str(C/'segmentation/engine.py'),'exec'),module.__dict__)
 return module
def main():
 sources=Sources();engine=instrument();saved={(r['candidate'],*key(r)):r for r in read(C/'reports/experiments_v1/pages.csv')}
 records=[];traces=[];begin=time.perf_counter()
 for i,p in enumerate(pages()):
  image,ref,entries=sources.get(p);processor=Processor(image)
  for name in ['C4','C5']:
   out=processor.run(old.CANDIDATES[name]);a=reference.assess(out,ref,entries);prior=saved[(name,*key(p))]
   assert out['accepted']==(prior['proposal']=='True') and a['correct']==(prior['correct_segmentation']=='True'),(p,name)
   record={k:p[k] for k in ['dataset','partition','parent_id','case_id','kind']}
   record.update(candidate=name,accepted=out['accepted'],correct=a['correct'],reason=a['reason'])
   # Recover trace even if strict mode rejected (no alternate chosen using annotations).
   try:
    if not out['accepted']:
     loose=dict(old.CANDIDATES[name]);loose['conservative']=False
     # For strict-rejected C5 preserve its original denoising transformation through cache.
     photo=old.load_local('d_trace_photo','preprocessing/photometric.py')
     from pipeline import geo
     from preprocessing.pipeline import crop_and_separate_rows
     pixels=photo.flatten(image);pixels,applied=photo.conservative_denoise(pixels)
     pixels,M,extra=geo.correct(pixels);pixels=crop_and_separate_rows(pixels,apply_deskew=False,apply_cleaning=not applied).deskewed_image
    else:pixels=out['image']
    rows,diag=engine.propose(pixels,False)
    expected=exhaustive(diag['bands'],diag['features'],diag['component_scale']) if len(diag['bands'])<=10 else dp_reference(diag['bands'],diag['features'],diag['component_scale'])
    assert all(abs(x[0]-y[0])<1e-9 for x,y in zip(diag['solutions'],expected))
    traces.append(dict(**record,diagnostics=diag,proposed_rows=rows,expected_roles=[[e['role'] for e in entries if e['row']==ri] for ri in range(1+max(e['row'] for e in entries))],shape=list(pixels.shape)))
    record.update(margin=diag['role_margin'],bands=diag['band_count'],scale=diag['component_scale'],expected_rows=1+max(e['row'] for e in entries),detected_rows=len(rows))
   except ValueError as exc:record.update(trace_error=str(exc))
   records.append(record)
  if (i+1)%40==0:print('Replay',i+1,'/',len(pages()),'elapsed',round(time.perf_counter()-begin),flush=True)
 csvout('reports/reproduced_pages.csv',records,sorted(set().union(*(r.keys() for r in records))))
 dump('reports/role_traces.json',traces)
 dump('reports/reproduction.json',dict(pages=len(pages()),candidate_outcomes=len(records),saved_acceptance_and_correctness_reproduced=True,elapsed_seconds=time.perf_counter()-begin,small_actual_sequences_exhaustively_checked=sum(t['diagnostics']['band_count']<=10 for t in traces)))
if __name__=='__main__':main()

