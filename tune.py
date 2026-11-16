"""Run the predeclared tuning grid, freeze D1 before exposed comparisons."""
from dcore import *
from pipeline_d1 import prepare,process_prepared
from collections import defaultdict
import time
def main():
 design=json.loads((ROOT/'configs/experiment_design.json').read_text());configs=design['grid']
 assert json.loads((ROOT/'reports/diagnosis_complete.json').read_text())['before_D1_implementation']
 assert json.loads((ROOT/'reports/preflight.json').read_text())['all_accounted']
 assert not (ROOT/'configs/D1.json').exists()
 src=Sources();records=[];cases=[p for p in pages() if p['partition']=='tuning'];begin=time.perf_counter()
 for index,p in enumerate(cases):
  im,ref,entries=src.get(p);pixels,M,extra=prepare(im)
  for name,conf in configs.items():
   out=process_prepared(pixels,M,conf,extra);a=reference.assess(out,ref,entries)
   records.append(dict(candidate=name,**{k:p[k] for k in ['partition','dataset','parent_id','case_id','kind']},proposal=out['accepted'],correct=a['correct'],reason=a['reason'],rows=len(out['rows'])))
  if (index+1)%80==0:print('Tuning',index+1,'/',len(cases),'seconds',round(time.perf_counter()-begin),flush=True)
 csvout('reports/tuning_grid.csv',records)
 ranking=[]
 for name in configs:
  scanned=[r for r in records if r['candidate']==name and r['kind']=='scanned']
  clean=[r for r in records if r['candidate']==name and r['kind']=='clean']
  pos=[r for r in scanned if r['dataset']=='positive'];gen=[r for r in scanned if r['dataset']=='general']
  wrong=sum(r['proposal'] and not r['correct'] for r in scanned);correct=sum(r['correct'] for r in scanned)
  # Relative safety gate fixed before experiments: halve historical C5 wrong scans, retain >=95% correct in each dataset.
  base=read(C/'reports/experiments_v1/pages.csv');b=[r for r in base if r['partition']=='tuning' and r['kind']=='scanned' and r['candidate']=='C5']
  bwrong=sum(r['proposal']=='True' and r['correct_segmentation']!='True' for r in b)
  bcorrect={d:sum(r['correct_segmentation']=='True' for r in b if r['dataset']==d) for d in ['positive','general']}
  gate=wrong<=bwrong*.5 and sum(r['correct'] for r in pos)>=.95*bcorrect['positive'] and sum(r['correct'] for r in gen)>=.95*bcorrect['general']
  ranking.append(dict(candidate=name,correct=correct,wrong=wrong,positive_correct=sum(r['correct'] for r in pos),general_correct=sum(r['correct'] for r in gen),clean_correct=sum(r['correct'] for r in clean),safety_gate=gate,wrong_reference=bwrong,correct_reference=bcorrect))
 # If gate fails for all, preserve the safest exploratory candidate, with failure explicit.
 ranking.sort(key=lambda r:(not r['safety_gate'], -r['correct'] if r['safety_gate'] else r['wrong'],r['wrong'] if r['safety_gate'] else -r['correct'],r['candidate']))
 selected=ranking[0]
 dump('configs/D1.json',configs[selected['candidate']])
 lock={str(p):old.digest(p) for p in [ROOT/'engine.py',ROOT/'pipeline_d1.py',ROOT/'dcore.py',ROOT/'configs/experiment_design.json',ROOT/'configs/D1.json',C/'manifests/pages.csv']}
 dump('configs/selection_lock.json',dict(selected_grid=selected['candidate'],candidate='D1',selection_partition='tuning',validation_used_for_selection=False,independent_holdout_claim=False,ranking=ranking,locked_files=lock,elapsed_seconds=time.perf_counter()-begin))
 print('FROZEN D1',json.dumps(selected),flush=True)
if __name__=='__main__':main()

