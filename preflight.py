"""Small tuning-only preflight before the full tuning comparison; no OCR."""
from dcore import *
from pipeline_d1 import prepare,process_prepared
from collections import defaultdict
import time
def main():
 design=json.loads((ROOT/'configs/experiment_design.json').read_text());configs=design['grid']
 tuning=[p for p in pages() if p['partition']=='tuning']
 # Prior failures sampled by parent ranking; all four versions stay together.
 ids=sorted({(p['dataset'],p['parent_id']) for p in tuning},key=lambda k:hashlib.sha256(('stage2d:42:'+str(k)).encode()).hexdigest())
 chosen=set([k for d in ['positive','general'] for k in [k for k in ids if k[0]==d][:8]])
 cases=[p for p in tuning if (p['dataset'],p['parent_id']) in chosen]
 src=Sources();records=[];begin=time.perf_counter()
 for p in cases:
  im,ref,entries=src.get(p);pixels,M,extra=prepare(im)
  for name,conf in configs.items():
   result=process_prepared(pixels,M,conf,extra);a=reference.assess(result,ref,entries)
   records.append(dict(candidate=name,**{k:p[k] for k in ['partition','dataset','parent_id','case_id','kind']},proposal=result['accepted'],correct=a['correct'],reason=a['reason']))
 csvout('reports/preflight.csv',records)
 for name in configs:
  g=[r for r in records if r['candidate']==name]
  print(name,'correct',sum(r['correct'] for r in g),'wrong',sum(r['proposal'] and not r['correct'] for r in g),'/',len(g),flush=True)
 dump('reports/preflight.json',dict(pages=len(cases),parents=len(chosen),partition='tuning_only',no_ocr=True,all_accounted=len(records)==len(cases)*len(configs),elapsed_seconds=time.perf_counter()-begin))
if __name__=='__main__':main()

