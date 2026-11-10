"""Controlled-experiment summaries, common subsets and parent-bootstrap changes."""
import json,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import *


def metrics(predictions):
 if not predictions:return dict(tiers=0,characters=0,cer=None,exact_accuracy=None,substitutions=0,deletions=0,insertions=0)
 chars=sum(len(r['truth']) for r in predictions);sums={k:sum(int(r[k]) for r in predictions) for k in ['substitutions','deletions','insertions']}
 return dict(tiers=len(predictions),characters=chars,cer=sum(sums.values())/chars,exact_accuracy=sum(r['truth']==r['prediction'] for r in predictions)/len(predictions),**sums)


def compile_results(root):
 pages=read_csv(root/'pages.csv');preds=read_csv(root/'predictions.csv');groups=defaultdict(list);predgroups=defaultdict(list)
 for row in pages:groups[(row['partition'],row['dataset'],row['kind'],row['candidate'])].append(row)
 for row in preds:predgroups[(row['partition'],row['dataset'],row['kind'],row['candidate'])].append(row)
 common={}
 for part,dataset,kind,candidate in groups:
  k=(part,dataset,kind)
  if k in common:continue
  sets=[{r['case_id'] for r in groups[k+(c,)] if r['correct_segmentation']=='True' and int(r['ocr_tiers'])>0} for c in CANDIDATES]
  common[k]=set.intersection(*sets)
 summaries=[];categories=[]
 for key,rows in sorted(groups.items()):
  part,dataset,kind,candidate=key;ps=predgroups[key];n=len(rows)
  accepted=sum(r['proposal']=='True' for r in rows);correct=sum(r['correct_segmentation']=='True' for r in rows)
  record=dict(partition=part,dataset=dataset,kind=kind,candidate=candidate,pages=n,parents=len({r['parent_id'] for r in rows}),proposals=accepted,proposal_coverage=accepted/n,
    correct_pages=correct,correct_segmentation_coverage=correct/n,wrong_acceptances=accepted-correct,safe_rejections=n-accepted,accepted_segmentation_accuracy=correct/accepted if accepted else None,
    structural_correct_pages=sum(r['structure_correct']=='True' for r in rows),geometry_applied_pages=sum(r['geometry_applied']=='True' for r in rows),ctc_failures=sum(int(r['ctc_failures']) for r in rows),
    page_exact_accuracy=sum(r['page_exact']=='True' for r in rows)/n,seconds=sum(float(r['seconds']) for r in rows),**metrics(ps))
  cs=metrics([r for r in ps if r['case_id'] in common[key[:3]]]);record.update(common_pages=len(common[key[:3]]),common_tiers=cs['tiers'],common_characters=cs['characters'],common_cer=cs['cer'],common_exact_accuracy=cs['exact_accuracy'])
  summaries.append(record)
  if kind=='scanned':
   for aug in sorted({r['augmentation'] for r in rows}):
    subset=[r for r in rows if r['augmentation']==aug]
    categories.append(dict(partition=part,dataset=dataset,candidate=candidate,augmentation=aug,pages=len(subset),proposals=sum(r['proposal']=='True' for r in subset),correct_pages=sum(r['correct_segmentation']=='True' for r in subset),**metrics([r for r in ps if r['augmentation']==aug])))
 write_csv(root/'summary.csv',summaries);write_json(root/'summary.json',summaries);write_csv(root/'augmentation_metrics.csv',categories)
 selected=json.loads((root/'selection_before_validation.json').read_text())['selected'];bootstrap=[]
 rng=np.random.default_rng(42)
 for dataset in ['positive','general']:
  a=groups[('validation',dataset,'scanned','C0')];b=groups[('validation',dataset,'scanned',selected)]
  parents=sorted({r['parent_id'] for r in a});values=[]
  for parent in parents:
   x=[r['correct_segmentation']=='True' for r in a if r['parent_id']==parent];y=[r['correct_segmentation']=='True' for r in b if r['parent_id']==parent]
   values.append(np.mean(y)-np.mean(x))
  values=np.array(values);indices=rng.integers(0,len(values),size=(2000,len(values)));draws=values[indices].mean(axis=1)
  bootstrap.append(dict(dataset=dataset,partition='validation',metric='correct_segmentation_coverage_change',candidate=selected,baseline='C0',parent_count=len(parents),mean_change=float(values.mean()),lower95=float(np.quantile(draws,.025)),upper95=float(np.quantile(draws,.975)),seed=42,resamples=2000,claim='Development-validation uncertainty, not independent test inference'))
 write_json(root/'parent_bootstrap.json',bootstrap)
 print('SELECTED BEFORE VALIDATION',selected)
 for r in summaries:
  if r['partition']=='validation' and r['kind']=='scanned' or r['partition']=='clean_regression':
   print(r['partition'],r['dataset'],r['candidate'],'proposal',r['proposals'],'correct',r['correct_pages'],'/',r['pages'],'wrong',r['wrong_acceptances'],'CER',r['cer'],'exact',r['exact_accuracy'],'commonCER',r['common_cer'])
 return summaries

if __name__=='__main__':compile_results(HERE/'reports/experiments_v1')
