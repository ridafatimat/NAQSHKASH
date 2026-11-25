"""Final paired comparison, predeclared model selection and error evidence."""
import sys,json,math,shutil,html,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from context import *
from metrics import aggregate
from selection import eligible,rank
from collections import defaultdict,Counter
def grouped(rows,keys):
 groups=defaultdict(list)
 for r in rows:groups[tuple(r[k] for k in keys)].append(r)
 return [dict(zip(keys,k),**aggregate(g)) for k,g in sorted(groups.items())]
def main():
 summaries={name:json.loads((E/'metrics'/f'{name}_summary.json').read_text()) for name in ['baseline','e1','e2']}
 samples={name:read(E/'metrics'/f'{name}_per_sample.csv') for name in summaries}
 ids=[r['id'] for r in samples['baseline']]
 assert all([r['id'] for r in ps]==ids for ps in samples.values())
 names={'baseline':'E0','e1':'E1','e2':'E2'}
 for name in ['e1','e2']:
  result=json.loads((E/'reports'/f'{name}_completion.json').read_text());assert result['original_unchanged'] and result['steps']>0
 candidates=[name for name in ['e1','e2'] if eligible(summaries[name],summaries['baseline'])]
 selected=min(candidates,key=lambda n:rank(summaries[n])) if candidates else 'baseline'
 source=CHECKPOINT if selected=='baseline' else Path(json.loads((E/'reports'/f'{selected}_completion.json').read_text())['candidate_checkpoint'])
 folder=E/'checkpoints/selected_model'
 if folder.exists():raise RuntimeError('Preserve frozen selection; do not overwrite')
 folder.mkdir();target=folder/'best.pt';shutil.copy2(source,target);assert digest(source)==digest(target)
 m,v,state=model_load(target);assert all(torch.isfinite(t).all() for t in m.state_dict().values())
 selection=dict(selected=names[selected],source_checkpoint=str(source),selected_checkpoint=str(target),sha256=digest(target),original_sha256=EXPECTED,selected_is_fine_tuned=selected!='baseline',selection_rule=json.loads((E/'configs/training_design.json').read_text())['config']['selection'],eligibility={names[n]:eligible(summaries[n],summaries['baseline']) for n in ['e1','e2']},full_development_used=True,independent_test=False,untouched_test_evaluations=0)
 dump('reports/selected_checkpoint.json',selection);dump('checkpoints/selected_model/lineage.json',selection)
 comparisons=[]
 for name,summary in summaries.items():
  for cohort,value in summary.items():
   if cohort=='runtime':continue
   comparisons.append(dict(candidate=names[name],cohort=cohort,**{k:value[k] for k in ['tiers','parents','characters','cer','exact_accuracy','macro_parent_cer','space_normalized_diagnostic_cer','substitutions','deletions','insertions','ctc_loss']}))
 write('metrics/experiment_comparison.csv',comparisons)
 allrows=[dict(r,candidate=names[n]) for n,rows in samples.items() for r in rows]
 write('metrics/by_length.csv',grouped(allrows,['candidate','cohort','length_bucket']))
 write('metrics/by_distortion.csv',grouped(allrows,['candidate','cohort','augmentation']))
 write('metrics/by_role.csv',grouped(allrows,['candidate','cohort','role']))
 write('metrics/by_parent.csv',grouped(allrows,['candidate','cohort','parent_id']))
 # Page end-to-end accounting includes all D1 rejects/wrong proposals.
 ledger=read(E/'manifests/page_outcomes.csv');ends=[]
 for name,rows in samples.items():
  bycase=defaultdict(list)
  for r in rows:bycase[(r['cohort'],r['case_id'])].append(r)
  for cohort in sorted({p['cohort'] for p in ledger}):
   pages=[p for p in ledger if p['cohort']==cohort];correct=[p for p in pages if p['correct']=='True']
   exact=sum(bool(bycase[(cohort,p['case_id'])]) and all(r['exact']=='True' for r in bycase[(cohort,p['case_id'])]) for p in correct)
   ends.append(dict(candidate=names[name],cohort=cohort,pages=len(pages),parents=len({p['parent_id'] for p in pages}),correct_segmentation=len(correct),wrong_segmentation=sum(p['proposal']=='True' and p['correct']!='True' for p in pages),segmentation_rejected=sum(p['proposal']!='True' for p in pages),end_to_end_exact_pages=exact,end_to_end_exact_accuracy=exact/len(pages)))
 write('metrics/end_to_end_pages.csv',ends)
 # Long spacing analysis retains exact original labels.
 long=[]
 for name,rows in samples.items():
  for r in rows:
