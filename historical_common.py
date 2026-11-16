"""Continuity with the historical six-candidate common subset; no selection."""
from dcore import *
from summarize_d1 import metric
def main():
 oldpages=read(C/'reports/experiments_v1/pages.csv');newpages=read(ROOT/'reports/evaluation_v1/pages.csv')
 predictions=read(C/'reports/experiments_v1/predictions.csv')+read(ROOT/'reports/evaluation_v1/predictions.csv')
 rows=[]
 for part,ds,kind in [('validation','positive','scanned'),('validation','general','scanned'),('clean_regression','normal','clean')]:
  cohort=lambda r:r['partition']==part and r['dataset']==ds and r['kind']==kind
  sets=[{r['case_id'] for r in oldpages if cohort(r) and r['candidate']==c and r['correct_segmentation']=='True' and int(r['ocr_tiers'])>0} for c in old.CANDIDATES]
  historic=set.intersection(*sets);new={r['case_id'] for r in newpages if cohort(r) and r['correct_segmentation']=='True' and int(r['ocr_tiers'])>0}
  common=historic&new
  for c in ['C0','C4','C5','D1']:
   ps=[r for r in predictions if cohort(r) and r['candidate']==c and r['case_id'] in common]
   rows.append(dict(partition=part,dataset=ds,kind=kind,candidate=c,historical_common_pages=len(historic),retained_common_pages=len(common),**metric(ps)))
 csvout('reports/historical_common_subset.csv',rows)
 saved=[r for r in predictions if r['candidate']=='C4' and r['dataset']=='positive' and r['parent_id']=='edge_00323']
 by={}
 for r in saved:
  by.setdefault(r['case_id'],dict(kind=r['kind'],characters=0,errors=0))
  by[r['case_id']]['characters']+=len(r['truth']);by[r['case_id']]['errors']+=sum(int(r[k]) for k in ['substitutions','deletions','insertions'])
 changes=read(ROOT/'reports/d1_case_changes.csv')
 losses=[r for r in changes if r['C5']=='correct' and r['D1']!='correct']
 dump('reports/regression_analysis.json',dict(C5_correct_D1_not_correct=losses,long_parent_C4_errors_by_case=by,notes='No candidate or labels changed after exposed validation. Lower conditional CER cannot be credited to OCR improvement; D1 abstains on the long perspective page.'))
 print(json.dumps(by),flush=True)
if __name__=='__main__':main()

