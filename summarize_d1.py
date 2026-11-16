"""C4/C5/D1 comparisons with explicit cohorts, common subsets and parent accounting."""
from dcore import *
from collections import defaultdict
def metric(ps):
 chars=sum(len(r['truth']) for r in ps);err=sum(sum(int(r[k]) for k in ['substitutions','deletions','insertions']) for r in ps)
 return dict(tiers=len(ps),characters=chars,errors=err,cer=err/chars if chars else None,exact_tier_accuracy=sum(r['truth']==r['prediction'] for r in ps)/len(ps) if ps else None)
def main():
 records=[r for r in read(C/'reports/experiments_v1/pages.csv') if r['candidate'] in ['C4','C5']]+read(ROOT/'reports/evaluation_v1/pages.csv')
 preds=[r for r in read(C/'reports/experiments_v1/predictions.csv') if r['candidate'] in ['C4','C5']]+read(ROOT/'reports/evaluation_v1/predictions.csv')
 groups=defaultdict(list);psgroups=defaultdict(list)
 for r in records:groups[(r['partition'],r['dataset'],r['kind'],r['candidate'])].append(r)
 for r in preds:psgroups[(r['partition'],r['dataset'],r['kind'],r['candidate'])].append(r)
 traces={(r['candidate'],*key(r)):r for r in json.loads((ROOT/'reports/role_traces.json').read_text())}
 summary=[];parentrows=[]
 for cohort in sorted({k[:3] for k in groups}):
  common=set.intersection(*[{r['case_id'] for r in groups[cohort+(c,)] if r['correct_segmentation']=='True' and int(r['ocr_tiers'])>0} for c in ['C4','C5','D1']])
  for candidate in ['C4','C5','D1']:
   k=cohort+(candidate,);rs=groups[k];ps=psgroups[k];n=len(rs);correct=sum(r['correct_segmentation']=='True' for r in rs);accepted=sum(r['proposal']=='True' for r in rs)
   row=dict(partition=k[0],dataset=k[1],kind=k[2],candidate=candidate,pages=n,parents=len({r['parent_id'] for r in rs}),proposals=accepted,proposal_coverage=accepted/n,correct_pages=correct,correct_coverage=correct/n,wrong_acceptances=accepted-correct,valid_rejections=n-accepted,accepted_segmentation_accuracy=correct/accepted if accepted else None,structurally_correct=sum(r['structure_correct']=='True' for r in rs),accepted_row_count_correct=sum(r['proposal']=='True' and r['reason']!='row_count_mismatch' for r in rs),page_exact_all=sum(r['page_exact']=='True' for r in rs)/n,ctc_failures=sum(int(r['ctc_failures']) for r in rs),**metric(ps))
   def row_match(r):
    if candidate=='D1':return int(r['inferred_row_count'])==int(r['expected_row_count'])
    t=traces.get((candidate,*key(r)))
    return bool(t) and len(t['proposed_rows'])==len(t['expected_roles'])
   row['hypothesis_row_count_correct_all']=sum(row_match(r) for r in rs)
   row['hypothesis_row_count_accuracy_all']=row['hypothesis_row_count_correct_all']/n
   row['row_count_accuracy_all']=row['accepted_row_count_correct']/n
   row['row_count_accuracy_accepted']=row['accepted_row_count_correct']/accepted if accepted else None
   commonmetrics=metric([p for p in ps if p['case_id'] in common])
   row.update(common_pages=len(common),common_cer=commonmetrics['cer'],common_exact_tier_accuracy=commonmetrics['exact_tier_accuracy'],common_characters=commonmetrics['characters'])
   for category,condition in [('parents_any_correct',lambda r:r['correct_segmentation']=='True'),('parents_any_wrong',lambda r:r['proposal']=='True' and r['correct_segmentation']!='True'),('parents_any_rejection',lambda r:r['proposal']!='True')]:
    row[category]=len({r['parent_id'] for r in rs if condition(r)})
   for parent in sorted({r['parent_id'] for r in rs}):
    g=[r for r in rs if r['parent_id']==parent];pp=[p for p in ps if p['parent_id']==parent]
    parentrows.append(dict(partition=k[0],dataset=k[1],kind=k[2],candidate=candidate,parent_id=parent,pages=len(g),correct=sum(r['correct_segmentation']=='True' for r in g),wrong=sum(r['proposal']=='True' and r['correct_segmentation']!='True' for r in g),rejected=sum(r['proposal']!='True' for r in g),**metric(pp)))
   summary.append(row)
 csvout('reports/candidate_summary.csv',summary);dump('reports/candidate_summary.json',summary);csvout('reports/parent_results.csv',parentrows)
 # Case-level gains/losses and their distinct parents; validation never reselects D1.
 by={(r['candidate'],*key(r)):r for r in records};changes=[]
 for p in pages():
  a=by[('C5',*key(p))];b=by[('D1',*key(p))]
  status=lambda r:'correct' if r['correct_segmentation']=='True' else 'wrong' if r['proposal']=='True' else 'rejected'
  changes.append(dict(**{k:p[k] for k in ['partition','dataset','parent_id','case_id','kind']},C5=status(a),D1=status(b),D1_reason=b['reason']))
 csvout('reports/d1_case_changes.csv',changes)
 countrows=[]
 for part,ds,kind in sorted({(r['partition'],r['dataset'],r['kind']) for r in changes}):
  group=[r for r in changes if (r['partition'],r['dataset'],r['kind'])==(part,ds,kind)]
  for a,b in sorted({(r['C5'],r['D1']) for r in group}):
   g=[r for r in group if (r['C5'],r['D1'])==(a,b)]
   countrows.append(dict(partition=part,dataset=ds,kind=kind,C5=a,D1=b,images=len(g),parents=len({r['parent_id'] for r in g})))
 csvout('reports/d1_change_counts.csv',countrows)
 lines=['# Stage 2D candidate comparison','', 'D1 is frozen from original tuning parents only. Validation below was already exposed in Stage 2C and is a development regression comparison, not an independent test. C4 and C5 are unchanged; their segmentation was replayed for all pages and OCR was recomputed from saved predictions. D1 uses the original frozen model and decoder. Exact-pixel-hash OCR reuse cannot select a segmentation.','', '| Scope | Candidate | Correct/pages | Wrong | Rejected valid | Correct coverage | Conditional CER | Exact tiers | Common pages | Common CER |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 pct=lambda v:'n/a' if v is None else f"{100*v:.3f}%"
 for r in summary:
  if r['partition']=='validation' and r['kind']=='scanned' or r['partition']=='clean_regression':
   lines.append(f"| {r['dataset']} {r['partition']} | {r['candidate']} | {r['correct_pages']}/{r['pages']} | {r['wrong_acceptances']} | {r['valid_rejections']} | {pct(r['correct_coverage'])} | {pct(r['cer'])} | {pct(r['exact_tier_accuracy'])} | {r['common_pages']} | {pct(r['common_cer'])} |")
 selection=json.loads((ROOT/'configs/selection_lock.json').read_text())
 lines+=['','## Frozen selection and reliability',json.dumps(selection['ranking'],indent=2),'','The safety gate requires at least 95% of C5 tuning correct scanned pages separately for positive/general, while halving C5 tuning scanned wrong acceptances. No OCR score or validation outcome chooses the candidate. Historical engineering targets are not tuning permissions. Six configurations and one documented training-parent gap-rule correction were tested; first-attempt outputs and its focused failure remain in reports/attempt1.','', 'All variants are correlated by source parent. See parent_results.csv, d1_change_counts.csv and candidate_summary.csv for per-parent coverage, wrong acceptance and rejection counts, including overlapping parent sets. Common-subset OCR uses only complete correctly segmented, CTC-feasible pages shared by C4/C5/D1 within each cohort. It differs from the historical six-candidate intersection, so its denominator is reported explicitly. Conditional OCR excludes wrong/rejected pages; page_exact_all includes them as unsuccessful.','', 'Accepted row-count accuracy and exact ordered tier-role structure are separate machine-readable columns. Rejected cases are never credited as successful segmentations. hypothesis_row_count_accuracy_all separately evaluates the raw best row-count hypothesis even if confidence later abstains; missing hypotheses count as wrong. Component preservation is measured independently on ideal source components >=3px; per-page geometric loss and approximate processed-ink support are in evaluation_v1/pages.csv. This is not proof that all real scanned dot shapes are visually correct.','', '## Interpretation', 'D1 targets grouping and role confidence; photometric normalization, denoising, geometry, model and decoder are inherited unchanged. Remaining wrong acceptances must stay visible and prevent automatic deployment approval. Positive conditional CER can remain dominated by the long exposed parent; no target spaces are removed and no failure is omitted. Do not interpret synthetic scanned-style development results as handwriting generalization.','', 'See failure_analysis.md for root-cause evidence, fine_tuning_readiness.md for verified training eligibility, and galleries/index.html for original/processed/full-resolution examples.']
 (ROOT/'reports/candidate_comparison.md').write_text('\n'.join(lines),encoding='utf-8')
 for r in summary:
  if r['partition']=='validation' and r['kind']=='scanned' or r['partition']=='clean_regression':print(r['dataset'],r['candidate'],r['correct_pages'],r['wrong_acceptances'],r['valid_rejections'],'CER',r['cer'],'common',r['common_cer'],flush=True)
if __name__=='__main__':main()

