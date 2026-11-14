"""MANUAL ONLY: R2 from saved R1 and shared raw observations; no OCR/image pass."""
import argparse
from collections import Counter
from copy import deepcopy
from integration import *
from dataset_loader import DevelopmentDataset,SPLIT
from observation_store import load_raw_run
from evaluate_rejection import load_candidate
from evaluation_metrics import grouped
from r2_structural import structural_decision


def verify_r2():
    result=verify_integrity()
    for name in ('r2_preservation_lock.json','r2_lock.json'):
        for p,h in load_json(ROOT/'configs'/name).items():
            if digest(ROOT/p)!=h:raise ValueError('R2/preserved artifact drift: '+p)
    return result


def apply_saved_row(row):
    # Strip the evaluation row BEFORE runtime: only decision + OCR tier text.
    decision,findings=structural_decision(row['decision'],row['interpreted_transcription'])
    out=deepcopy(row);out.update(actual_outcome=decision['outcome'],decision=decision,
        triggered_reason_codes=decision['reason_codes'],r2_structural_findings=findings)
    return out


def compare_rows(old,new):
    a={r['image_id']:r for r in old};b={r['image_id']:r for r in new}
    if set(a)!=set(b):raise ValueError('Unpaired candidates')
    transitions=Counter((a[k]['validity'],a[k]['actual_outcome'],b[k]['actual_outcome']) for k in a)
    return dict(transitions=[dict(validity=v,before=x,after=y,images=n,
        independent_parents=len({a[k]['source_parent_id'] for k in a if (a[k]['validity'],a[k]['actual_outcome'],b[k]['actual_outcome'])==(v,x,y)})) for (v,x,y),n in sorted(transitions.items())],
        unsafe_acceptance_reduced=sum(a[k]['validity']=='INVALID' and a[k]['actual_outcome']=='ACCEPT' and b[k]['actual_outcome']!='ACCEPT' for k in a),
        new_definitive_invalid_rejections=sum(a[k]['validity']=='INVALID' and a[k]['actual_outcome']!='REJECT_INVALID' and b[k]['actual_outcome']=='REJECT_INVALID' for k in a),
        new_invalid_acceptances=sum(a[k]['validity']=='INVALID' and a[k]['actual_outcome']!='ACCEPT' and b[k]['actual_outcome']=='ACCEPT' for k in a),
        new_valid_reviews=sum(a[k]['validity']=='VALID' and a[k]['actual_outcome']=='ACCEPT' and b[k]['actual_outcome']=='REVIEW_UNCERTAIN' for k in a),
        new_valid_definitive_rejections=sum(a[k]['validity']=='VALID' and a[k]['actual_outcome']!='REJECT_INVALID' and b[k]['actual_outcome']=='REJECT_INVALID' for k in a))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=(SPLIT,),default=SPLIT);p.add_argument('--raw-dir',type=Path,required=True)
    p.add_argument('--r0-dir',type=Path,required=True);p.add_argument('--r1-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    verify_r2();ds=DevelopmentDataset(split=a.split,verify_files=False)
    r0=load_candidate(a.r0_dir,ds);r1=load_candidate(a.r1_dir,ds)
    if r0['lineage']!=r1['lineage']:raise ValueError('R0/R1 raw lineage mismatch')
    raw,lineage=load_raw_run(a.raw_dir,ds)
    if (lineage['completion']['raw_sha256']!=r1['lineage']['raw_observations_sha256'] or
        lineage['completion']['plan_sha256']!=r1['lineage']['raw_plan_sha256']):raise ValueError('Wrong raw run')
    raw_byid={r['identity']['image_id']:r for r in raw}
    for row in r1['rows']:
        if raw_byid[row['image_id']]['observation']['decoded_rows']!=row['raw_OCR_transcription']:raise ValueError('Saved raw transcription mismatch')
    rows=[apply_saved_row(row) for row in r1['rows']]
    summary,parents,groups=grouped(rows);baselines={c:grouped(rr) for c,rr in [('R0',r0['rows']),('R1',r1['rows'])]};out=new_output(a.output_dir)
    write_json(out/'results.json',dict(candidate='R2',lineage=r1['lineage'],r2_lock_sha256=digest(ROOT/'configs/r2_lock.json'),rows=rows,
        scope='development only; count-gap OCR suspicion -> REVIEW; no new definitive evidence'))
    write_json(out/'summary.json',summary);write_json(out/'by_parent.json',parents);write_json(out/'by_group.json',groups)
    write_csv(out/'per_image.csv',rows)
    comparison=dict(R2=summary,R2_parent_results=parents,R2_group_results=groups,
        versus_R0=compare_rows(r0['rows'],rows),versus_R1=compare_rows(r1['rows'],rows),
        note='ACCEPT to REVIEW reduces unsafe acceptance; it is not definitive rejection. No automatic winner.')
    for c,old in [('R0',r0['rows']),('R1',r1['rows'])]:
        comparison[c]=baselines[c][0]
        comparison[c+'_parent_results']=baselines[c][1];comparison[c+'_group_results']=baselines[c][2]
        comparison['versus_'+c]['by_invalidity_category']={cat:compare_rows([r for r in old if r['invalidity_category']==cat],[r for r in rows if r['invalidity_category']==cat]) for cat in ('none','wrong_tier','invalid_count_token','missing_count')}
        comparison['versus_'+c]['metric_deltas_R2_minus_baseline']={key:summary[key]-baselines[c][0][key] for key in
            ('invalid_accepted_rate','true_invalid_rejection_rate','valid_acceptance_rate','false_definitive_rejection_rate','valid_review_rate','invalid_review_rate')}
    write_json(out/'comparison.json',comparison)
    write_csv(out/'comparison.csv',[dict(candidate=c,**metrics) for c,metrics in [('R0',baselines['R0'][0]),('R1',baselines['R1'][0]),('R2',summary)]])
    write_json(out/'completion.json',dict(complete=True,candidate='R2',records=len(rows),results_sha256=digest(out/'results.json')))
    verify_r2();print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
