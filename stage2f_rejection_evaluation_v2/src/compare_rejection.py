"""Paired R0/R1 comparison; never selects/tunes a policy automatically."""
import argparse
from collections import Counter
from integration import *
from dataset_loader import DevelopmentDataset,SPLIT
from evaluate_rejection import load_candidate
from evaluation_metrics import grouped


def compare(r0,r1):
    if r0['candidate']!='R0' or r1['candidate']!='R1':raise ValueError('Expected R0 and R1')
    if r0['lineage']!=r1['lineage']:raise ValueError('R0/R1 must reuse the EXACT same raw inference and frozen inputs')
    a={r['image_id']:r for r in r0['rows']};b={r['image_id']:r for r in r1['rows']}
    if set(a)!=set(b) or len(a)!=len(r0['rows']) or len(b)!=len(r1['rows']):raise ValueError('Candidate record sets differ/duplicate')
    for key in a:
        for field in ('source_parent_id','split','image_sha256','validity','raw_OCR_transcription','valid_reference_transcription'):
            if a[key][field]!=b[key][field]:raise ValueError('Unpaired source/raw reference: '+field)
    m0,p0,g0=grouped(r0['rows']);m1,p1,g1=grouped(r1['rows'])
    keys=('true_invalid_rejection_rate','invalid_accepted_rate','valid_acceptance_rate','false_definitive_rejection_rate',
          'invalid_review_rate','valid_review_rate','review_rate','exact_transcription_accuracy','valid_interpreted_CER','accepted_output_transcription_correctness')
    deltas={k:m1[k]-m0[k] if m0[k] is not None and m1[k] is not None else None for k in keys}
    transitions=Counter((a[k]['validity'],a[k]['actual_outcome'],b[k]['actual_outcome']) for k in a)
    critical=dict(invalid_false_acceptance_increased=(deltas['invalid_accepted_rate'] or 0)>0,
                  valid_false_rejection_increased=(deltas['false_definitive_rejection_rate'] or 0)>0)
    return dict(R0=m0,R1=m1,deltas_R1_minus_R0=deltas,critical_regressions=critical,
        outcome_transitions=[dict(validity=k[0],R0=k[1],R1=k[2],images=n) for k,n in sorted(transitions.items())],
        parent_results=dict(R0=p0,R1=p1),group_results=dict(R0=g0,R1=g1),
        selection_decision='NOT_AUTOMATIC: review critical errors, real transcription correctness and unscorable interventions together',
        evidence_scope='development synthetic canonical labels; no Talim expert review; no protected-test inference')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path);p.add_argument('--split',choices=(SPLIT,),default=SPLIT)
    p.add_argument('--r0-dir',type=Path,required=True);p.add_argument('--r1-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();verify_integrity();dataset=DevelopmentDataset(a.dataset,a.split)
    result=compare(load_candidate(a.r0_dir,dataset),load_candidate(a.r1_dir,dataset));out=new_output(a.output_dir)
    write_json(out/'comparison.json',result)
    write_csv(out/'comparison.csv',[dict(candidate=k,**result[k]) for k in ('R0','R1')])
    write_csv(out/'parent_comparison.csv',[dict(candidate=k,source_parent_id=pid,**values) for k in ('R0','R1') for pid,values in result['parent_results'][k].items()])
    text='# R0 versus R1 — actual saved development results\n\nNo automatic winner. REVIEW is not definitive rejection. Inspect invalid false acceptance, valid false rejection, exact/CER, accepted correctness, parent worst cases and intervention harm together.\n\n'
    text+='```json\n'+json.dumps({k:v for k,v in result.items() if k not in ('parent_results','group_results')},indent=2)+'\n```\n'
    (out/'comparison.md').write_text(text,encoding='utf-8')
    write_json(out/'completion.json',dict(complete=True,comparison_sha256=digest(out/'comparison.json')))
    print(out)


if __name__=='__main__':main()
