import unittest,json,sys
from pathlib import Path
from copy import deepcopy
HERE=Path(__file__).resolve().parents[1]
R=Path(r'C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2')
sys.path.insert(0,str(R/'src'));sys.path.insert(0,str(HERE/'src'))
from integration import rules,apply_candidate
from r2_structural import structural_decision,count_gap_findings
from evaluate_r2 import apply_saved_row,compare_rows

def decision(outcome='ACCEPT'):
    return dict(outcome=outcome,reason_codes=['BASE'],independent_image_invalidity_confirmed=False)

class Tests(unittest.TestCase):
    def test_canonical_count_grammar_has_no_internal_blanks(self):
        for n in range(1,260):self.assertNotIn(' ',rules.encode_count(n))
        for symbol in rules.TRUSTED_SYMBOLS:
            if symbol in rules.AMBIGUOUS_SYMBOLS:continue
            self.assertEqual(len(symbol),1)
    def test_three_way_preservation(self):
        for outcome in ('ACCEPT','REVIEW_UNCERTAIN','REJECT_INVALID'):
            d,f=structural_decision(decision(outcome),[dict(count='a a')])
            self.assertEqual(d['outcome'],'REVIEW_UNCERTAIN' if outcome=='ACCEPT' else outcome)
    def test_internal_gap_is_review_not_image_proof(self):
        d,f=structural_decision(decision(),[dict(count='jajifka aa')])
        self.assertEqual(d['outcome'],'REVIEW_UNCERTAIN');self.assertFalse(d['independent_image_invalidity_confirmed'])
        self.assertEqual(f[0]['evidence_strength'],'OCR_only_uncertainty')
    def test_edge_blanks_do_not_infer_missing_count(self):
        for text in ('  a','a  ','a','',None):
            self.assertEqual(structural_decision(decision(),[dict(count=text)])[0]['outcome'],'ACCEPT')
    def test_count_order_without_run_boundaries_is_unresolved(self):
        # sh versus hs needs observed symbol boundaries; do not invent them.
        for token in ('sh','hs'):
            self.assertEqual(structural_decision(decision(),[dict(count=token)])[0]['outcome'],'ACCEPT')
    def test_missing_count_collapsed_by_ocr_is_unresolved(self):
        self.assertEqual(structural_decision(decision(),[dict(count='jajifkaaa')])[0]['outcome'],'ACCEPT')
    def test_wrong_tier_suspicion_stays_review(self):
        d,f=structural_decision(decision('REVIEW_UNCERTAIN'),[dict(lower='D',count='a')])
        self.assertEqual(d['outcome'],'REVIEW_UNCERTAIN');self.assertEqual(f,[])
    def test_B_X_ambiguity_is_never_globally_replaced(self):
        rows=[dict(upper='B',count='a',lower='X')];original=deepcopy(rows)
        structural_decision(decision(),rows);self.assertEqual(rows,original)
    def test_inputs_are_not_modified(self):
        d=decision();rows=[dict(count='a a')];before=deepcopy((d,rows));structural_decision(d,rows)
        self.assertEqual((d,rows),before)
    def test_no_label_leakage(self):
        row=dict(decision=decision(),interpreted_transcription=[dict(count='a a')],actual_outcome='ACCEPT')
        a=apply_saved_row(dict(row,validity='VALID',mutation_parameters={'hidden':'a'},matched_valid_control_id='x'))
        b=apply_saved_row(dict(row,validity='INVALID',mutation_parameters={'hidden':'b'},matched_valid_control_id='y'))
        self.assertEqual(a['decision'],b['decision'])
    def test_original_R0_R1_are_unchanged(self):
        obs=dict(preprocessing_status='ok',segmentation_accepted=True,ocr_status='ok',decoded_rows=[dict(upper='D',count='a a')],segmentation_rows=[],tiers=[],diagnostics={})
        before=[apply_candidate(obs,{},c) for c in ('R0','R1')]
        structural_decision(before[1]['final_decision'],obs['decoded_rows'])
        self.assertEqual(before,[apply_candidate(obs,{},c) for c in ('R0','R1')])
    def test_saved_representative_pairs_only(self):
        rows=json.loads((R/'metrics/20261106_204802_280_R1/results.json').read_text())['rows'];byid={x['image_id']:x for x in rows}
        selected=[]
        for category in ('wrong_tier','invalid_count_token','missing_count'):
            selected.append(next(x for x in rows if x['invalidity_category']==category and x['actual_outcome']=='ACCEPT' and
                (category!='missing_count' or any(' ' in z['count'].strip() for z in x['interpreted_transcription']))))
        for invalid in selected:
            valid=byid[invalid['matched_valid_control_id']]
            self.assertNotEqual(apply_saved_row(valid)['actual_outcome'],'REJECT_INVALID')
            self.assertNotEqual(apply_saved_row(invalid)['actual_outcome'],'REJECT_INVALID')
        self.assertEqual(apply_saved_row(selected[-1])['actual_outcome'],'REVIEW_UNCERTAIN')
    def test_comparison_distinguishes_review_from_rejection(self):
        a=[dict(image_id='x',source_parent_id='p',validity='INVALID',actual_outcome='ACCEPT')];b=deepcopy(a);b[0]['actual_outcome']='REVIEW_UNCERTAIN'
        result=compare_rows(a,b);self.assertEqual(result['unsafe_acceptance_reduced'],1);self.assertEqual(result['new_definitive_invalid_rejections'],0)

if __name__=='__main__':unittest.main()
