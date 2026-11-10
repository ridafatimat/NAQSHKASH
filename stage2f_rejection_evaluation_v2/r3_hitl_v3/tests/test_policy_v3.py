import unittest,sys,copy
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from policy_v3 import *


def base(outcome='ACCEPT'):
    return dict(outcome=outcome,reason_codes=['R2_ORIGINAL'],reason_details=[dict(evidence='retained')],explanation='Original evidence',independent_image_invalidity_confirmed=outcome=='REJECT_INVALID')


STATUS=dict(preprocessing_status='ok',segmentation_accepted=True,ocr_status='ok')
def run(rows,outcome='ACCEPT',status=None,config=None):return decide(base(outcome),rows,status or STATUS,config)


class Tests(unittest.TestCase):
    def test_accept_unique_consistent(self):self.assertEqual(run([dict(upper='D',count='a')])['decision']['outcome'],'ACCEPT')
    def test_valid_lower(self):self.assertEqual(run([dict(lower='R',count='sh')])['decision']['outcome'],'ACCEPT')
    def test_valid_multiple_symbols(self):self.assertEqual(run([dict(upper='D D',count='aaa',lower='R')])['decision']['outcome'],'ACCEPT')
    def test_complete_contradiction_review(self):
        r=run([dict(upper='D',count='hs')]);self.assertTrue(r['new_review']);self.assertIn('R3_COMPLETE_TEXT_CONTRADICTION',r['decision']['reason_codes'])
    def test_incomplete_search_review(self):
        c=copy.deepcopy(load_policy());c['max_offset']=0
        r=run([dict(upper='D',lower='R',count='aa')],config=c)
        self.assertIn('R3_SEARCH_CAP_REACHED',r['decision']['reason_codes']);self.assertEqual(r['decision']['outcome'],'REVIEW_UNCERTAIN')
    def test_unsupported_leading_spaces_review(self):
        r=run([dict(upper=' D',lower='R',count='aa')]);self.assertIn('R3_UNSUPPORTED_REPRESENTATION',r['decision']['reason_codes'])
    def test_multiple_complete_candidates_accept(self):
        r=run([dict(upper='D',lower='R',count='aa')])
        self.assertGreater(r['alignment_evidence']['analysis']['page_candidate_count_in_search'],1)
        self.assertFalse(r['alignment_evidence']['analysis']['unique_page_alignment_proven_in_model'])
        self.assertEqual(r['decision']['outcome'],'ACCEPT');self.assertEqual(r['findings'],[])
    def test_incomplete_witness_still_review(self):
        c=copy.deepcopy(load_policy());c['max_candidates']=2
        r=run([dict(upper='D',lower='R',count='aa')],config=c)
        self.assertGreater(r['alignment_evidence']['analysis']['page_candidate_count_in_search'],0)
        self.assertEqual(r['decision']['outcome'],'REVIEW_UNCERTAIN')
    def test_multiple_candidates_preserve_prior_review_and_reject(self):
        for outcome in ('REVIEW_UNCERTAIN','REJECT_INVALID'):
            self.assertEqual(run([dict(upper='D',lower='R',count='aa')],outcome)['decision'],base(outcome))
    def test_prior_review_consistent_preserved(self):
        before=base('REVIEW_UNCERTAIN');r=decide(before,[dict(upper='D',count='a')],STATUS)
        self.assertEqual(r['decision'],before);self.assertEqual(r['findings'],[])
    def test_prior_review_contradiction_preserved(self):
        before=base('REVIEW_UNCERTAIN');self.assertEqual(decide(before,[dict(lower='D',count='a')],STATUS)['decision'],before)
    def test_prior_reject_with_any_alignment_preserved(self):
        for rows in ([dict(upper='D',count='a')],[dict(lower='D',count='a')],[]):
            self.assertEqual(run(rows,'REJECT_INVALID')['decision'],base('REJECT_INVALID'))
    def test_reason_codes_and_evidence_retained(self):
        r=run([dict(lower='D',count='a')])['decision'];self.assertEqual(r['reason_codes'][0],'R2_ORIGINAL');self.assertEqual(r['reason_details'][0],dict(evidence='retained'))
    def test_missing_first_count_no_exact_missing_claim(self):
        r=run([dict(upper='D',lower='R',count='sh')]);self.assertEqual(r['decision']['outcome'],'ACCEPT')
        self.assertFalse(r['image_invalidity_newly_confirmed'])
        self.assertTrue(all(f['associated_symbol'] is None for f in r['findings']))
    def test_missing_middle_count(self):self.assertEqual(run([dict(upper='D F',lower='R',count='a a')])['decision']['outcome'],'REVIEW_UNCERTAIN')
    def test_missing_last_count(self):self.assertEqual(run([dict(upper='D',lower='R',count='sh ')])['decision']['outcome'],'REVIEW_UNCERTAIN')
    def test_wrong_tier_suspicion_is_review_only(self):self.assertEqual(run([dict(lower='D',count='a')])['decision']['outcome'],'REVIEW_UNCERTAIN')
    def test_wrong_tier_OCR_substitution_cannot_be_recovered(self):
        r=run([dict(lower='_',count='a')]);self.assertEqual(r['decision']['outcome'],'ACCEPT');self.assertFalse(r['image_invalidity_newly_confirmed'])
    def test_conflicting_row_totals(self):
        r=run([dict(upper='D',count='a'),dict(lower='R',count='b')]);self.assertIn('R3_ROW_TOTAL_CONFLICT',r['decision']['reason_codes'])
    def test_lost_internal_spaces(self):self.assertEqual(run([dict(upper='DD',lower='R',count='aaa')])['decision']['outcome'],'REVIEW_UNCERTAIN')
    def test_OCR_unavailable(self):
        r=run([],status=dict(preprocessing_status='ok',segmentation_accepted=False,ocr_status='not_run'))
        self.assertIn('R3_OCR_OR_SEGMENTATION_UNAVAILABLE',r['decision']['reason_codes'])
    def test_no_new_reject_property(self):
        fixtures=[[dict(upper='D',count='a')],[dict(upper='D',lower='R',count='aa')],[dict(lower='D',count='a')],[]]
        for before in OUTCOMES:
            for rows in fixtures:
                after=run(rows,before)['decision']['outcome'];assert_invariants(before,after)
                self.assertEqual(after=='REJECT_INVALID',before=='REJECT_INVALID')
    def test_evidence_bound_to_runtime_inputs(self):
        c=load_policy();rows=[dict(upper='D',count='a')];ev=evidence(rows,STATUS,c)
        with self.assertRaises(ValueError):decide(base(),[dict(lower='D',count='a')],STATUS,c,ev)
    def test_configuration_has_no_confidence_threshold(self):
        c=copy.deepcopy(load_policy());c['threshold']=-0.0002
        with self.assertRaises(ValueError):validate_policy(c)
    def test_inputs_are_not_modified(self):
        b=base();rows=[dict(lower='D',count='a')];before=copy.deepcopy((b,rows,STATUS));decide(b,rows,STATUS)
        self.assertEqual((b,rows,STATUS),before)
    def test_no_label_or_mutation_parameter_in_runtime_API(self):
        with self.assertRaises(TypeError):decide(base(),[],STATUS,validity='INVALID')
        with self.assertRaises(TypeError):decide(base(),[],STATUS,mutation_parameters={})
    def test_canonical_uncertainty_is_review(self):
        from integration import rules
        r=run([dict(upper=sorted(rules.AMBIGUOUS_SYMBOLS)[0],count='a')]);self.assertEqual(r['decision']['outcome'],'REVIEW_UNCERTAIN')
    def test_model_confidence_is_not_a_decision_input(self):
        rows=[dict(upper='D',count='hs')]
        a=decide(base(),rows,dict(STATUS,model_confidence=-1000));b=decide(base(),rows,dict(STATUS,model_confidence=1000))
        self.assertEqual(a['decision'],b['decision'])


if __name__=='__main__':unittest.main()

