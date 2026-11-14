import unittest,sys,copy,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from policy_v3_1 import *
from paths_v3_1 import evidence
from policy_v3 import load_policy
STATUS=dict(preprocessing_status='ok',segmentation_accepted=True,ocr_status='ok')
MULTIPLE=[dict(upper='D',lower='R',count='aa')]
UNIQUE=[dict(upper='D',count='a')]


def base(outcome='ACCEPT'):return dict(outcome=outcome,reason_codes=['R3_ORIGINAL'],reason_details=[dict(retained=True)],explanation='Original explanation')
def analysis(text):return evidence(text,STATUS,load_policy())['analysis']
def measured(text,values):
    return dict(status='complete',geometry_verified=True,box_equality=True,coordinate_system='D1_processed_pixel_centers',decoded_rows_sha256=content_hash(text),
                rows=[dict(logical_row_index=i,tiers={role:dict(measured_offset=v,offset_interval=[v,v]) for role,v in row.items()}) for i,row in enumerate(values)])
def run(text=MULTIPLE,positions=None,outcome='ACCEPT'):
    return decide(base(outcome),text,analysis(text),positions)


class Tests(unittest.TestCase):
    def test_two_distinct_readings_before(self):self.assertEqual(reading_space(catalogue(MULTIPLE,analysis(MULTIPLE)))['distinct_readings'],2)
    def test_one_reading_selected_by_ink(self):
        r=run(positions=measured(MULTIPLE,[dict(upper=0,count=0,lower=1)]))
        self.assertEqual(r['decision']['outcome'],'ACCEPT');self.assertTrue(r['carpet_generation_ready']);self.assertTrue(r['ambiguity_resolved'])
        self.assertEqual(r['selected_reading']['logical_rows'][0]['runs'],[dict(symbol='D',count=1),dict(symbol='R',count=1)])
    def test_opposite_reading_selected(self):
        r=run(positions=measured(MULTIPLE,[dict(upper=1,count=0,lower=0)]))
        self.assertEqual(r['selected_reading']['logical_rows'][0]['runs'][0]['symbol'],'R')
    def test_no_candidate_matches(self):
        r=run(positions=measured(MULTIPLE,[dict(upper=0,count=0,lower=3)]))
        self.assertEqual(r['decision']['outcome'],'REVIEW_UNCERTAIN');self.assertEqual(r['findings'][0]['reason_code'],'R3_1_MEASURED_POSITION_MISMATCH')
    def test_half_column_inclusive(self):
        r=run(UNIQUE,measured(UNIQUE,[dict(upper=0.5,count=0)]));self.assertTrue(r['carpet_generation_ready'])
    def test_just_beyond_half_column(self):
        r=run(UNIQUE,measured(UNIQUE,[dict(upper=0.500001,count=0)]));self.assertEqual(r['decision']['outcome'],'REVIEW_UNCERTAIN')
    def test_multiple_after_measurement_review(self):
        r=run(positions=measured(MULTIPLE,[dict(upper=0.5,count=0,lower=0.5)]))
        self.assertEqual(r['findings'][0]['reason_code'],'R3_1_AMBIGUOUS_AFTER_MEASUREMENT');self.assertFalse(r['carpet_generation_ready'])
    def test_missing_position_multiple_review(self):self.assertEqual(run()['findings'][0]['reason_code'],'R3_1_POSITION_UNAVAILABLE')
    def test_missing_position_unique_accept_not_ready(self):
        r=run(UNIQUE);self.assertEqual(r['decision'],base());self.assertIsNotNone(r['selected_reading']);self.assertFalse(r['carpet_generation_ready'])
    def test_unverified_position_is_unavailable(self):
        m=measured(MULTIPLE,[dict(upper=0,count=0,lower=1)]);m['geometry_verified']=False
        self.assertEqual(run(positions=m)['findings'][0]['reason_code'],'R3_1_POSITION_UNAVAILABLE')
    def test_geometry_mismatch_cannot_enable_generation(self):
        m=measured(UNIQUE,[dict(upper=0,count=0)]);m['box_equality']=False
        self.assertFalse(run(UNIQUE,m)['carpet_generation_ready'])
    def test_uncertainty_straddling_boundary_is_unavailable(self):
        m=measured(UNIQUE,[dict(upper=0.5,count=0)]);m['rows'][0]['tiers']['upper']['offset_interval']=[0.4,0.6]
        r=run(UNIQUE,m);self.assertEqual(r['decision']['outcome'],'ACCEPT');self.assertFalse(r['carpet_generation_ready'])
    def test_wrong_text_digest_is_unavailable(self):
        m=measured(MULTIPLE,[dict(upper=0,count=0,lower=1)]);m['decoded_rows_sha256']='other'
        self.assertEqual(run(positions=m)['decision']['outcome'],'REVIEW_UNCERTAIN')
    def test_review_preserved_without_candidates(self):self.assertEqual(decide(base('REVIEW_UNCERTAIN'),[],{},None)['decision'],base('REVIEW_UNCERTAIN'))
    def test_reject_preserved_without_candidates(self):self.assertEqual(decide(base('REJECT_INVALID'),[],{},None)['decision'],base('REJECT_INVALID'))
    def test_no_new_reject(self):
        for text in (UNIQUE,MULTIPLE,[]):
            for before in ('ACCEPT','REVIEW_UNCERTAIN','REJECT_INVALID'):
                r=run(text,outcome=before);self.assertEqual(r['decision']['outcome']=='REJECT_INVALID',before=='REJECT_INVALID')
    def test_previous_explanation_and_details_retained(self):
        r=run();self.assertEqual(r['decision']['explanation'],base()['explanation']);self.assertEqual(r['decision']['reason_details'][0],base()['reason_details'][0])
    def test_source_metadata_not_runtime_parameter(self):
        with self.assertRaises(TypeError):decide(base(),MULTIPLE,analysis(MULTIPLE),validity='VALID')
    def test_unique_reading_json_round_trip(self):
        r=run(UNIQUE);self.assertEqual(json.loads(json.dumps(r))['selected_reading'],r['selected_reading'])
    def test_identical_runs_are_one_reading_not_offsets(self):
        c=dict(offsets=dict(upper=0,count=0,lower=None),runs=[dict(symbol='D',count=1)],row_total=1)
        other=copy.deepcopy(c);other['offsets']['upper']=1
        space=reading_space([[c,other]])
        self.assertEqual(space['distinct_readings'],1);self.assertEqual(len(space['selected']['logical_rows'][0]['supporting_offsets']),2)
    def test_row_total_agreement_required(self):
        rows=[dict(upper='D',count='a'),dict(upper='D',count='b')]
        self.assertEqual(run(rows)['decision']['outcome'],'REVIEW_UNCERTAIN')
    def test_complete_multirow_reading(self):
        rows=UNIQUE+UNIQUE;m=measured(rows,[dict(upper=0,count=0),dict(upper=0,count=0)])
        r=run(rows,m);self.assertTrue(r['carpet_generation_ready']);self.assertEqual(len(r['selected_reading']['logical_rows']),2)
    def test_unavailable_candidates_have_explicit_trace(self):
        r=run();self.assertEqual(len(r['candidate_trace']),2)
        self.assertTrue(all(t['position_check']=='unavailable' and t['retained_by_position'] is None for t in r['candidate_trace']))
    def test_fabricated_or_missing_runs_blocked(self):
        for bad in ([],[dict(symbol='D',count=259)]):
            a=analysis(UNIQUE);a['rows'][0]['successful_candidates'][0]['runs']=bad
            r=decide(base(),UNIQUE,a);self.assertEqual(r['findings'][0]['reason_code'],'R3_1_ALIGNMENT_UNAVAILABLE')
    def test_incomplete_evidence_not_accepted(self):
        a=analysis(UNIQUE);a['all_rows_exhaustive_in_literal_model']=False
        self.assertEqual(decide(base(),UNIQUE,a)['decision']['outcome'],'REVIEW_UNCERTAIN')
    def test_inputs_not_modified(self):
        a=analysis(MULTIPLE);b=base();original=copy.deepcopy((a,b));decide(b,MULTIPLE,a);self.assertEqual((a,b),original)
    def test_explicit_status_on_every_accept(self):
        for m in (None,measured(UNIQUE,[dict(upper=0,count=0)])):
            r=run(UNIQUE,m);self.assertTrue(r['selected_reading_status'].startswith('unique_'));self.assertIsNotNone(r['selected_reading'])
    def test_readiness_invariant_rejects_missing_reading(self):
        r=run(UNIQUE);r.update(carpet_generation_ready=True,image_position_verified=True,distinct_readings_after=1,selected_reading=None)
        with self.assertRaises(AssertionError):assert_invariants('ACCEPT',r)


if __name__=='__main__':unittest.main()
