import unittest,sys,copy,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ink_measurement import estimate_pitch
from offline_scoring import score,summary
from test_ink_measurement import count_tier
from test_policy_v3_1 import run,UNIQUE,measured
from test_evaluation import source


class Tests(unittest.TestCase):
    def test_different_glyph_bearings_cancel_in_same_shape_pairs(self):
        t=count_tier((5,18,25,38,45,58),'abcdef')
        for i,g in enumerate(t['groups']):g['shape_sha256']='even' if i%2==0 else 'odd'
        p=estimate_pitch([t]);self.assertEqual(p['pitch_pixels'],10)
        self.assertTrue(all(o['decoded_column_gap']==2 for o in p['observations']))
    def test_unique_silhouettes_do_not_invent_pitch(self):
        t=count_tier()
        for i,g in enumerate(t['groups']):g['shape_sha256']=str(i)
        p=estimate_pitch([t]);self.assertEqual(p['status'],'unavailable');self.assertEqual(p['usable_observations'],0)
    def test_two_translation_observations_are_insufficient(self):
        p=estimate_pitch([count_tier((5,15,25),'aaa')]);self.assertEqual(p['status'],'unavailable')
    def test_conflicting_shape_translation_pitches_unavailable(self):
        a=count_tier();b=count_tier((5,25,45,65));b['row']=1
        self.assertEqual(estimate_pitch([a,b])['status'],'unavailable')
    def test_decoded_column_gaps_not_group_index_differences(self):
        t=count_tier((5,25,45,65),'a a a a');p=estimate_pitch([t]);self.assertEqual(p['pitch_pixels'],10)
    def test_shape_translation_interval_has_pixel_units(self):
        p=estimate_pitch([count_tier((5,25,45,65),'a a a a')])
        self.assertEqual(p['pitch_interval'],[9.5,10.5])
    def test_no_source_pitch_argument(self):
        with self.assertRaises(TypeError):estimate_pitch([count_tier()],ground_truth_pitch=50)
    def test_rich_source_run_metadata_does_not_create_false_carpet(self):
        m=measured(UNIQUE,[dict(upper=0,count=0)]);r=run(UNIQUE,m);s=source()
        s['original_structured_notation']['logical_rows'][0]['runs'][0].update(direction='UP',encoded_count='a')
        result=score(r,m,s,UNIQUE);self.assertTrue(result['selected_reading_eligible']);self.assertTrue(result['selected_reading_correct'])
    def test_unavailable_physical_selection_is_unscorable(self):
        r=run(UNIQUE);s=source();result=score(r,None,s,UNIQUE)
        self.assertFalse(result['selected_reading_eligible']);self.assertIsNone(result['selected_reading_correct'])
        records=[dict(validity='VALID',R3_1=r,offline_score=result)]
        totals=summary(records);self.assertEqual(totals['selected_reading_false'],0);self.assertIsNone(totals['false_selected_reading_rate'])
        self.assertEqual(totals['valid_ACCEPT_unavailable_selected_reading'],1)
    def test_unavailable_selection_not_incorrect_even_if_text_differs(self):
        r=run(UNIQUE);s=source();s['original_structured_notation']['logical_rows'][0]['runs'][0]['count']=2
        self.assertIsNone(score(r,None,s,UNIQUE)['selected_reading_correct'])
    def test_truly_wrong_verified_selection_still_false(self):
        m=measured(UNIQUE,[dict(upper=0,count=0)]);r=run(UNIQUE,m);s=source()
        s['original_structured_notation']['logical_rows'][0]['runs'][0]['count']=2
        result=score(r,m,s,UNIQUE);self.assertFalse(result['selected_reading_correct'])
        self.assertEqual(summary([dict(validity='VALID',R3_1=r,offline_score=result)])['selected_reading_false'],1)
    def test_three_cached_diagnostic_cases_keep_support_limits(self):
        fixtures=json.loads(Path(__file__).with_name('cached_pitch_examples.json').read_text(encoding='utf-8'))
        for f in fixtures:
            p=estimate_pitch(f['count_tiers'])
            self.assertEqual(p['status'],f['expected_status'])
            if p['status']=='measured':self.assertEqual(p['pitch_pixels'],50)


if __name__=='__main__':unittest.main()
