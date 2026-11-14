import unittest,sys,copy,tempfile,types
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE/'src'))
from text_alignment import *
from evaluate_text_alignment import evaluate_case,evaluate_records,FONT_PROFILE,summarize


def decoded(runs):
    u,c,l=rules.canonical_row([dict(symbol=s,count=n) for s,n in runs])
    return dict(upper=u.lstrip(' '),count=c.lstrip(' '),lower=l.lstrip(' '))


class Tests(unittest.TestCase):
    def test_upper_legal_count(self):
        r=analyze_row(dict(upper='D',count='a'))
        self.assertEqual(r['passing_count'],1);self.assertEqual(r['row_totals'],[1]);self.assertTrue(r['unique_reconstruction_proven_in_model'])
    def test_lower_legal_multichar_count(self):
        r=analyze_row(dict(lower='R',count=rules.encode_count(108)))
        self.assertEqual(r['passing_count'],1);self.assertEqual(r['row_totals'],[108])
    def test_multiple_symbols_and_counts(self):
        r=analyze_row(decoded([('D',11),('R',2),('H',8)]))
        self.assertTrue(r['passing_count']);self.assertIn(21,r['row_totals'])
    def test_alternative_leading_offsets_are_enumerated_not_supplied(self):
        r=analyze_row(dict(upper='D',lower='R',count='aa'))
        offsets=[c['offsets'] for c in r['successful_candidates']]
        self.assertIn(dict(upper=0,count=0,lower=1),offsets);self.assertIn(dict(upper=1,count=0,lower=0),offsets)
    def test_observed_leading_spaces_are_preserved(self):
        self.assertEqual(validate_aligned_row(' D','aa','R')['outcome'],'ACCEPT')
        r=analyze_row(dict(upper=' D',lower='R',count='aa'))
        self.assertEqual(r['status'],'unsupported');self.assertFalse(r['coverage']['conditional_text_model_exhaustive'])
        self.assertEqual(r['observed_edge_spaces']['upper']['leading'],1)
    def test_trailing_spaces_not_silently_removed(self):
        self.assertEqual(validate_aligned_row('D',rules.encode_count(11),'')['outcome'],'ACCEPT')
        r=analyze_row(dict(upper='D ',count=rules.encode_count(11)))
        self.assertEqual(r['passing_count'],0);self.assertTrue(r['conditional_text_contradiction'])
    def test_empty_optional_tiers(self):
        self.assertEqual(analyze_row(dict(upper='',count='a',lower='R'))['passing_count'],1)
        self.assertEqual(analyze_row(dict(upper=' ',count='a',lower='R'))['status'],'unsupported')
    def test_invalid_single_token_order_known_and_reconstructed(self):
        self.assertNotIn('hs',{rules.encode_count(n) for n in range(1,260)})
        self.assertEqual(validate_aligned_row('D','hs','')['outcome'],'REJECT_INVALID')
        a=analyze_observation([dict(upper='D',count='hs')])
        self.assertTrue(a['conditional_text_contradiction']);self.assertFalse(a['image_geometry_exhaustive'])
        self.assertEqual(review_only_comparison('ACCEPT',a),'REVIEW_UNCERTAIN')
    def test_missing_first_count_known_geometry_vs_lost_prefix(self):
        self.assertEqual(validate_aligned_row('D',' sh',' R')['outcome'],'REJECT_INVALID')
        r=analyze_row(dict(upper='D',lower='R',count='sh'))
        self.assertTrue(r['passing_count'])  # alternative tokens s,h; not repaired original geometry
    def test_missing_middle_count(self):
        self.assertEqual(validate_aligned_row('D F','a a',' R')['outcome'],'REJECT_INVALID')
        a=analyze_observation([dict(upper='D F',count='a a',lower='R')])
        self.assertTrue(a['conditional_text_contradiction']);self.assertFalse(a['image_geometry_exhaustive'])
    def test_missing_last_count_with_plausible_reinterpretation(self):
        self.assertEqual(validate_aligned_row('D','sh ','  R')['outcome'],'REJECT_INVALID')
        self.assertTrue(analyze_row(dict(upper='D',count='sh',lower='R'))['passing_count'])
    def test_wrong_tier_glyph(self):
        a=analyze_observation([dict(lower='D',count='a')]);self.assertTrue(a['conditional_text_contradiction'])
        self.assertEqual(a['rows'][0]['failure_reasons'],{'WRONG_SYMBOL_DIRECTION':1})
    def test_ambiguity_is_not_uniqueness(self):
        r=analyze_row(dict(upper='D',lower='R',count='aa'))
        self.assertEqual(r['passing_count'],2);self.assertFalse(r['unique_reconstruction_proven_in_model'])
    def test_space_collapse_can_look_valid(self):
        known=validate_aligned_row('D F','a sh',' R')
        self.assertEqual(known['outcome'],'REJECT_INVALID')
        self.assertTrue(analyze_row(dict(upper='D F',lower='R',count='ash'))['passing_count'])
    def test_space_collapse_can_create_false_text_contradiction(self):
        good=decoded([('D',1),('R',1),('D',1)])
        self.assertTrue(analyze_row(good)['passing_count'])
        bad=dict(good,upper='DD');a=analyze_observation([bad])
        self.assertTrue(a['conditional_text_contradiction']);self.assertFalse(a['image_geometry_exhaustive'])
    def test_repeated_characters_and_adjacent_identical_runs(self):
        self.assertTrue(analyze_row(decoded([('D',1),('R',1),('D',1)]))['passing_count'])
        self.assertEqual(analyze_row(dict(upper='DD',count='aa'))['passing_count'],0)
    def test_conflicting_row_totals_are_candidate_derived(self):
        a=analyze_observation([dict(upper='D',count='a'),dict(lower='R',count='b')])
        self.assertEqual(a['common_row_totals'],[]);self.assertTrue(a['conditional_text_contradiction'])
        b=analyze_observation([dict(upper='D',lower='R',count='aa'),dict(lower='R',count='b')])
        self.assertEqual(b['common_row_totals'],[2]);self.assertTrue(b['any_multiple_candidates'])
        self.assertEqual(b['page_candidate_count_in_search'],2)
        c=analyze_observation([dict(upper='D',lower='R',count='aa')]*2)
        self.assertEqual(c['page_candidate_count_in_search'],4)
    def test_unavailable_OCR_and_segmentation(self):
        for kwargs in (dict(segmentation_accepted=False),dict(ocr_status='not_run'),dict(preprocessing_status='no_ink')):
            a=analyze_observation([],**kwargs);self.assertFalse(a['conditional_text_contradiction']);self.assertEqual(a['status'],'unknown_incomplete_or_unsupported')
        self.assertEqual(analyze_row(dict(upper='D',count=''))['status'],'unsupported')
    def test_offset_and_trial_caps_are_incomplete(self):
        row=dict(upper='D',lower='R',count='aa')
        a=analyze_observation([row],max_offset=0);self.assertFalse(a['conditional_text_contradiction']);self.assertEqual(a['status'],'unknown_incomplete_or_unsupported')
        r=analyze_row(row,max_candidates=2);self.assertEqual(r['passing_count'],1);self.assertFalse(r['unique_reconstruction_proven_in_model'])
    def test_validator_uncertainty_is_not_contradiction(self):
        symbol=sorted(rules.AMBIGUOUS_SYMBOLS)[0]
        a=analyze_observation([dict(upper=symbol,count='a')]);self.assertTrue(a['any_validator_uncertainty']);self.assertFalse(a['conditional_text_contradiction'])
    def test_no_expected_cells_or_ground_truth_offsets(self):
        original=validate_aligned_row
        with patch('text_alignment.validate_aligned_row',wraps=original) as validator:
            analyze_row(dict(upper='D',lower='R',count='aa'))
            self.assertTrue(validator.call_count);self.assertTrue(all(len(c.args)==3 and not c.kwargs for c in validator.call_args_list))
        with self.assertRaises(TypeError):analyze_row(dict(upper='D',count='a'),expected_cells=1)
    def test_review_only_preserves_prior_outcomes(self):
        a=analyze_observation([dict(lower='D',count='a')])
        self.assertEqual(review_only_comparison('REJECT_INVALID',a),'REJECT_INVALID')
        self.assertEqual(review_only_comparison('REVIEW_UNCERTAIN',a),'REVIEW_UNCERTAIN')
    def test_tiny_offline_evaluation_no_leakage(self):
        r=dict(image_id='synthetic',source_parent_id='parent',interpreted_transcription=[dict(upper='D',count='hs')],actual_outcome='ACCEPT')
        raw=dict(preprocessing_status='ok',segmentation_accepted=True,ocr_status='ok')
        source=dict(layout=dict(font_size_px=48,font_sha256=FONT_PROFILE['font_sha256']),raw_display=[dict(upper='D',count_line='hs',lower='')],
            validity_status='INVALID',invalidity_category='invalid_count_token',augmentation_recipe=None,
            mutation_parameters=dict(row=0,column=0,displayed_mutated_token='hs'))
        a=evaluate_case(r,raw,source,128,258)
        poisoned=copy.deepcopy(source);poisoned['validity_status']='VALID';poisoned['invalidity_category']='none';poisoned['raw_display'][0]['count_line']='a'
        b=evaluate_case(r,raw,poisoned,128,258)
        self.assertEqual(a['analysis'],b['analysis']);self.assertEqual(a['review_only_counterfactual'],b['review_only_counterfactual'])
        ds=types.SimpleNamespace(by_id={'synthetic':source})
        cases,metrics=evaluate_records(ds,[r],{'synthetic':raw},128,258)
        self.assertEqual(metrics['overall']['invalid_ACCEPT_to_REVIEW']['images'],1);self.assertTrue(metrics['overall']['no_new_definitive_rejection'])


if __name__=='__main__':unittest.main()
