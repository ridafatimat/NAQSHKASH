import copy
import unittest
import tempfile
from pathlib import Path
from runtime import *
from tier_aware_decoder import interpret
from contextual_pipeline import process, role_evidence
from evaluate_tier_aware import evaluate_item, load_saved
from compare_tier_aware import compare, load_result, gallery
from measurements import page_errors, assess_corrections, grouped, alignment_targets


def observation(upper='A', lower='X'):
    decoded = [dict(upper=upper, count='ja', lower=lower)]
    roles = ['upper','count','lower']
    segments = [[dict(row=0,role=r,box=[0,10*j,100,10*j+8]) for j,r in enumerate(roles)]]
    tiers = [dict(**t,prediction=decoded[0][t['role']]) for t in segments[0]]
    return dict(preprocessing_status='ok',segmentation_accepted=True,ocr_status='ok',decoded_rows=decoded,
        segmentation_rows=segments,tiers=tiers,confidence={},
        diagnostics=dict(legal_alternatives=2,role_margin=1.,rejection='',path=[[0,roles]]))


class DecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping=load_registry()
        cls.rendering=load_json(WORKSPACE/'configs/rendering_profile.json')
        cls.on=load_json(WORKSPACE/'configs/t1_tier_aware.yaml')
        cls.off=load_json(WORKSPACE/'configs/t0_baseline.yaml')

    def decode(self,text,role,**kw):
        return interpret(text,role,kw.pop('metadata',dict(rendering_context=self.rendering)),
            kw.pop('mapping',self.mapping),kw.pop('evidence',dict(reliable=True,role=role)),**kw)

    def test_verified_pair_at_multiple_sizes_and_limit(self):
        audit=load_json(WORKSPACE/'reports/glyph_audit_summary.json')
        self.assertIn(['B','X'],audit['exact_pairs_by_size']['48'])
        self.assertNotIn(['B','X'],audit['exact_pairs_by_size']['32'])
        self.assertEqual({(r['source'],r['target']) for r in self.mapping},{('B','X'),('X','B')})

    def test_upper_context(self):
        r=self.decode('X','upper')
        self.assertEqual(r['interpreted_prediction'],'B')
        self.assertTrue(r['corrected'])
        self.assertEqual(r['corrections'][0]['original_character'],'X')

    def test_lower_context(self):
        self.assertEqual(self.decode('B','lower')['interpreted_prediction'],'X')

    def test_missing_and_ambiguous_role(self):
        for role in (None,'unknown'):
            r=self.decode('X',role)
            self.assertTrue(r['uncertain']);self.assertEqual(r['interpreted_prediction'],'X')
        self.assertFalse(self.decode('X','upper',evidence=dict(reliable=False,role='upper'))['corrected'])

    def test_unknown_character_is_not_repaired(self):
        r=self.decode('?','upper')
        self.assertTrue(r['uncertain']);self.assertEqual(r['interpreted_prediction'],'?')

    def test_similar_not_equivalent(self):
        self.assertFalse(self.decode('Y','upper')['corrected'])
        self.assertFalse(self.decode('G','lower')['corrected'])
        self.assertFalse(self.decode('L','lower')['corrected'])

    def test_multiple_targets_remain_uncertain(self):
        # Deliberately ambiguous synthetic registry; never written to production mapping.
        mapping=copy.deepcopy(self.mapping)
        other=copy.deepcopy(next(r for r in mapping if r['source']=='X'))
        other['target']='C';mapping.append(other)
        r=self.decode('X','upper',mapping=mapping)
        self.assertIn('T1_MULTIPLE_ROLE_EQUIVALENTS',r['reason_codes'])
        self.assertFalse(r['corrected'])

    def test_already_valid_prediction_unchanged(self):
        for text,role in [('A B','upper'),('X R','lower'),('jaji','count')]:
            r=self.decode(text,role)
            self.assertFalse(r['corrected']);self.assertFalse(r['uncertain'])

    def test_raw_output_and_input_objects_preserved(self):
        obs=observation('X');before=copy.deepcopy(obs)
        result=process(obs,self.on,self.mapping,self.rendering)
        self.assertEqual(obs,before)
        self.assertEqual(result['raw_observation'],before)
        self.assertEqual(result['raw_observation']['decoded_rows'][0]['upper'],'X')

    def test_preserve_original_f1_code_and_evidence(self):
        result=process(observation('X'),self.on,self.mapping,self.rendering)
        self.assertIn('DECODED_UNSUPPORTED_OR_WRONG_TIER_GLYPH',result['original_reason_codes'])
        self.assertEqual(result['original_evidence']['evidence']['glyphs'],['X'])
        self.assertTrue(result['syntax_failure_resolved'])

    def test_runtime_ground_truth_independence(self):
        metadata=dict(rendering_context=self.rendering)
        r=self.decode('X','upper',metadata=metadata)
        metadata.update(image_ground_truth='invalid',transcription_exact=True,expected_system_outcome='REJECT_INVALID',truth='Z')
        self.assertEqual(r,self.decode('X','upper',metadata=metadata))
        obs=observation('X');before=process(obs,self.on,self.mapping,self.rendering)
        obs.update(image_ground_truth='invalid',expected_system_outcome='REJECT_INVALID',truth_rows=[{'upper':'Y'}])
        after=process(obs,self.on,self.mapping,self.rendering)
        self.assertEqual(before['final_decision'],after['final_decision'])
        self.assertEqual(before['interpreted_observation']['decoded_rows'],after['interpreted_observation']['decoded_rows'])

    def test_f1_v2_compatibility(self):
        r=process(observation('X'),self.on,self.mapping,self.rendering)
        self.assertEqual(r['raw_f1_decision']['outcome'],'REVIEW_UNCERTAIN')
        self.assertEqual(r['final_decision']['outcome'],'ACCEPT')
        self.assertEqual(r['final_decision']['decision_policy_version'],'f1_ocr_uncertainty_v2')
        self.assertEqual(r['interpreted_observation']['decoded_rows'][0]['upper'],'B')

    def test_disabled_is_exact_baseline(self):
        obs=observation('X')
        r=process(obs,self.off,self.mapping,{})
        self.assertEqual(r['final_decision'],decide(obs,'F1'))
        self.assertEqual(r['interpreted_observation'],obs)
        self.assertEqual(r['corrections'],[])

    def test_long_sequences_internal_spaces(self):
        text=('A  X B '*80).rstrip()
        r=self.decode(text,'upper')
        self.assertEqual(r['raw_prediction'],text)
        self.assertEqual(r['interpreted_prediction'],text.replace('X','B'))
        self.assertEqual(len(r['interpreted_prediction']),len(text))
        self.assertEqual([i for i,c in enumerate(text) if c==' '],[i for i,c in enumerate(r['interpreted_prediction']) if c==' '])

    def test_wrong_render_scope_blocks_correction(self):
        for key,value in [('font_size_px',32),('font_sha256','other'),('source_mode','1'),('provenance','unknown')]:
            context=dict(self.rendering);context[key]=value
            r=self.decode('X','upper',metadata=dict(rendering_context=context))
            self.assertFalse(r['corrected']);self.assertTrue(r['uncertain'])
        self.assertTrue(self.decode('X','upper',metadata={})['uncertain'])

    def test_conflicting_evidence_blocks(self):
        r=self.decode('X','upper',metadata=dict(rendering_context=self.rendering,conflicting_evidence=['role contested']))
        self.assertFalse(r['corrected']);self.assertTrue(r['uncertain'])

    def test_no_partial_correction_or_count_replacement(self):
        r=self.decode('X?','upper')
        self.assertFalse(r['corrected']);self.assertEqual(r['interpreted_prediction'],'X?')
        self.assertFalse(self.decode('X','count')['corrected'])

    def test_existing_d1_reliability_and_metadata_conflict(self):
        obs=observation('X')
        self.assertTrue(role_evidence(obs,0,'upper','X')['reliable'])
        obs['diagnostics']['role_margin']=0.
        self.assertFalse(role_evidence(obs,0,'upper','X')['reliable'])
        obs['diagnostics']['legal_alternatives']=1;obs['diagnostics']['role_margin']=None
        self.assertTrue(role_evidence(obs,0,'upper','X')['reliable'])
        obs['tiers'][0]['prediction']='A'
        self.assertFalse(role_evidence(obs,0,'upper','X')['reliable'])

    def test_independent_artifact_rejection_is_not_overridden(self):
        obs=observation('X');obs['preprocessing_status']='no_ink'
        r=process(obs,self.on,self.mapping,self.rendering)
        self.assertEqual(r['final_decision']['outcome'],'REJECT_INVALID')
        self.assertFalse(r['corrections'])

    def test_four_development_regressions_only_verified_pair_changes(self):
        fixture=load_json(WORKSPACE/'tests/fixtures/development_cases.json')
        self.assertIn('dev_baseline',fixture['source_baseline'])
        self.assertEqual(len(fixture['cases']),4)
        for item in fixture['cases']:
            r=process(item['observation'],self.on,self.mapping,self.rendering)
            if item['image_id']=='normal_train_044361__clean':
                self.assertTrue(r['corrections'])
                self.assertTrue(all(e['original_character']=='X' and e['replacement_character']=='B' for e in r['corrections']))
            else:
                self.assertFalse(r['corrections'])
                self.assertEqual(r['final_decision']['outcome'],'REVIEW_UNCERTAIN')


class MetricTests(unittest.TestCase):
    def item(self, id='unit',parent='parent',truth=None,upper='X'):
        return dict(record=dict(image_id=id,source_parent_id=parent,original_image_path='',data_origin='unit'),
                    observation=observation(upper),truth_rows=truth or [dict(upper='B',count='ja',lower='X')])

    def row(self,id='unit',parent='parent',upper='X'):
        return evaluate_item(self.item(id,parent,upper=upper),load_json(WORKSPACE/'configs/t1_tier_aware.yaml'),load_registry(),load_json(WORKSPACE/'configs/rendering_profile.json'))

    def test_exact_and_cer_track_transcription_not_just_accept(self):
        row=self.row()
        self.assertTrue(row['interpreted_errors']['exact'])
        self.assertFalse(row['raw_errors']['exact'])
        self.assertEqual(row['correction_assessments'][0]['assessment'],'correct')
        bad=self.item(upper='X');bad['truth_rows'][0]['upper']='A'
        r=evaluate_item(bad,load_json(WORKSPACE/'configs/t1_tier_aware.yaml'),load_registry(),load_json(WORKSPACE/'configs/rendering_profile.json'))
        self.assertEqual(r['decision']['outcome'],'ACCEPT')
        self.assertFalse(r['interpreted_errors']['exact'])
        self.assertEqual(r['correction_assessments'][0]['assessment'],'incorrect')

    def test_all_optimal_alignments_not_favorable_tie(self):
        self.assertEqual(alignment_targets('XX','B',0),{None,'B'})
        edits=[dict(row=0,tier_role='upper',index=0,original_character='X',replacement_character='B')]
        assessment=assess_corrections([{'upper':'B'}],[{'upper':'XX'}],edits)
        self.assertEqual(assessment[0]['assessment'],'uncertain_alignment')

    def test_missing_tiers_are_not_dropped(self):
        errors=page_errors([dict(upper='A',count='ja')],[])
        self.assertEqual(errors['edits'],3);self.assertEqual(errors['characters'],3)

    def test_internal_spaces_in_cer(self):
        self.assertEqual(page_errors([{'upper':'A  B'}],[{'upper':'A B'}])['edits'],1)

    def test_parent_macro_weighting(self):
        rows=[self.row(str(i),'large') for i in range(9)]
        small=self.row('small','small',upper='A')
        rows.append(small)
        summary,parents,_=grouped(rows)
        self.assertEqual(summary['exact_transcription_accuracy'],.9)
        self.assertEqual(summary['macro_parent']['exact_transcription_accuracy'],.5)
        self.assertEqual(len(parents),2)

    def test_exposed_partition_is_rejected_before_loading(self):
        with self.assertRaises(ValueError):load_saved(LEGACY/'metrics/20261106_154728_812_eval_baseline')

    def test_paired_comparison_guards(self):
        row=self.row()
        a=dict(candidate='T0',rows=[row],mapping_sha256='same',code_sha256={},baseline_lineage=dict(manifest_sha256='same',observations_sha256='same',checkpoint_sha256='same'))
        b=copy.deepcopy(a);b['candidate']='T1'
        report,_=compare(a,b)
        self.assertEqual(report['T0']['images'],1)
        b['baseline_lineage']['observations_sha256']='other'
        with self.assertRaises(ValueError):compare(a,b)

    def test_historical_inputs_checkpoint_and_f1_unchanged(self):
        for p,h in load_json(WORKSPACE/'configs/historical_lock.json').items():
            self.assertEqual(digest(p),h,p)

    def test_experiment_code_and_configuration_frozen(self):
        self.assertTrue(verify_integrity()['unchanged'])

    def test_completed_output_and_gallery_roundtrip(self):
        row=self.row()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            result=dict(candidate='T1',rows=[row],mapping_sha256='unit',code_sha256={},baseline_lineage={})
            write_json(path/'results.json',result)
            write_json(path/'completion.json',dict(complete=True,rows=1,results_sha256=digest(path/'results.json')))
            self.assertEqual(load_result(path),result)
            serialize_csv(path/'per_image.csv',[row])
            exported=read_csv(path/'per_image.csv')
            self.assertEqual(json.loads(exported[0]['corrections']),row['corrections'])
            gallery(path/'gallery.html',[row],1)
            self.assertIn('Contextual edits', (path/'gallery.html').read_text(encoding='utf-8'))

    def test_unresolved_context_cannot_turn_into_accept(self):
        obs=observation('X');obs['diagnostics']['role_margin']=0.
        result=process(obs,load_json(WORKSPACE/'configs/t1_tier_aware.yaml'),load_registry(),load_json(WORKSPACE/'configs/rendering_profile.json'))
        self.assertFalse(result['corrections'])
        self.assertEqual(result['final_decision']['outcome'],'REVIEW_UNCERTAIN')
        # Even role-legal text is held for explicit unresolved D1 context.
        obs=observation('A');obs['diagnostics']['role_margin']=0.
        result=process(obs,load_json(WORKSPACE/'configs/t1_tier_aware.yaml'),load_registry(),{})
        self.assertEqual(result['raw_f1_decision']['outcome'],'ACCEPT')
        self.assertIn('T1_CONTEXT_UNCERTAIN',result['final_decision']['reason_codes'])


if __name__=='__main__':unittest.main()
