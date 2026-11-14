import unittest,sys,copy,tempfile,types,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import evaluate_v3_1 as ev
from paths_v3_1 import config,content_hash,load_json,digest
from ink_measurement import D1Only
from offline_scoring import score
from test_policy_v3_1 import base,analysis,measured,MULTIPLE,UNIQUE,run


def source(valid=True):
    return dict(validity_status='VALID' if valid else 'INVALID',invalidity_category='none' if valid else 'missing_count',augmentation_recipe=None,
                raw_display=[dict(upper='D',count_line='a',lower='')],original_structured_notation=dict(logical_rows=[dict(runs=[dict(symbol='D',count=1)])]))


def prior(text=UNIQUE):
    return dict(image_id='synthetic',source_parent_id='synthetic_parent',image_sha256='fake',R3=base(),interpreted_transcription=text,
                alignment_evidence=dict(analysis=analysis(text)),findings=[dict(old_finding=True)],review_regions={'regions':[]},localization={'old':True})


class Tests(unittest.TestCase):
    def test_offline_labels_do_not_change_decision(self):
        a=source();b=source(False);b['mutation_parameters']={'poison':'must never reach policy'}
        p=prior(MULTIPLE);m=measured(MULTIPLE,[dict(upper=0,count=0,lower=1)])
        self.assertEqual(ev.process_case(p,{},m,a)['R3_1'],ev.process_case(p,{},m,b)['R3_1'])
    def test_old_findings_regions_and_localization_retained(self):
        p=prior();r=ev.process_case(p,{},None,source())
        self.assertEqual(r['prior_R3_findings'],p['findings']);self.assertEqual(r['prior_review_regions'],p['review_regions']);self.assertEqual(r['prior_localization'],p['localization'])
    def test_selected_reading_strict_scoring(self):
        m=measured(UNIQUE,[dict(upper=0,count=0)]);r=run(UNIQUE,m)
        self.assertTrue(score(r,m,source(),UNIQUE)['selected_reading_correct'])
        bad=source();bad['original_structured_notation']['logical_rows'][0]['runs'][0]['count']=2
        self.assertFalse(score(r,m,bad,UNIQUE)['selected_reading_correct'])
    def test_invalid_reference_carpet_not_fabricated(self):self.assertFalse(score(run(UNIQUE),None,source(False),UNIQUE)['selected_reading_eligible'])
    def test_mutated_display_offset_scoring(self):
        m=measured(UNIQUE,[dict(upper=0,count=1)]);r=run(UNIQUE,m)
        s=source(False);s['raw_display'][0]['count_line']=' a'
        scores=score(r,m,s,UNIQUE)['tiers'];self.assertTrue(all(t['exact'] for t in scores))
    def test_only_development_cli(self):
        with patch.object(sys,'argv',['evaluate_v3_1.py','--split','protected_test','--preflight']),patch.object(ev,'load_inputs') as loader:
            with self.assertRaises(SystemExit):ev.main()
            loader.assert_not_called()
    def test_output_cannot_escape_runs(self):
        with self.assertRaises(ValueError):ev.output_path(Path(tempfile.gettempdir())/'escape')
    def test_preflight_does_not_initialize_d1_or_read_pixels(self):
        with patch.object(ev,'load_inputs',return_value=(types.SimpleNamespace(groups={'p':[]}),[{}],{},{},{})),patch.object(ev,'D1Only') as d1,patch.object(ev,'image_bytes') as image:
            ev.evaluate(types.SimpleNamespace(preflight=True,measure_scope='accepted'))
            d1.assert_not_called();image.assert_not_called()
    def test_original_d1_only_blank_synthetic_smoke(self):
        import cv2
        raw=cv2.imencode('.png',np.full((64,128),255,np.uint8))[1].tobytes()
        d1=D1Only();result=d1.measure(raw,[],[],config())
        self.assertEqual(result['status'],'unavailable');self.assertTrue(result['reason'].startswith('D1_REJECTED'))
        self.assertNotIn('rejection_v2_original_pixel_pipeline',sys.modules)
    def test_small_mock_completed_evaluation(self):
        b=dict(image_id='synthetic',source_parent_id='synthetic_parent',image_sha256='fake',interpreted_transcription=UNIQUE,
               actual_outcome='ACCEPT',decision=base(),validity='VALID',expected_outcome='ACCEPT',
               interpreted_errors=dict(characters=2,edits=0,exact=True),raw_errors=dict(characters=2,edits=0,exact=True),
               correction_assessments=[],corrections=[],triggered_reason_codes=['BASE'],raw_decision={'reason_codes':['BASE']},interpreted_syntax={'outcome':'ACCEPT'},
               preprocessing_failure=False,segmentation_failure=False,ocr_failure=False,pipeline_exception=None,invalidity_category='none',paired_mutation_category='missing_count',
               variant='clean',length_group='short',row_group='single_row')
        ds=types.SimpleNamespace(by_id={'synthetic':source()},directory=Path('unused'),groups={'synthetic_parent':[]})
        data=(ds,[b],{'synthetic':{'segmentation_rows':[]}}, {'synthetic':prior()}, {'r3_completion_sha256':'fake'})
        with tempfile.TemporaryDirectory() as tmp,patch.object(ev,'load_inputs',return_value=data),patch.object(ev,'verify_locks'),patch.object(ev,'verify_r2'),patch.object(ev,'output_path',side_effect=lambda p:Path(p)),patch.object(ev,'image_bytes',return_value=b'synthetic'),patch.object(ev,'D1Only') as d1,patch.object(ev,'own_hashes',return_value={}):
            d1.return_value.measure.return_value=measured(UNIQUE,[dict(upper=0,count=0)])
            ev.evaluate(types.SimpleNamespace(preflight=False,output_dir=Path(tmp)/'run',measure_scope='accepted'))
            out=Path(tmp)/'run';done=load_json(out/'completion.json');metrics=load_json(out/'metrics.json')
            self.assertTrue(done['complete']);self.assertEqual(metrics['new_definitive_rejections'],0);self.assertEqual(metrics['carpet_generation_ready'],1)
            for name,h in done['outputs'].items():self.assertEqual(digest(out/name),h)


if __name__=='__main__':unittest.main()
