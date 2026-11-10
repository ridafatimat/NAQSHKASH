import unittest,sys,copy,tempfile,io,types,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from policy_v3 import decide,load_policy,evidence
from review_overlay import plan_regions,validate_geometry,render_image,generate_gallery,LABEL
from localization_metrics import score,summarize
import evaluate_v3 as ev

STATUS=dict(preprocessing_status='ok',segmentation_accepted=True,ocr_status='ok')
def base():return dict(outcome='ACCEPT',reason_codes=['BASE'],reason_details=[])
def finding():return dict(logical_row_index=0,tier=None,candidate_text_columns=[99999])
def geometry():return dict(verified=True,source_dimensions=[100,100],processed_dimensions=[100,100],source_to_processed_matrix=[[1,0,0],[0,1,0],[0,0,1]],
    rows=[[dict(role='upper',box=[2,2,90,20]),dict(role='count',box=[2,22,90,43]),dict(role='lower',box=[2,45,90,70])]])
def source(validity='INVALID'):
    return dict(validity_status=validity,invalidity_category='invalid_count_token' if validity=='INVALID' else 'none',augmentation_recipe=None,
        raw_display=[dict(upper='D ',count_line='ab',lower=' R')],
        mutation_parameters=dict(row=0,column=1,width=1,target_tier='count_line',source_tier='count_line'),
        layout=dict(padding=5,extra_left=0,cell_width=10,line_height=20,row_gap=5))
def record():
    return dict(image_id='synthetic',source_parent_id='p',validity='INVALID',new_review=True,OCR_available=True,R3=dict(outcome='REVIEW_UNCERTAIN',explanation='Review'),
        interpreted_transcription=[dict(upper='D',count='ab',lower='R')],
        findings=[dict(logical_row_index=0,tier='count',candidate_text_columns=[1],explanation='Row 1: candidate text conflict')],
        review_regions=plan_regions([finding()],geometry()))


class Tests(unittest.TestCase):
    def test_row_location_and_human_numbers(self):
        r=decide(base(),[dict(lower='D',count='a')],STATUS)
        f=r['findings'][0];self.assertEqual(f['logical_row_index'],0);self.assertEqual(f['human_readable_row_number'],1);self.assertIn('Row 1',f['explanation'])
    def test_candidate_text_columns_never_become_pixels(self):
        r=decide(base(),[dict(upper='D',lower='R',count='a')],STATUS);f=r['findings'][0]
        self.assertEqual(f['candidate_text_columns'],[0]);self.assertEqual(f['human_readable_candidate_columns'],[1]);self.assertFalse(f['image_geometry_verified'])
        self.assertEqual(f['source_image_regions'],[]);self.assertFalse(f['candidate_columns_exhaustive'])
    def test_no_exact_column_when_unsupported(self):
        r=decide(base(),[dict(upper=' D',count='a')],STATUS);self.assertEqual(r['findings'][0]['candidate_text_columns'],[])
    def test_ambiguity_does_not_reuse_rejected_candidate_error_columns(self):
        rows=[dict(upper='D',lower='R',count='aa')]
        analysis=evidence(rows,STATUS,load_policy())['analysis']
        from policy_v3 import findings_for
        f=findings_for(analysis,rows)[0]
        self.assertEqual(f['candidate_text_columns'],[]);self.assertGreater(len(f['candidate_offsets']),1)
    def test_total_conflict_does_not_invent_faulty_row(self):
        r=decide(base(),[dict(upper='D',count='a'),dict(lower='R',count='b')],STATUS);f=r['findings'][0]
        self.assertIsNone(f['logical_row_index']);self.assertEqual(f['human_readable_candidate_rows'],[1,2])
    def test_verified_broad_row_overlay(self):
        r=plan_regions([finding()],geometry())['regions'][0]
        self.assertEqual(r['source_box'],[2.,2.,90.,70.]);self.assertTrue(r['approximate']);self.assertIn('precise invalid glyph',r['label'])
    def test_text_column_changes_do_not_move_overlay(self):
        a=finding();b=copy.deepcopy(a);b['candidate_text_columns']=[0,1,2]
        self.assertEqual(plan_regions([a],geometry()),plan_regions([b],geometry()))
    def test_unverified_mapping_has_no_box(self):self.assertEqual(plan_regions([finding()],None)['regions'],[])
    def test_singular_and_out_of_bounds_mapping_rejected(self):
        g=geometry();g['source_to_processed_matrix']=[[0,0,0]]*3
        self.assertEqual(plan_regions([finding()],g)['regions'],[])
        g=geometry();g['rows'][0][0]['box']=[-2,0,300,20]
        self.assertEqual(plan_regions([finding()],g)['regions'],[])
    def test_render_synthetic_overlay_only(self):
        from PIL import Image
        b=io.BytesIO();Image.new('L',(100,100),255).save(b,format='PNG')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'overlay.png';render_image(b.getvalue(),plan_regions([finding()],geometry())['regions'],p);self.assertTrue(p.exists())
    def test_no_geometry_gallery_does_not_read_image_bytes(self):
        r=record();r['review_regions']=dict(regions=[],unavailable_reason='No transform');r['image_sha256']='test'
        ds=types.SimpleNamespace(by_id={},directory=Path('C:/unused'))
        with tempfile.TemporaryDirectory() as tmp,patch('dataset_loader.image_bytes',side_effect=AssertionError('Pixels must not be read')):
            out=Path(tmp)/'gallery';info=generate_gallery([r],ds,out,1)
            self.assertEqual(info['text_only'],1);self.assertIn('No verified image-coordinate region',(out/'index.html').read_text())
    def test_overlay_protected_identity_blocked(self):
        r=record();r['image_sha256']='test';ds=types.SimpleNamespace(by_id={'synthetic':dict(dataset_split='protected_test',image_sha256='test')},directory=Path('C:/unused'))
        with tempfile.TemporaryDirectory() as tmp,patch('dataset_loader.image_bytes',side_effect=AssertionError('No protected read')):
            with self.assertRaises(ValueError):generate_gallery([r],ds,Path(tmp)/'gallery',1)
    def test_offline_localization_distinguishes_pixel_precision(self):
        r=record();s=score(r,source());self.assertTrue(s['metrics']['logical_row']['eligible']);self.assertTrue(s['metrics']['text_column']['candidate_hit'])
        self.assertTrue(s['metrics']['broad_image_region']['candidate_hit']);self.assertFalse(s['precise_pixel_localization_eligible'])
        r['localization']=s;m=summarize([r]);self.assertEqual(m['precise_pixel_localization']['eligible'],0);self.assertIsNone(m['precise_pixel_localization']['conditional_accuracy'])
    def test_wrong_transcription_column_is_unscorable(self):
        r=record();r['interpreted_transcription'][0]['count']='ba'
        self.assertFalse(score(r,source())['metrics']['text_column']['eligible'])
    def test_valid_review_has_no_true_invalid_location(self):
        r=record();r['validity']='VALID';r['localization']=score(r,source('VALID'))
        self.assertEqual(r['localization']['scope'],'valid_review_no_true_invalid_mutation');self.assertEqual(summarize([r])['valid_review_availability']['images'],1)
    def test_mutation_metadata_cannot_change_runtime_decision(self):
        c=load_policy();rows=[dict(upper='D',count='hs')];a=evidence(rows,STATUS,c)['analysis']
        b=dict(image_id='synthetic',source_parent_id='p',image_sha256='test',interpreted_transcription=rows,decision=base(),interpreted_errors=None)
        saved=dict(analysis=a,review_only_counterfactual='REVIEW_UNCERTAIN')
        x=ev.process_case(b,STATUS,saved,source(),c);y=ev.process_case(b,STATUS,saved,source('VALID'),c)
        self.assertEqual(x['R3'],y['R3']);self.assertEqual(x['findings'],y['findings'])
    def test_freeze_cannot_be_automatic(self):
        with self.assertRaises(ValueError):ev.prepare_freeze(types.SimpleNamespace(ack_development_reviewed=False,run_dir=Path('unused')))
    def test_heldout_split_has_no_entrypoint(self):
        import contextlib
        with patch.object(sys,'argv',['evaluate_v3.py','evaluate','--split','protected_test','--preflight']),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as e:ev.main()
        self.assertEqual(e.exception.code,2)
    def test_two_case_mock_evaluation_artifacts(self):
        from integration import digest,load_json
        config=load_policy();bases=[];raw={};saved={};sources={}
        for id,validity,rows in [('synthetic_valid','VALID',[dict(upper='D',lower='R',count='aa')]),('synthetic_invalid','INVALID',[dict(upper='D',count='hs')])]:
            available=copy.deepcopy(STATUS);analysis=evidence(rows,available,config)['analysis']
            b=dict(image_id=id,source_parent_id=id+'_p',image_sha256='fake',interpreted_transcription=rows,decision=base(),
                actual_outcome='ACCEPT',validity=validity,expected_outcome='ACCEPT' if validity=='VALID' else 'REJECT_INVALID',
                interpreted_errors=dict(characters=4,edits=0,exact=True) if validity=='VALID' else None,
                raw_errors=dict(characters=4,edits=0,exact=True) if validity=='VALID' else None,
                correction_assessments=[],corrections=[],triggered_reason_codes=['BASE'],raw_decision=dict(reason_codes=['BASE']),
                interpreted_syntax=dict(outcome='ACCEPT'),preprocessing_failure=False,segmentation_failure=False,ocr_failure=False,pipeline_exception=None,
                invalidity_category='none' if validity=='VALID' else 'invalid_count_token',paired_mutation_category='invalid_count_token',variant='clean',length_group='short',row_group='single_row')
            bases.append(b);raw[id]=available;saved[id]=dict(analysis=analysis,review_only_counterfactual='ACCEPT' if validity=='VALID' else 'REVIEW_UNCERTAIN')
            sources[id]=source(validity)
        ds=types.SimpleNamespace(by_id=sources,groups={b['source_parent_id']:[] for b in bases})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r2dir=root/'baseline';r2dir.mkdir();(r2dir/'results.json').write_text('{}')
            st=root/'stage1';st.mkdir();(st/'per_image.jsonl').write_text('synthetic-only\n')
            lineage=dict(r2_dir=str(r2dir),r2_results_sha256=digest(r2dir/'results.json'),stage1_dir=str(st),stage1_evidence_sha256=digest(st/'per_image.jsonl'),raw_sha256='fake',dataset_manifest_sha256='fake')
            with patch.object(ev,'load_sources',return_value=(ds,bases,raw,saved,lineage)),patch.object(ev,'geometry_index',return_value=({},dict(sources={},issues=[]))),patch.object(ev,'verify_r2'),patch.object(ev,'verify_dependencies'),patch.object(ev,'safe_output',side_effect=lambda p:Path(p)):
                ev.evaluate(types.SimpleNamespace(r2_dir=None,stage1_dir=None,preflight=False,output_dir=root/'run'))
                ev.verify_completed(root/'run')
            m=load_json(root/'run/metrics.json');done=load_json(root/'run/completion.json')
            self.assertTrue(done['complete']);self.assertEqual(done['images'],2);self.assertEqual(m['new_definitive_rejections'],0)
            self.assertEqual(len(m['Stage1_differences']),0);self.assertTrue((root/'run/Stage1_differences.csv').exists())


if __name__=='__main__':unittest.main()
