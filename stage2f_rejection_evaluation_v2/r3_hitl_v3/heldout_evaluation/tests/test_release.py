import unittest,tempfile,copy,json,sys,types
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import heldout_cli as cli
import release_io as io
from heldout_dataset import audit_ownership,select_records,safe_image_path,rendering_metadata,identity
from journal import Journal,read_completed
from verify_freeze import verify,sha,canonical


def fixture():
    rows=[];parents=[]
    for j,split in enumerate(('development_design','validation_calibration','protected_test')):
        parent='p'+str(j);parents.append(dict(source_parent_id=parent,semantic_key='semantic'+str(j),dataset_split=split))
        for i in range(4):rows.append(dict(image_id=parent+'_'+str(i),source_parent_id=parent,image_sha256=parent+'_hash'+str(i),dataset_split=split,
            validity_status='VALID' if i<2 else 'INVALID',augmentation_recipe=None if i%2==0 else {},image_file_path=parent+'/'+str(i)+'.png'))
    return rows,parents
EXPECTED={s:(4,1) for s in ('development_design','validation_calibration','protected_test')}


def observation(count='a'):
    rows=[[dict(row=0,role='upper',box=[0,0,20,10]),dict(row=0,role='count',box=[0,10,20,20])]]
    return dict(preprocessing_status='ok',segmentation_accepted=True,segmentation_reason='',segmentation_rows=rows,
                ocr_status='ok',decoded_rows=[dict(upper='D',count=count)],confidence={},
                diagnostics=dict(legal_alternatives=1,path=[[0,['upper','count']]],rejection=''),
                tiers=[dict(t,prediction='D' if t['role']=='upper' else count,signals={}) for t in rows[0]])


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.runtime=cli.frozen_runtime()
    def test_release_integrity(self):self.assertEqual(verify()['existing_snapshot_count'],11)
    def test_frozen_modules_only(self):
        sources=self.runtime.loaded_sources();self.assertIn('policy_v3',sources)
        self.assertEqual(sources['policy_v3'],'src\\policy_v3.py')
    def test_source_bytes_unchanged(self):
        m=self.runtime.manifest
        p=io.RELEASE/'src/policy_v3.py';self.assertEqual(sha(p),m['files']['src/policy_v3.py']['sha256'])
    def test_original_root_maps_to_snapshot(self):
        self.assertEqual(self.runtime.remap(r'C:\NAQSHKASH\models\crnn.py'),io.RELEASE/'dependencies/models/crnn.py')
    def test_original_external_rules_map_to_snapshot(self):
        self.assertEqual(self.runtime.remap(r'C:\FYP Dataset Generation\config.py'),io.RELEASE/'external/Talamat/config.py')
    def test_frozen_path_not_remapped_twice(self):self.assertEqual(self.runtime.remap(io.RELEASE/'src/policy_v3.py'),io.RELEASE/'src/policy_v3.py')
    def test_snapshot_tampering_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'x.txt').write_text('original');m=dict(policy_id='R3_HITL_V3',status='FINAL_FROZEN',files={'x.txt':{'sha256':sha(root/'x.txt')}})
            io.write_json(root/'release_manifest.json',m);(root/'RELEASE_SHA256.txt').write_text(sha(root/'release_manifest.json'))
            verify(root,adapter=False);(root/'x.txt').write_text('changed')
            with self.assertRaises(ValueError):verify(root,adapter=False)
    def test_release_manifest_tampering_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'release_manifest.json').write_text('{}');(root/'RELEASE_SHA256.txt').write_text('wrong')
            with self.assertRaises(ValueError):verify(root,adapter=False)
    def test_disjoint_fixture(self):self.assertEqual(audit_ownership(*fixture(),expected=EXPECTED)['protected_test']['images'],4)
    def test_cross_split_parent_blocked(self):
        rows,parents=fixture();rows[4]['source_parent_id']=rows[0]['source_parent_id']
        with self.assertRaises(ValueError):audit_ownership(rows,parents,EXPECTED)
    def test_cross_split_identical_image_blocked(self):
        rows,parents=fixture();rows[4]['image_sha256']=rows[0]['image_sha256']
        with self.assertRaises(ValueError):audit_ownership(rows,parents,EXPECTED)
    def test_duplicate_sample_blocked(self):
        rows,parents=fixture();rows[4]['image_id']=rows[0]['image_id']
        with self.assertRaises(ValueError):audit_ownership(rows,parents,EXPECTED)
    def test_cross_split_semantic_parent_blocked(self):
        rows,parents=fixture();parents[1]['semantic_key']=parents[0]['semantic_key']
        with self.assertRaises(ValueError):audit_ownership(rows,parents,EXPECTED)
    def test_incomplete_split_blocked(self):
        rows,parents=fixture()
        with self.assertRaises(ValueError):audit_ownership(rows[:-1],parents,EXPECTED)
    def test_development_selection_blocked(self):
        with self.assertRaises(ValueError):select_records(fixture()[0],'development_design')
    def test_no_implicit_split(self):
        with self.assertRaises(ValueError):select_records(fixture()[0],None)
    def test_selected_split_only(self):self.assertEqual({r['dataset_split'] for r in select_records(fixture()[0],'protected_test')},{'protected_test'})
    def test_cross_split_image_path_blocked(self):
        with self.assertRaises(ValueError):safe_image_path(Path.cwd(),fixture()[0][0],'protected_test')
    def test_traversal_blocked(self):
        r=fixture()[0][4];r['image_file_path']='../escape.png'
        with self.assertRaises(ValueError):safe_image_path(Path.cwd(),r,'validation_calibration')
    def test_rendering_metadata_strips_truth(self):
        row=dict(layout={'font_sha256':'font','font_size_px':48,'convention':'fixed','mutated_column':9},augmentation_recipe=None,validity_status='INVALID',mutation_parameters={'bad':True})
        result=rendering_metadata(row);self.assertNotIn('validity_status',result);self.assertNotIn('mutated_column',result['layout'])
    def test_no_authorization_blocks_before_data(self):
        with self.assertRaises(ValueError):cli.authorize(types.SimpleNamespace(split='validation_calibration',authorization=''))
    def test_development_token_not_valid_for_test(self):
        with self.assertRaises(ValueError):cli.authorize(types.SimpleNamespace(split='protected_test',authorization='I_AUTHORIZE_VALIDATION'))
    def test_validation_explicit_authorization(self):cli.authorize(types.SimpleNamespace(split='validation_calibration',authorization='I_AUTHORIZE_VALIDATION'))
    def test_protected_requires_validation_review(self):
        with self.assertRaises(ValueError):cli.authorize(types.SimpleNamespace(split='protected_test',authorization=cli.TOKENS['protected_test'],ack_validation_reviewed=False,validation_run=None))
    def test_protected_requires_completed_registered_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=types.SimpleNamespace(split='protected_test',authorization=cli.TOKENS['protected_test'],ack_validation_reviewed=True,validation_run=Path(tmp))
            with self.assertRaises((ValueError,FileNotFoundError)):cli.authorize(args)
    def test_protected_registration_positive(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(cli,'HERE',Path(tmp)),patch.object(cli,'check_completion',return_value={'split':'validation_calibration','records':600}):
            v=Path(tmp)/'validation';v.mkdir();(v/'completion.json').write_text('sealed')
            io.write_json(Path(tmp)/'state/validation_calibration/evaluation_completed.json',{'completion_sha256':sha(v/'completion.json')})
            cli.authorize(types.SimpleNamespace(split='protected_test',authorization=cli.TOKENS['protected_test'],ack_validation_reviewed=True,validation_run=v))
    def test_no_cli_split_default(self):
        with patch.object(sys,'argv',['heldout_cli.py','preflight']):
            with self.assertRaises(SystemExit):cli.main()
    def test_cli_inference_requires_authorization(self):
        with patch.object(sys,'argv',['heldout_cli.py','infer','--split','validation_calibration','--output-dir','unused']):
            with self.assertRaises(SystemExit):cli.main()
    def test_output_path_cannot_escape(self):
        with self.assertRaises(ValueError):io.output_path(Path(tempfile.gettempdir())/'outside')
    def test_split_cannot_be_claimed_twice(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(cli,'HERE',Path(tmp)),patch.object(cli,'output_path',side_effect=lambda p:Path(p)):
            args=types.SimpleNamespace(split='validation_calibration',output_dir=Path(tmp)/'raw',resume=False);plan={'release_sha256':'r','selection_sha256':'s'}
            cli.claim_inference(args,plan)
            with self.assertRaises(ValueError):cli.claim_inference(args,plan)
    def test_exact_claim_can_resume(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(cli,'HERE',Path(tmp)),patch.object(cli,'output_path',side_effect=lambda p:Path(p)):
            args=types.SimpleNamespace(split='validation_calibration',output_dir=Path(tmp)/'raw',resume=False);plan={'release_sha256':'r','selection_sha256':'s'}
            cli.claim_inference(args,plan);args.resume=True;self.assertEqual(cli.claim_inference(args,plan),args.output_dir)
    def test_resume_cannot_change_release(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(cli,'HERE',Path(tmp)),patch.object(cli,'output_path',side_effect=lambda p:Path(p)):
            args=types.SimpleNamespace(split='validation_calibration',output_dir=Path(tmp)/'raw',resume=False);plan={'release_sha256':'r','selection_sha256':'s'}
            cli.claim_inference(args,plan);args.resume=True;plan['release_sha256']='changed'
            with self.assertRaises(ValueError):cli.claim_inference(args,plan)
    def test_completed_split_blocks_new_inference(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(cli,'HERE',Path(tmp)),patch.object(cli,'output_path',side_effect=lambda p:Path(p)):
            io.write_json(Path(tmp)/'state/validation_calibration/evaluation_completed.json',{})
            with self.assertRaises(ValueError):cli.claim_inference(types.SimpleNamespace(split='validation_calibration',output_dir=Path(tmp)/'raw',resume=False),{})
    def test_journal_roundtrip_and_immutable_write(self):
        import journal
        with tempfile.TemporaryDirectory() as tmp,patch.object(journal,'output_path',side_effect=lambda p:Path(p)):
            store=Journal(Path(tmp)/'raw',{'release_sha256':'r'});ident={'image_id':'fixture','dataset_split':'validation_calibration'}
            store.append(ident,{'pixel_status':'fixture'},{},None);self.assertEqual(store.read(ident)['observation'],{'pixel_status':'fixture'})
            with self.assertRaises(FileExistsError):store.append(ident,{}, {},None)
    def test_journal_tampering_blocked(self):
        import journal
        with tempfile.TemporaryDirectory() as tmp,patch.object(journal,'output_path',side_effect=lambda p:Path(p)):
            store=Journal(Path(tmp)/'raw',{});ident={'image_id':'fixture'};store.append(ident,{}, {},None)
            value=io.load(store.path('fixture'));value['observation']={'altered':True};store.path('fixture').write_text(json.dumps(value))
            with self.assertRaises(ValueError):store.read(ident)
    def test_development_cache_cannot_substitute_for_heldout(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(io,'check_completion',return_value={'split':'development_design','records':600,'parents':150}):
            io.write_json(Path(tmp)/'plan.json',{})
            with self.assertRaises(ValueError):read_completed(tmp,types.SimpleNamespace(split='validation_calibration'), 'release')
    def test_runtime_has_no_ground_truth_parameter(self):
        with self.assertRaises(TypeError):self.runtime.decide(observation(),{},validity='VALID')
    def test_runtime_ignores_unknown_observation_labels(self):
        raw=observation();a=self.runtime.decide(raw,{})
        raw['validity']='INVALID';raw['mutation_parameters']={'column':42}
        b=self.runtime.decide(raw,{});self.assertEqual(a['R3']['decision'],b['R3']['decision'])
    def test_r2_review_never_promoted(self):
        r=self.runtime.decide(observation('a a'),{})
        self.assertEqual(r['R2']['outcome'],'REVIEW_UNCERTAIN');self.assertEqual(r['R3']['decision'],r['R2'])
    def test_accept_not_carpet_ready(self):
        r=self.runtime.decide(observation(),{});self.assertEqual(r['R3']['decision']['outcome'],'ACCEPT');self.assertFalse(r['carpet_generation_ready']);self.assertIsNone(r['selected_carpet_reading'])
    def test_no_new_text_only_reject(self):
        r=self.runtime.decide(observation('hs'),{});self.assertEqual(r['R3']['decision']['outcome'],'REVIEW_UNCERTAIN')
    def test_geometry_hook_does_not_change_observe_return(self):
        import numpy as np
        pixels=np.full((20,20),255,np.uint8);rows=observation()['segmentation_rows']
        out=dict(matrix=np.eye(3),image=pixels,rows=rows);original=lambda image,config:out
        d1=types.SimpleNamespace(process_image=original)
        def obs(raw):
            self.assertIs(d1.process_image(pixels,{}),out);return {'segmentation_rows':rows,'unchanged':True}
        pipeline=types.SimpleNamespace(d1=d1,observe=obs);result,g=cli.observe(pipeline,b'fixture')
        self.assertEqual(result,{'segmentation_rows':rows,'unchanged':True});self.assertTrue(g['verified']);self.assertIs(d1.process_image,original)
    def test_geometry_hook_restores_after_exception(self):
        d1=types.SimpleNamespace(process_image=lambda a,b:None);original=d1.process_image
        def fail(raw):raise RuntimeError('synthetic')
        with self.assertRaises(RuntimeError):cli.observe(types.SimpleNamespace(d1=d1,observe=fail),b'fixture')
        self.assertIs(d1.process_image,original)
    def test_geometry_mismatch_unavailable(self):
        import numpy as np
        d1=types.SimpleNamespace(process_image=lambda a,b:dict(matrix=np.eye(3),image=a,rows=[]))
        def obs(raw):d1.process_image(np.full((5,5),255,np.uint8),{});return {'segmentation_rows':[[{'different':True}]]}
        self.assertIsNone(cli.observe(types.SimpleNamespace(d1=d1,observe=obs),b'fixture')[1])
    def test_software_mismatch_blocks_inference(self):
        import importlib
        helper=importlib.import_module('run_ocr_baseline')
        with patch.object(helper,'execution_environment',return_value={'different':'software'}):
            with self.assertRaises(ValueError):cli.environment(self.runtime)
    def test_offline_source_labels_cannot_change_final_decisions(self):
        from heldout_dataset import identity
        rows=io.load(io.RELEASE/self.runtime.manifest['dataset']['manifest_snapshot'])
        row=copy.deepcopy(next(r for r in rows if r['dataset_split']=='development_design' and r['validity_status']=='VALID'))
        item=dict(identity=identity(row),observation=observation(),rendering_context={},geometry=None)
        first,base=cli.process_record(self.runtime,item,row,'missing_count')
        row['validity_status']='INVALID';row['invalidity_category']='wrong_tier';row['expected_system_outcome']='REJECT_INVALID'
        second,_=cli.process_record(self.runtime,item,row,'wrong_tier')
        self.assertEqual(first['R2'],second['R2']);self.assertEqual(first['R3'],second['R3']);self.assertEqual(first['review_regions'],second['review_regions'])
        self.assertIsNone(second['localization'].get('precise_pixel_position'))
    def test_offline_metrics_reuse_three_outcome_summary(self):
        import importlib
        rows=io.load(io.RELEASE/self.runtime.manifest['dataset']['manifest_snapshot']);row=next(r for r in rows if r['dataset_split']=='development_design' and r['validity_status']=='VALID')
        item=dict(identity=identity(row),observation=observation(),rendering_context={},geometry=None)
        record,base=cli.process_record(self.runtime,item,row,'missing_count');summary,parents,groups=importlib.import_module('evaluation_metrics').grouped([base])
        self.assertEqual(summary['three_way_confusion']['VALID']['ACCEPT'],1);self.assertEqual(summary['three_way_confusion']['VALID']['REJECT_INVALID'],0)
        self.assertIn(row['source_parent_id'],parents);self.assertIn('invalidity_category',groups)
    def test_journal_completion_is_sealed_and_idempotent(self):
        import journal
        with tempfile.TemporaryDirectory() as tmp,patch.object(journal,'output_path',side_effect=lambda p:Path(p)):
            store=Journal(Path(tmp)/'raw',{'release_sha256':'fixture_release'})
            row=dict(image_id='fixture',image_sha256='fixture_sha',source_parent_id='p',original_valid_parent_id='p',mutation_family_id='p',dataset_split='validation_calibration',source_lineage={})
            dataset=types.SimpleNamespace(rows=[row],split='validation_calibration',groups={'p':[row]})
            store.append(identity(row),{}, {},None);done=store.finish(dataset)
            self.assertEqual(done,store.finish(dataset));self.assertTrue(done['complete'])
            for name,h in done['outputs'].items():self.assertEqual(sha(store.directory/name),h)
            (store.directory/'observations.jsonl').write_text('changed')
            with self.assertRaises(ValueError):store.finish(dataset)


if __name__=='__main__':unittest.main()
