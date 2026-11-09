"""Tiny fixtures/mocks only. No CRNN constructor, forward pass or real dataset pixels."""
import copy
import tempfile
import unittest
from unittest.mock import patch
from integration import *
from dataset_loader import DevelopmentDataset,select_records,SPLIT,image_bytes
from observation_store import ObservationStore,load_raw_run
from run_ocr_baseline import infer_pending
from evaluate_rejection import evaluate_case,evaluate_records,load_candidate
from evaluation_metrics import summarize,grouped
from compare_rejection import compare
from generate_error_gallery import failure_groups,make_gallery


def observation(upper='A',count='ja'):
    roles=['upper','count'];decoded=[dict(upper=upper,count=count)]
    segments=[[dict(row=0,role=role,box=[0,i*10,20,i*10+8]) for i,role in enumerate(roles)]]
    return dict(preprocessing_status='ok',segmentation_accepted=True,segmentation_reason='',ocr_status='ok',
        decoded_rows=decoded,segmentation_rows=segments,tiers=[dict(**t,prediction=decoded[0][t['role']]) for t in segments[0]],
        diagnostics=dict(legal_alternatives=2,role_margin=1.,path=[[0,roles]],rejection=''),confidence={})


def fixture(directory):
    """Synthetic metadata is a mock contract, not generated/verified Talim data."""
    from PIL import Image
    directory=Path(directory);directory.mkdir()
    (directory/'annotations').mkdir();(directory/'images').mkdir()
    original=dict(sample_id='unit_parent',logical_rows=[dict(runs=[dict(symbol='A',count=11)])])
    key=rules.semantic_key(original);pid='v2p_'+key[:24]
    lineage=dict(historical_parent_reused=False,exact_semantic_overlap_found=False,historical_overlap_coverage='full',known_historical_semantic_novelty_verified=True)
    profile=load_json(T1/'configs/rendering_profile.json')
    layout=dict(font_sha256=profile['font_sha256'],font_size_px=48,convention='existing Talamat grayscale integer lt grid; matched parent layout is frozen')
    rows=[]
    for i,(status,variant) in enumerate([('VALID',False),('VALID',True),('INVALID',False),('INVALID',True)]):
        id='unit_'+str(i);path=directory/'images'/(id+'.png')
        im=Image.new('L',(4,4),255);im.putpixel((i,0),0);im.save(path)
        recipe=dict(padding=0,operations=[dict(type='brightness',factor=.96)],output_format='PNG') if variant else None
        row=dict(image_id=id,image_file_path='images/'+id+'.png',image_sha256=digest(path),source_parent_id=pid,original_valid_parent_id='unit_parent',
            mutation_family_id=pid,validity_status=status,expected_system_outcome='ACCEPT' if status=='VALID' else 'REJECT_INVALID',
            invalidity_category='none' if status=='VALID' else 'wrong_tier',canonical_rule_id='CANONICAL_VALID' if status=='VALID' else 'CANON_SYMBOL_DIRECTION',
            verification_status='AUTO_VERIFIED',verification_evidence=dict(status='AUTO_VERIFIED',classification=status),
            original_structured_notation=original,raw_display=[dict(upper='A' if status=='VALID' else '',count_line='ja',lower='' if status=='VALID' else 'A')],
            layout=layout,original_image_hash='',matched_valid_control_id='unit_0',augmentation_recipe=recipe,dataset_split=SPLIT,
            source_lineage=lineage,ctc_transcription=rules.canonical_text(original) if status=='VALID' else None,semantic_validity=status,ambiguity_or_exclusion_reason='')
        rows.append(row)
    for row in rows:
        row['original_image_hash']=rows[0]['image_sha256'];write_json(directory/'annotations'/(row['image_id']+'.json'),row)
    # Excluded split has no image or annotation file. Access would fail loudly.
    hidden=dict(image_id='protected',source_parent_id='protected_parent',dataset_split='protected_test',image_sha256='private_hash',image_file_path='DO_NOT_ACCESS.png')
    all_rows=rows+[hidden]
    write_json(directory/'manifest.json',all_rows)
    write_json(directory/'parents.json',[dict(source_parent_id=pid,semantic_key=key,dataset_split=SPLIT)])
    history=directory/'history.json';write_json(history,dict(full_historical_coverage=True,errors=[],semantic_keys=[]))
    write_json(directory/'completion.json',dict(complete=True,mode='full',history_coverage='full',manifest_sha256=digest(directory/'manifest.json'),history_index_sha256=digest(history)))
    spec=dict(dataset_dir=str(directory),metadata_hashes={name:digest(directory/name) for name in ('manifest.json','parents.json','completion.json')},
              history_index=str(history),total_records=5,development_records=4,development_parents=1)
    return spec,rows


def plan(dataset):
    return dict(dataset_manifest_sha256=dataset.manifest_sha256,selection_sha256=dataset.selection_sha256,split=SPLIT,
                checkpoint_sha256=CHECKPOINT_SHA,unit_test=True)


class LoaderTests(unittest.TestCase):
    def test_exact_development_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            spec,rows=fixture(Path(temp)/'data');data=DevelopmentDataset(spec=spec,verify_files=True)
            self.assertEqual(len(data.rows),4);self.assertEqual(set(data.by_id),{r['image_id'] for r in rows})
            self.assertFalse((data.directory/'DO_NOT_ACCESS.png').exists())

    def test_protected_and_validation_block_before_file_access(self):
        for split in ('protected_test','validation_calibration','all'):
            with self.assertRaisesRegex(ValueError,'Only'):DevelopmentDataset('/does/not/exist',split)

    def test_quarantined_dataset_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            spec,_=fixture(Path(temp)/'data');write_json(Path(spec['dataset_dir'])/'QUARANTINED.json',{})
            with self.assertRaisesRegex(ValueError,'Quarantined'):DevelopmentDataset(spec=spec)

    def test_parent_and_lineage_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            spec,rows=fixture(Path(temp)/'data');data=DevelopmentDataset(spec=spec)
            self.assertEqual(data.identity(rows[0])['source_lineage'],rows[0]['source_lineage'])
            self.assertEqual(data.identity(rows[0])['source_parent_id'],rows[0]['source_parent_id'])

    def test_image_hash_drift_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            spec,rows=fixture(Path(temp)/'data');(Path(spec['dataset_dir'])/rows[0]['image_file_path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash'):DevelopmentDataset(spec=spec,verify_files=True)

    def test_fabricated_invalid_ctc_target_forbidden(self):
        with tempfile.TemporaryDirectory() as temp:
            spec,rows=fixture(Path(temp)/'data');p=Path(spec['dataset_dir'])/'manifest.json';value=load_json(p)
            value[2]['ctc_transcription']='fake';p.write_text(json.dumps(value))
            completion=load_json(Path(spec['dataset_dir'])/'completion.json');completion['manifest_sha256']=digest(p)
            (Path(spec['dataset_dir'])/'completion.json').write_text(json.dumps(completion))
            spec['metadata_hashes']={k:digest(Path(spec['dataset_dir'])/k) for k in spec['metadata_hashes']}
            with self.assertRaisesRegex(ValueError,'fabricated'):DevelopmentDataset(spec=spec)

    def test_cross_split_duplicate_hash_rejected(self):
        rows=[dict(image_id='a',source_parent_id='p',dataset_split=SPLIT,image_sha256='same'),
              dict(image_id='b',source_parent_id='q',dataset_split='protected_test',image_sha256='same')]
        with self.assertRaisesRegex(ValueError,'cross'):select_records(rows)

    def test_image_access_guard_does_not_read_other_splits(self):
        with patch.object(Path,'read_bytes',side_effect=AssertionError('must not read')):
            with self.assertRaises(ValueError):image_bytes(Path('unused'),dict(dataset_split='protected_test'))

    def test_manifest_hash_drift_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            spec,_=fixture(Path(temp)/'data');p=Path(spec['dataset_dir'])/'manifest.json';p.write_text('[]')
            with self.assertRaisesRegex(ValueError,'metadata drift'):DevelopmentDataset(spec=spec)


class JournalTests(unittest.TestCase):
    def test_resumable_one_pass_and_identical_candidate_ids(self):
        class MockPipeline:
            def __init__(self):self.calls=[]
            def observe(self,raw):self.calls.append(raw);return observation()
        with tempfile.TemporaryDirectory() as temp:
            spec,_=fixture(Path(temp)/'data');data=DevelopmentDataset(spec=spec);p=plan(data)
            first=ObservationStore(Path(temp)/'run',p);mock=MockPipeline()
            partial=infer_pending(data,first,mock,max_new_records=2,progress_every=0)
            self.assertFalse(partial['complete']);self.assertEqual(len(mock.calls),2)
            self.assertTrue(all(isinstance(value,bytes) for value in mock.calls))
            resumed=ObservationStore(Path(temp)/'run',p,resume=True)
            done=infer_pending(data,resumed,mock,progress_every=0)
            self.assertTrue(done['complete']);self.assertEqual(len(mock.calls),4)
            raw,lineage=load_raw_run(resumed.directory,data)
            self.assertEqual(len(raw),4)
            r0=evaluate_records(raw,data,'R0');r1=evaluate_records(raw,data,'R1')
            self.assertEqual([r['image_id'] for r in r0],[r['image_id'] for r in r1])
            self.assertEqual([r['raw_OCR_transcription'] for r in r0],[r['raw_OCR_transcription'] for r in r1])
            self.assertEqual(len(mock.calls),4)

    def test_incomplete_raw_run_not_evaluable(self):
        with tempfile.TemporaryDirectory() as temp:
            spec,_=fixture(Path(temp)/'data');data=DevelopmentDataset(spec=spec);store=ObservationStore(Path(temp)/'run',plan(data))
            with self.assertRaises(FileNotFoundError):load_raw_run(store.directory,data)

    def test_observations_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            spec,rows=fixture(Path(temp)/'data');data=DevelopmentDataset(spec=spec);store=ObservationStore(Path(temp)/'run',plan(data))
            id=data.identity(rows[0]);store.append(id,observation(),{})
            with self.assertRaises(FileExistsError):store.append(id,observation(),{})

    def test_payload_tampering_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            spec,rows=fixture(Path(temp)/'data');data=DevelopmentDataset(spec=spec);store=ObservationStore(Path(temp)/'run',plan(data))
            id=data.identity(rows[0]);store.append(id,observation(),{})
            p=store.path(id['image_id']);item=load_json(p);item['observation']['decoded_rows']=[];p.write_text(json.dumps(item))
            with self.assertRaisesRegex(ValueError,'payload drift'):store.read(id)

    def test_resume_changed_lineage_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=dict(unit=1);ObservationStore(Path(temp)/'run',p)
            with self.assertRaises(ValueError):ObservationStore(Path(temp)/'run',dict(unit=2),resume=True)

    def test_completed_candidate_artifacts_roundtrip(self):
        class MockPipeline:
            def observe(self,raw):return observation()
        with tempfile.TemporaryDirectory() as temp:
            spec,_=fixture(Path(temp)/'data');data=DevelopmentDataset(spec=spec);store=ObservationStore(Path(temp)/'raw',plan(data))
            infer_pending(data,store,MockPipeline(),progress_every=0);raw,source=load_raw_run(store.directory,data)
            lineage=dict(dataset_manifest_sha256=data.manifest_sha256,selection_sha256=data.selection_sha256,
                checkpoint_sha256=CHECKPOINT_SHA,experiment_lock_sha256=digest(ROOT/'configs/experiment_lock.json'),
                raw_observations_sha256=source['completion']['raw_sha256'])
            loaded=[]
            for candidate in ('R0','R1'):
                out=Path(temp)/candidate;out.mkdir()
                write_json(out/'results.json',dict(candidate=candidate,lineage=lineage,rows=evaluate_records(raw,data,candidate)))
                write_json(out/'completion.json',dict(candidate=candidate,complete=True,records=4,results_sha256=digest(out/'results.json')))
                loaded.append(load_candidate(out,data))
            result=compare(*loaded)
            self.assertEqual(result['R0']['images'],4)
            self.assertEqual(result['R1']['invalid_accepted_rate'],1)

    def test_runtime_failure_is_recorded_and_not_definitive_invalidity(self):
        class BadPipeline:
            def observe(self,raw):raise RuntimeError('unit mock failure')
        with tempfile.TemporaryDirectory() as temp:
            spec,_=fixture(Path(temp)/'data');data=DevelopmentDataset(spec=spec);store=ObservationStore(Path(temp)/'run',plan(data))
            infer_pending(data,store,BadPipeline(),progress_every=0)
            raw,_=load_raw_run(store.directory,data);rows=evaluate_records(raw,data,'R0')
            self.assertTrue(all(r['actual_outcome']=='REVIEW_UNCERTAIN' for r in rows))
            self.assertTrue(all(r['pipeline_exception'] for r in rows))
            metrics=summarize(rows)
            self.assertEqual(metrics['pipeline_exceptions'],4)
            self.assertEqual(metrics['D1_segmentation_failures'],0)
            self.assertEqual(metrics['OCR_decoding_failures'],0)


class RuntimeAndMetricsTests(unittest.TestCase):
    def row(self,validity='VALID',upper='A',candidate='R0'):
        with tempfile.TemporaryDirectory() as temp:
            spec,rows=fixture(Path(temp)/'data');row=copy.deepcopy(next(r for r in rows if r['validity_status']==validity))
            return evaluate_case(dict(observation=observation(upper),rendering_context=runtime_rendering_context(row)),row,candidate,'wrong_tier')

    def test_three_way_confusion_and_critical_denominators(self):
        a=self.row('INVALID');b=copy.deepcopy(a);b['actual_outcome']='REVIEW_UNCERTAIN';c=copy.deepcopy(a);c['actual_outcome']='REJECT_INVALID'
        d=self.row('VALID');e=copy.deepcopy(d);e['actual_outcome']='REJECT_INVALID'
        m=summarize([a,b,c,d,e])
        self.assertEqual(m['invalid_denominator'],3);self.assertEqual(m['valid_denominator'],2)
        self.assertAlmostEqual(m['true_invalid_rejection_rate'],1/3)
        self.assertAlmostEqual(m['invalid_accepted_rate'],1/3)
        self.assertAlmostEqual(m['invalid_review_rate'],1/3)
        self.assertEqual(m['false_definitive_rejection_rate'],.5)
        self.assertEqual(m['three_way_confusion']['INVALID']['REVIEW_UNCERTAIN'],1)

    def test_review_is_not_rejection(self):
        row=self.row('INVALID');row['actual_outcome']='REVIEW_UNCERTAIN'
        m=summarize([row]);self.assertEqual(m['true_invalid_rejection_rate'],0);self.assertEqual(m['invalid_review_rate'],1)

    def test_empty_denominators_are_null(self):
        result=summarize([])
        self.assertIsNone(result['true_invalid_rejection_rate']);self.assertIsNone(result['valid_interpreted_CER'])

    def test_explicit_preprocessing_and_ocr_failures_are_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            _,rows=fixture(Path(temp)/'data')
            before=observation();before.update(segmentation_accepted=False,segmentation_reason='preprocessing:unit error',ocr_status='not_run',decoded_rows=[])
            a=evaluate_case(dict(observation=before,rendering_context={}),rows[0],'R0','wrong_tier')
            self.assertTrue(a['preprocessing_failure']);self.assertFalse(a['segmentation_failure']);self.assertFalse(a['ocr_failure'])
            before=observation();before.update(ocr_status='crop_too_narrow',decoded_rows=[])
            b=evaluate_case(dict(observation=before,rendering_context={}),rows[0],'R0','wrong_tier')
            self.assertFalse(b['preprocessing_failure']);self.assertTrue(b['ocr_failure'])

    def test_invalid_has_no_fabricated_cer_or_reference(self):
        row=self.row('INVALID')
        self.assertIsNone(row['raw_errors']);self.assertIsNone(row['interpreted_errors']);self.assertIsNone(row['valid_reference_transcription'])
        self.assertIsNone(summarize([row])['valid_interpreted_CER'])

    def test_accepted_output_correctness_is_not_acceptance(self):
        row=self.row('VALID',upper='B')
        self.assertEqual(row['actual_outcome'],'ACCEPT')
        self.assertEqual(summarize([row])['accepted_output_transcription_correctness'],0)

    def test_unknown_rendering_does_not_force_t1_correction(self):
        raw=observation('X');result=apply_candidate(raw,{},'R1')
        self.assertFalse(result['corrections']);self.assertEqual(result['final_decision']['outcome'],'REVIEW_UNCERTAIN')

    def test_t1_gate_and_raw_reason_preservation(self):
        profile=load_json(T1/'configs/rendering_profile.json')
        result=apply_candidate(observation('X'),profile,'R1')
        self.assertTrue(result['corrections']);self.assertEqual(result['interpreted_observation']['decoded_rows'][0]['upper'],'B')
        self.assertIn('DECODED_UNSUPPORTED_OR_WRONG_TIER_GLYPH',result['original_reason_codes'])

    def test_runtime_rendering_extractor_ignores_ground_truth(self):
        class Poison(dict):
            def get(self,key,default=None):
                if key not in ('layout','augmentation_recipe'):raise AssertionError('Ground truth read: '+key)
                return super().get(key,default)
        profile=load_json(T1/'configs/rendering_profile.json')
        row=Poison(layout=dict(font_sha256=profile['font_sha256'],font_size_px=48,
            convention='existing Talamat grayscale integer lt grid; matched parent layout is frozen'),augmentation_recipe=None)
        self.assertTrue(runtime_rendering_context(row))

    def test_validity_does_not_change_runtime_decision(self):
        with tempfile.TemporaryDirectory() as temp:
            _,rows=fixture(Path(temp)/'data');valid=rows[0];invalid=copy.deepcopy(valid)
            invalid.update(validity_status='INVALID',expected_system_outcome='REJECT_INVALID',ctc_transcription=None)
            raw=dict(observation=observation('X'),rendering_context=runtime_rendering_context(valid))
            a=evaluate_case(raw,valid,'R1','wrong_tier');b=evaluate_case(raw,invalid,'R1','wrong_tier')
            self.assertEqual(a['decision'],b['decision']);self.assertEqual(a['interpreted_transcription'],b['interpreted_transcription'])
            self.assertIsNotNone(a['interpreted_errors']);self.assertIsNone(b['interpreted_errors'])

    def test_invalid_t1_edits_are_unscorable_not_claimed_correct(self):
        row=self.row('INVALID',upper='X',candidate='R1')
        self.assertTrue(row['corrections']);self.assertEqual(row['correction_assessments'][0]['assessment'],'unscorable_invalid_no_CTC_reference')

    def test_parent_macro_and_worst_cases(self):
        a=self.row('INVALID');b=copy.deepcopy(a);b['source_parent_id']='other';b['actual_outcome']='REJECT_INVALID'
        c=copy.deepcopy(a);c['image_id']='other_copy'
        m,p,_=grouped([a,b,c])
        self.assertAlmostEqual(m['invalid_accepted_rate'],2/3)
        self.assertEqual(m['macro_parent']['invalid_accepted_rate'],.5)
        self.assertTrue(p[a['source_parent_id']]['worst_case']['any_invalid_accepted'])

    def test_pairing_requires_identical_raw_reuse_and_ids(self):
        row=self.row();a=dict(candidate='R0',lineage={'raw':'same'},rows=[row]);b=dict(candidate='R1',lineage={'raw':'same'},rows=[copy.deepcopy(row)])
        result=compare(a,b);self.assertIn('NOT_AUTOMATIC',result['selection_decision'])
        b['lineage']['raw']='different'
        with self.assertRaises(ValueError):compare(a,b)
        b['lineage']['raw']='same';b['rows'][0]['image_id']='different'
        with self.assertRaises(ValueError):compare(a,b)

    def test_failure_reporting_lists_overlap_without_merging_review(self):
        row=self.row('INVALID');review=copy.deepcopy(row);review['actual_outcome']='REVIEW_UNCERTAIN'
        rejected=self.row('VALID');rejected['actual_outcome']='REJECT_INVALID'
        groups=failure_groups([row,review,rejected])
        self.assertEqual(len(groups['invalid_accepted']),1);self.assertEqual(len(groups['invalid_review']),1);self.assertEqual(len(groups['valid_rejected']),1)
        self.assertFalse(groups['correctly_rejected_invalid'])

    def test_gallery_uses_only_declared_development_images(self):
        with tempfile.TemporaryDirectory() as temp:
            spec,rows=fixture(Path(temp)/'data');data=DevelopmentDataset(spec=spec)
            raw=dict(observation=observation(),rendering_context={});case=evaluate_case(raw,rows[2],'R0','wrong_tier')
            output=Path(temp)/'gallery.html';make_gallery(output,failure_groups([case]),data,1)
            self.assertIn('Evaluated input',output.read_text(encoding='utf-8'))
            poison=copy.deepcopy(case);poison['split']='protected_test'
            with self.assertRaises(ValueError):make_gallery(Path(temp)/'bad.html',{'bad':[poison]},data,1)

    def test_historical_artifact_integrity(self):
        self.assertTrue(verify_integrity()['unchanged'])


if __name__=='__main__':unittest.main()
