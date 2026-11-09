import sys,unittest,tempfile,types,copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE/'src'))
from alignment import *
from experiment import Capture,materialize,gallery,validate_selection,asset_path
import experiment


class FakeModel:
    def register_forward_hook(self,fn):
        self.fn=fn
        return types.SimpleNamespace(remove=lambda:setattr(self,'fn',None))
    def emit(self):
        # Synthetic tensors, not a neural model forward.
        a=np.zeros((32,1,66),np.float32);a[:,:,0]=1;a[2:4,0,8]=3
        self.fn(self,(np.zeros((1,1,64,128)),),a)


class FakePipeline:
    def __init__(self):
        self.model=FakeModel();self.args=[];self.fail=False
        self.rows=[[dict(row=i,role=role,box=[10,120*i+35*j,80,120*i+35*j+35]) for j,role in enumerate(('upper','count','lower'))] for i in range(3)]
        def process(image,config):
            return dict(image=image,matrix=np.eye(3),accepted=True,reason='',rows=self.rows,diagnostics={})
        self.d1=types.SimpleNamespace(process_image=process)
    def observe(self,raw):
        self.args.append(raw);out=self.d1.process_image(np.full((400,200),255,np.uint8),{})
        if self.fail:raise RuntimeError('synthetic failure')
        tiers=[]
        for row in out['rows']:
            for t in row:self.model.emit();tiers.append(dict(role=t['role'],box=t['box'],prediction='A'))
        return dict(preprocessing_status='ok',segmentation_accepted=out['accepted'],ocr_status='ok',tiers=tiers,decoded_rows=[{t['role']:'A' for t in tiers}])


def record():
    return dict(layout=dict(padding=24,extra_left=1,cell_width=50,line_height=50,row_gap=20),
        raw_display=[dict(upper='D F',count_line='baa',lower='   ')])


def perfect_capture(r,font):
    refs=reference_glyphs(r,0,font);tiers=[]
    for role,gs in refs.items():
        chars=[dict(character=g['character'],start_frame=i,end_frame=i+1,mapping=dict(source_center=[g['x'],g['box'][1]])) for i,g in enumerate(gs)]
        tiers.append(dict(role=role,characters=chars))
    return dict(tiers=tiers,reproduction=dict(all_match=True))


class Tests(unittest.TestCase):
    def test_blank_space_and_repeated_characters(self):
        runs,chars=path_runs([0,8,8,0,8,1,1,0,9],{0:'<BLANK>',1:' ',8:'A',9:'B'})
        self.assertEqual(''.join(c['character'] for c in chars),'AA B')
        self.assertEqual([(c['start_frame'],c['end_frame']) for c in chars],[(1,3),(4,5),(5,7),(8,9)])
        self.assertIsNone(runs[0]['character'])
    def test_identity_mapping(self):
        m=map_span(dict(start_frame=1,end_frame=2),[10,20,110,70],100,np.eye(3))
        self.assertEqual(m['source_center'],[15.5,44.5]);self.assertEqual(m['lattice']['x_interval'],[13.5,17.5])
    def test_actual_rounding_ratio(self):
        m=map_span(dict(start_frame=1,end_frame=2),[10,20,111,70],129,np.eye(3))
        self.assertAlmostEqual(m['source_center'][0],10+6*101/129-.5)
    def test_projective_inverse(self):
        M=np.array([[1.,.1,8],[.05,1,3],[.001,0,1.]])
        m=map_span(dict(start_frame=3,end_frame=5),[10,20,110,70],128,M)
        np.testing.assert_allclose(project([m['source_center']],M)[0],m['processed_center'])
        self.assertEqual(len(m['lattice']['polygon']),4)
    def test_singular_transform(self):
        with self.assertRaises(np.linalg.LinAlgError):map_span(dict(start_frame=0,end_frame=1),[0,0,20,20],64,np.zeros((3,3)))
    def test_all_optimal_correspondences(self):
        self.assertTrue(all(c['status']=='ambiguous' for c in unique_correspondences('aa','aaa')))
        self.assertEqual(unique_correspondences('X','B')[0]['reference_index'],0)
        self.assertIsNone(unique_correspondences('D','_')[0]['reference_index'])
    def test_geometry_only_association(self):
        a=associate([dict(x=0),dict(x=100)],[dict(x=0),dict(x=30),dict(x=100)])
        self.assertEqual(a['groups'],[[0,1],[2]])
    def test_ambiguous_and_missing_tokens(self):
        a=associate([dict(x=0),dict(x=50)],[dict(x=0),dict(x=25),dict(x=75)])
        self.assertEqual(a['status'],'ambiguous_partition')
        self.assertEqual(associate([dict(x=0),dict(x=100)],[dict(x=10)])['status'],'insufficient_visible_glyphs')
    def test_capture_filters_one_row_and_restores_original(self):
        p=FakePipeline();original=p.d1.process_image
        result=Capture(p).observe(b'pixel-bytes',1)
        self.assertEqual(len(result['calls']),3);self.assertEqual(len(result['full_segmentation_rows']),3)
        self.assertEqual(p.args,[b'pixel-bytes']);self.assertIs(p.d1.process_image,original);self.assertIsNone(p.model.fn)
    def test_hook_cleanup_on_error(self):
        p=FakePipeline();p.fail=True;original=p.d1.process_image
        with self.assertRaises(RuntimeError):Capture(p).observe(b'pixels',0)
        self.assertIs(p.d1.process_image,original);self.assertIsNone(p.model.fn)
    def test_materialize_mismatch_and_gallery(self):
        p=FakePipeline();captured=Capture(p).observe(b'pixels',1)
        saved=dict(segmentation_rows=p.rows,decoded_rows=[{},dict(upper='B',count='A',lower='A')])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);o=materialize(captured,1,saved,root/'assets',{0:'<BLANK>',8:'A'})
            self.assertFalse(o['reproduction']['all_match']);self.assertEqual(o['reproduction']['mismatched_roles'],['upper'])
            for t in o['tiers']:
                self.assertEqual(experiment.digest(root/'assets'/t['tensor_file']),t['tensor_sha256'])
                with np.load(root/'assets'/t['tensor_file'],allow_pickle=False) as data:
                    self.assertEqual(data['logits'].dtype,np.float32);self.assertEqual(data['logits'].shape,(32,66))
                    np.testing.assert_array_equal(data['argmax_path'],data['logits'].argmax(-1))
            o['asset_directory']='assets';case=dict(image_id='synthetic_TEST_ONLY',variant='clean',validity='VALID',capture=o,score=dict(reproduced=False,reference={r:[] for r in ('upper','count','lower')}))
            gallery([case],root,root)
            self.assertTrue((root/'gallery/index.html').is_file());self.assertEqual(len(list((root/'assets').glob('*.npz'))),3)
    def test_source_reference_uses_mutated_display_and_score(self):
        from PIL import ImageFont
        font=ImageFont.truetype(r'C:\FYP Dataset Generation\Assets\Talamat.ttf',48);r=record()
        capture=perfect_capture(r,font);before=copy.deepcopy(capture)
        a=score_image(capture,r,0,font,0)
        self.assertEqual(a['metrics']['matched_glyphs'],5);self.assertEqual(a['metrics']['association_correct'],3)
        self.assertEqual(a['metrics']['multi_glyph_token_correct'],1);self.assertFalse(a['metrics']['false_association_risk'])
        self.assertEqual(a['focus']['status'],'correct')
        r['original_structured_notation']={'hidden':'poison'}
        self.assertEqual(score_image(capture,r,0,font,0),a);self.assertEqual(capture,before)
        r['raw_display'][0]['count_line']='  a'
        b=score_image(perfect_capture(r,font),r,0,font,0)
        self.assertEqual(b['metrics']['reference_empty_tokens'],1);self.assertNotEqual(b['association']['status'],'unique')
        self.assertEqual(b['focus']['status'],'no_visible_count_to_align')
    def test_reproduction_gate_and_coverage_denominators(self):
        from PIL import ImageFont
        font=ImageFont.truetype(r'C:\FYP Dataset Generation\Assets\Talamat.ttf',48);r=record();capture=perfect_capture(r,font);capture['reproduction']['all_match']=False
        score=score_image(capture,r,0,font);metrics=summarize([dict(score=score,parent_id='p',validity='VALID')])
        self.assertEqual(metrics['alignment_coverage'],0);self.assertEqual(metrics['association_scoring_coverage'],0)
        self.assertEqual(metrics['valid_control_incomplete_evidence_images'],1);self.assertIsNone(metrics['position_error_pixels'])
    def test_configuration_exact_development_quartets(self):
        from dataset_loader import DevelopmentDataset
        from integration import load_json
        config=load_json(HERE/'configs/selection.json');ds=DevelopmentDataset(verify_files=False)
        self.assertEqual(len(validate_selection(config,ds)),48)
        bad=copy.deepcopy(config);bad['split']='protected_test'
        with self.assertRaises(ValueError):validate_selection(bad,ds)
        bad=copy.deepcopy(config);bad['parents'][0]['images'].pop()
        with self.assertRaises(ValueError):validate_selection(bad,ds)
    def test_cached_asset_path_escape(self):
        with self.assertRaises(ValueError):asset_path(Path('C:/pilot'),Path('../protected.png'))
    def test_mock_cached_evaluation_writes_report_without_a_model(self):
        # Four tiny synthetic variants, not the real 48-image inference job.
        from integration import digest,load_json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'cache';run.mkdir();(run/'observations.jsonl').write_text('synthetic-only\n')
            p=FakePipeline();observed=materialize(Capture(p).observe(b'synthetic',0),0,dict(segmentation_rows=p.rows,decoded_rows=[dict(upper='A',count='A',lower='A')]),run/'assets',{0:'<BLANK>',8:'A'})
            observed['asset_directory']='assets';entries=[];items=[]
            font_path=Path(r'C:\FYP Dataset Generation\Assets\Talamat.ttf')
            for validity in ('VALID','INVALID'):
                for illuminated in (False,True):
                    r=record();r.update(image_id=validity+str(illuminated),validity_status=validity,augmentation_recipe={} if illuminated else None)
                    if illuminated:r['augmentation_recipe']={'synthetic_test_only':True}
                    r['layout'].update(font_sha256=digest(font_path),font_size_px=48)
                    entries.append((r,dict(parent_id='synthetic_parent',category='missing_count',affected_row=0,focus_column=0)))
                    items.append(dict(observation=copy.deepcopy(observed)))
            with patch.object(experiment,'load_inputs',return_value=(dict(font_path=str(font_path)),None,entries,{})),patch.object(experiment,'load_extraction',return_value=items),patch.object(experiment,'inside',side_effect=lambda p:Path(p)),patch.object(experiment,'check_lock'):
                experiment.evaluate(types.SimpleNamespace(run_dir=run,output_dir=root/'evaluated'))
            result=load_json(root/'evaluated/metrics.json')
            self.assertEqual(result['overall']['images'],4);self.assertEqual(len(result['paired_illumination']),2)
            self.assertTrue((root/'evaluated/report.md').exists());self.assertTrue((root/'evaluated/gallery/index.html').exists())


if __name__=='__main__':unittest.main()
