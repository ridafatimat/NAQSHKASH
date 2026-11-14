"""Focused validation-only regression tests; no model inference or test images."""
import sys,unittest,importlib.util,json,csv,hashlib
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('baseline',ROOT/'evaluate_baseline.py');baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
import context
import pipeline_d1
geo=pipeline_d1.geo
HOUGH=cv2.HoughLinesP
def wrapped(shape):
    def f(*args,**kwargs):
        result=HOUGH(*args,**kwargs)
        return None if result is None else np.asarray(result).reshape(shape)
    return f
def old_correct():
    text=Path(geo.__file__).read_text().replace('for x0,y0,x1,y1 in np.asarray(lines).reshape(-1,4):','for x0,y0,x1,y1 in lines[:,0]:')
    namespace=dict(geo.__dict__);exec(compile(text,'historical_geometry_in_memory','exec'),namespace)
    return namespace['correct']
class Regression(unittest.TestCase):
    def test_hough_old_and_new_return_shapes(self):
        im=np.full((240,600),255,np.uint8)
        for y in range(30,220,30):cv2.line(im,(20,y),(580,y),0,2)
        with patch.object(geo.cv2,'HoughLinesP',wrapped((-1,1,4))):a=geo.correct(im)
        with patch.object(geo.cv2,'HoughLinesP',wrapped((-1,4))):b=geo.correct(im)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_allclose(a[1],b[1]);self.assertEqual(a[2],b[2])
        # Demonstrate the historical bug on the same deterministic input.
        with patch.object(geo.cv2,'HoughLinesP',wrapped((-1,4))):
            with self.assertRaises(TypeError):old_correct()(im)
    def test_no_hough_lines_safe_fallback(self):
        im=np.full((120,240),255,np.uint8)
        with patch.object(geo.cv2,'HoughLinesP',return_value=None):result=geo.correct(im)
        self.assertFalse(result[2]['perspective_applied'])
    def test_historical_clean_preprocessing_unchanged(self):
        paths=[Path(r'C:\FYP Dataset Generation\datasets\talim_real_capture_pilot_source_pack\selected_sources')/(s+'.png') for s in ['all_003191','all_009197']]
        manifest=baseline.PROJECT/'stage2/stage2e_finetuning/manifests/evaluation.csv'
        with manifest.open(encoding='utf-8-sig') as f:
            positive=next(r for r in csv.DictReader(f) if r['cohort']=='positive_development' and r['kind']=='clean')
        paths.append(Path(positive['path']))
        old=old_correct();config=json.loads((baseline.PROJECT/'stage2/stage2d_experiments/configs/D1.json').read_text())
        for path in paths:
            im=cv2.imread(str(path),0);self.assertIsNotNone(im,str(path))
            with patch.object(geo.cv2,'HoughLinesP',wrapped((-1,1,4))):
                after=pipeline_d1.process_image(im,config)
                with patch.object(geo,'correct',old):before=pipeline_d1.process_image(im,config)
            np.testing.assert_array_equal(before['image'],after['image']);np.testing.assert_allclose(before['matrix'],after['matrix'])
            for k in ['rows','accepted','reason','diagnostics']:self.assertEqual(before[k],after[k],str(path))
    def test_image_only_segmentation_and_blank(self):
        im=np.full((300,600,3),245,np.uint8)
        for y in [90,220]:
            for x in [80,200,320,440]:cv2.circle(im,(x,y),20,(25,25,25),2)
            cv2.line(im,(80,y-45),(100,y-45),(25,25,25),2)
            cv2.line(im,(200,y+35),(220,y+35),(25,25,25),2)
        tiers,mask,diag=baseline.handwritten_candidates(im)
        self.assertEqual(len([t for t in tiers if t['role']=='count']),2)
        for t in tiers:
            x0,y0,x1,y1=t['box'];self.assertTrue(0<=x0<x1<=600 and 0<=y0<y1<=300)
        self.assertEqual(baseline.handwritten_candidates(np.full_like(im,245))[0],[])
        # API takes only pixels: no label/token-count inputs are possible.
        import inspect
        self.assertEqual(list(inspect.signature(baseline.handwritten_candidates).parameters),['bgr'])
    def test_no_unverified_scoring_or_silent_page_omission(self):
        m=json.loads((ROOT/'metrics.json').read_text());self.assertEqual(m['source_photographs_attempted'],3);self.assertIsNone(m['cer']);self.assertIsNone(m['exact_tier_accuracy'])
        with (ROOT/'predictions.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        self.assertEqual({r['source_id'] for r in rows},set(baseline.IDS));self.assertTrue(all(r['scored']=='False' for r in rows))
        self.assertEqual(m['successfully_inferred_candidate_tiers'],sum(not r['inference_error'] for r in rows))
        checkpoint=Path(m['checkpoint']);self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),m['checkpoint_sha256'])
        self.assertEqual(m['expected_nonempty_tiers'],14)
if __name__=='__main__':unittest.main(verbosity=2)
