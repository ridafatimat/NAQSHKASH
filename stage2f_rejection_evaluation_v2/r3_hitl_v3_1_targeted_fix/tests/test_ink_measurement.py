import unittest,sys,copy
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ink_measurement import ink_groups,first_ink,estimate_pitch,measure_processed,D1Only
from paths_v3_1 import config


def pixels():
    a=np.full((30,60),255,np.uint8)
    for x in (5,15,25,35):a[12:17,x:x+3]=0
    a[2:7,5:8]=0;a[22:27,25:28]=0
    return a


def boxes():return [[dict(row=0,role='upper',box=[0,0,60,10]),dict(row=0,role='count',box=[0,10,60,20]),dict(row=0,role='lower',box=[0,20,60,30])]]
TEXT=[dict(upper='D',count='aaaa',lower='R')]


def measurement():return measure_processed(pixels(),boxes(),boxes(),TEXT,{'synthetic':True},config())


def count_tier(starts=(5,15,25,35),text='aaaa'):
    return dict(row=0,text=text,groups=[dict(first_ink_x=x,box=[x,0,x+3,5],shape_sha256='same',components=1) for x in starts])


class Tests(unittest.TestCase):
    def test_first_ink_global_coordinates(self):
        m=first_ink(pixels()<128,[3,10,50,20]);self.assertEqual(m['first_ink_x'],5)
    def test_empty_tier(self):self.assertEqual(first_ink(np.zeros((5,5),bool),[0,0,5,5])['status'],'empty_tier')
    def test_outside_box(self):
        for b in ([-1,0,5,5],[0,0,6,5],[0,0,0,5],[0,0,5,5.0]):
            with self.assertRaises(ValueError):first_ink(np.zeros((5,5),bool),b)
    def test_boundary_last_pixel(self):
        a=np.zeros((5,5),bool);a[4,4]=True;self.assertEqual(first_ink(a,[4,4,5,5])['first_ink_x'],4)
    def test_multicomponent_overlapping_projection(self):
        a=np.zeros((10,10),bool);a[1:3,2:5]=True;a[6:8,2:5]=True
        g=ink_groups(a);self.assertEqual(len(g),1);self.assertEqual(g[0]['components'],2)
    def test_separated_fragments_not_invented_glyph(self):
        a=np.zeros((8,10),bool);a[2:6,1:3]=True;a[2:6,5:7]=True
        self.assertEqual(len(ink_groups(a)),2)
    def test_pitch_median(self):self.assertEqual(estimate_pitch([count_tier()])['pitch_pixels'],10)
    def test_pitch_no_single_glyph_fallback(self):self.assertEqual(estimate_pitch([count_tier((5,), 'a')])['status'],'unavailable')
    def test_pitch_cardinality_mismatch(self):self.assertEqual(estimate_pitch([count_tier(text='aaa')])['status'],'unavailable')
    def test_pitch_inconsistent_spacing(self):self.assertEqual(estimate_pitch([count_tier((5,15,28,38))])['reason'],'INCONSISTENT_GLYPH_START_SPACING')
    def test_same_OCR_character_may_have_distinct_silhouettes(self):
        t=count_tier((5,18,25,38,45,58),'aaaaaa')
        for j,g in enumerate(t['groups']):g['shape_sha256']='A' if j%2==0 else 'B'
        p=estimate_pitch([t]);self.assertEqual(p['status'],'measured');self.assertEqual(p['pitch_pixels'],10)
        self.assertEqual(p['repeated_OCR_character_shape_variants'],{'a':2})
    def test_pitch_touching_group_not_accepted(self):
        t=count_tier();t['groups'][0]['box'][2]=15;self.assertEqual(estimate_pitch([t])['reason'],'GROUP_SPANS_CELL_PITCH_OR_TOUCHING_GLYPHS')
    def test_pitch_does_not_use_raw_space_gaps(self):
        t=count_tier((5,15,25,35,55),'aaaa a');self.assertEqual(estimate_pitch([t])['pitch_pixels'],10)
    def test_lost_ocr_space_lattice_fails(self):
        t=count_tier((5,15,25,35,55),'aaaaa');self.assertEqual(estimate_pitch([t])['status'],'unavailable')
    def test_full_offset_measurement(self):
        m=measurement();self.assertEqual(m['status'],'complete');self.assertEqual(m['rows'][0]['row_origin'],5)
        self.assertEqual(m['rows'][0]['tiers']['lower']['measured_offset'],2)
    def test_page_threshold_is_d1_otsu(self):self.assertEqual(measurement()['threshold']['method'],'frozen_D1_page_OTSU')
    def test_geometry_mismatch_unavailable(self):
        saved=boxes();saved[0][0]['box'][0]=1
        m=measure_processed(pixels(),boxes(),saved,TEXT,{},config());self.assertEqual(m['reason'],'D1_BOX_MISMATCH');self.assertFalse(m['geometry_verified'])
    def test_tier_schema_mismatch_unavailable(self):
        text=copy.deepcopy(TEXT);text[0]['lower']=''
        self.assertEqual(measure_processed(pixels(),boxes(),boxes(),text,{},config())['status'],'unavailable')
    def test_constant_image_unavailable(self):self.assertEqual(measure_processed(np.full((30,60),255,np.uint8),boxes(),boxes(),TEXT,{},config())['reason'],'CONSTANT_IMAGE')
    def test_measurement_has_no_source_or_layout_parameter(self):
        with self.assertRaises(TypeError):measure_processed(pixels(),boxes(),boxes(),TEXT,{},config(),validity='VALID')
    def test_count_group_failure_cannot_borrow_pitch_from_other_row(self):
        a=np.concatenate([pixels(),pixels()],axis=0);bb=boxes()+copy.deepcopy(boxes())
        for t in bb[1]:t['row']=1;t['box'][1]+=30;t['box'][3]+=30
        text=TEXT+[dict(upper='D',count='aaa',lower='R')]
        m=measure_processed(a,bb,bb,text,{},config());self.assertEqual(m['status'],'unavailable')
        self.assertIn('COUNT_GROUPING_UNSUPPORTED',m['reason'])


if __name__=='__main__':unittest.main()
