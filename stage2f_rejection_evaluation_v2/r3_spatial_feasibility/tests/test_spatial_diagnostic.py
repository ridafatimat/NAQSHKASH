import unittest,sys,tempfile,os,copy
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from spatial_diagnostic import ink_groups,component_observation,geometry_associations,score_components,latest
from alignment import map_span


def crop(rectangles,width=70,height=30):
    im=np.full((height,width),255,np.uint8)
    for a,b,c,d in rectangles:im[b:d,a:c]=0
    return im


def case(reference):return dict(capture=dict(source_to_processed_matrix=np.eye(3).tolist()),score=dict(reference=reference))


def glyph(c,box,col):return dict(character=c,box=box,column=col,x=(box[0]+box[2]-1)/2,width=box[2]-box[0])


class Tests(unittest.TestCase):
    def test_multicomponent_bar_and_dots_merge_by_projection(self):
        im=crop([(4,3,7,6),(13,3,16,6),(2,12,19,15)])
        r=ink_groups(im);self.assertEqual(len(r['components']),3);self.assertEqual(len(r['groups']),1)
        self.assertEqual(r['groups'][0]['component_count'],3);self.assertEqual(r['glyph_identities'],'NOT_INFERRED')
    def test_separated_dots_not_forced_into_one_glyph(self):
        r=ink_groups(crop([(3,3,6,6),(15,3,18,6)]));self.assertEqual(len(r['groups']),2)
        cap=dict(tiers=[dict(role='upper',box=[0,0,70,30])])
        o=component_observation(cap,lambda t:crop([(3,3,6,6),(15,3,18,6)]))
        score=score_components(o,case(dict(upper=[glyph('B',[3,3,18,6],0)],count=[],lower=[])))['metrics']
        self.assertEqual(score['fragmented_reference_glyphs'],1);self.assertEqual(score['matched_glyphs'],0)
        self.assertEqual(score['empty_candidates_unscorable'],2)
    def test_constant_crop_cannot_prove_semantic_absence(self):
        r=ink_groups(np.full((10,20),255,np.uint8));self.assertEqual(r['status'],'constant_unreadable_or_blank')
        self.assertNotIn('absence_confirmed',r)
    def test_image_only_association_and_offline_safety(self):
        images={'upper':crop([(10,3,20,9)]),'lower':crop([(40,3,50,9)]),'count':crop([(10,3,25,15),(40,3,55,15)])}
        cap=dict(tiers=[dict(role=k,box=[0,0,70,30]) for k in images]);o=component_observation(cap,lambda t:images[t['role']])
        ref=dict(upper=[glyph('D',[10,3,20,9],0)],lower=[glyph('R',[40,3,50,9],1)],count=[glyph('a',[10,3,25,15],0),glyph('b',[40,3,55,15],1)])
        m=score_components(o,case(ref))['metrics'];self.assertEqual(m['association_correct'],2);self.assertEqual(m['false_empty_tokens'],0)
        self.assertEqual(m['position_errors'],[0.,0.,0.,0.])
    def test_true_missing_vs_false_empty_on_valid_control(self):
        images={'upper':crop([(10,3,20,9)]),'lower':crop([(40,3,50,9)]),'count':crop([(40,3,55,15)])}
        cap=dict(tiers=[dict(role=k,box=[0,0,70,30]) for k in images]);o=component_observation(cap,lambda t:images[t['role']])
        ref=dict(upper=[glyph('D',[10,3,20,9],0)],lower=[glyph('R',[40,3,50,9],1)],count=[glyph('b',[40,3,55,15],1)])
        m=score_components(o,case(ref))['metrics'];self.assertEqual(m['detected_empty_tokens'],1)
        ref['count'].insert(0,glyph('a',[10,3,25,15],0));m=score_components(o,case(ref))['metrics']
        self.assertEqual(m['false_empty_tokens'],1)  # lost crop ink is not proven source absence
    def test_equal_anchor_ambiguity(self):
        t={r:dict(groups=[dict(global_box=[10,2,20,10])]) for r in ('upper','lower','count')}
        self.assertEqual(geometry_associations(t)['status'],'ambiguous_equal_symbol_anchors')
    def test_metadata_poison_does_not_change_observation(self):
        cap=dict(tiers=[dict(role='upper',box=[0,0,70,30])]);read=lambda t:crop([(10,3,20,9)])
        a=component_observation(cap,read);poison=copy.deepcopy(cap);poison.update(validity='INVALID',layout={'cell_width':999},mutation_category='missing_count')
        self.assertEqual(a,component_observation(poison,read))
    def test_merged_foreground_is_unscorable(self):
        cap=dict(tiers=[dict(role='upper',box=[0,0,70,30])]);o=component_observation(cap,lambda t:crop([(10,3,35,10)]))
        ref=dict(upper=[glyph('D',[10,3,20,10],0),glyph('F',[25,3,35,10],1)],count=[],lower=[])
        m=score_components(o,case(ref))['metrics'];self.assertEqual(m['matched_glyphs'],0);self.assertEqual(m['merged_or_unassigned_groups'],1)
    def test_lattice_math_is_not_a_fitted_bias(self):
        start=.5;jump=1;field=1
        for layer in range(6):
            field+=2*jump
            if layer<2:start+=.5*jump;field+=jump;jump*=2
        self.assertEqual((start,jump,field),(2.,4,42))
        m=map_span(dict(start_frame=0,end_frame=1),[20,20,1022,55],1832,np.eye(3))
        self.assertAlmostEqual(m['source_center'][0],20+2*1002/1832-.5)
    def test_latest_completed_ignores_incomplete(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in (1,2,3):
                p=root/str(i);p.mkdir();(p/'completion.json').write_text(json.dumps(dict(complete=i!=3)));(p/'metrics.json').write_text('{}')
                os.utime(p/'completion.json',(i,i))
            self.assertEqual(latest(root,('metrics.json',)).name,'2')


if __name__=='__main__':unittest.main()
