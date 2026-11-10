import copy
import tempfile
import unittest
from unittest.mock import patch
from collections import Counter
from common_v2 import *
from generate_negative_v2 import make_parent, source_plan
from render_invalid import layout_for,render_display,pixel_difference
from independent_verify import verify_image,verify_matched_negative,display_agrees,role_candidates
from mutations import mutate
from scan_variants import make_variant
from create_splits import assign_splits,verify_splits
from validate_dataset import validate


class RasterRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample,_=make_parent('unit_v2',2,42)
        cls.layout=layout_for(cls.sample)
        cls.valid=render_display(displayed_rows(cls.sample),cls.layout)

    def test_control_original_validator_and_inverse_parser(self):
        self.assertEqual(check_sample(self.sample),[])
        self.assertEqual(verify_image(self.valid,self.layout)['classification'],'VALID')

    def test_canonical_and_isolated_renderers_agree(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'valid.png';render_talim_text(rules.canonical_text(self.sample),p,FONT)
            with Image.open(p) as im:self.assertEqual(im.tobytes(),self.valid.tobytes())

    def test_all_three_visible_mutations_independently_invalid(self):
        for category,rule in RULES.items():
            display,_=mutate(self.sample,category,self.layout)
            invalid=render_display(display,self.layout)
            self.assertGreater(pixel_difference(self.valid,invalid)['changed_pixels'],0)
            proof=verify_matched_negative(self.valid,invalid,self.layout,display,rule)
            self.assertEqual(proof['status'],'AUTO_VERIFIED',proof)
            self.assertIn(rule,proof['reason_codes'])
            self.assertEqual(proof['evidence']['legal_canonical_interpretations'],0)

    def test_annotation_only_defect_cannot_enter_collection(self):
        fake=copy.deepcopy(self.sample);fake['logical_rows'][0]['runs'][0]['count']=0
        self.assertTrue(check_sample(fake))
        proof=verify_matched_negative(self.valid,self.valid,self.layout,displayed_rows(self.sample))
        self.assertEqual(proof['status'],'EXCLUDED')
        self.assertEqual(verify_image(self.valid,self.layout)['classification'],'VALID')

    def test_alias_only_wrong_tier_is_excluded_even_when_pixels_change(self):
        sample=generate_talim('alias',1,30,['B','R'],1,2,max_run_length=259,
                             run_pattern='controlled_rows',controlled_row_pattern=[('B',11),('R',19)]).to_metadata()
        layout=layout_for(sample);display=displayed_rows(sample);valid=render_display(display,layout)
        self.assertIn('X',role_candidates('B','lower',layout))
        row={k:list(v.ljust(layout['columns'])) for k,v in display[0].items()}
        row['upper'][0]=' ';row['lower'][0]='B'
        raw=[{k:''.join(v) for k,v in row.items()}];im=render_display(raw,layout)
        self.assertGreater(pixel_difference(valid,im)['changed_pixels'],0)
        self.assertEqual(verify_image(im,layout)['classification'],'VALID')
        self.assertEqual(verify_matched_negative(valid,im,layout,raw)['status'],'EXCLUDED')

    def test_count_reverse_outside_authentic_set(self):
        display,params=mutate(self.sample,'invalid_count_token',self.layout)
        self.assertNotIn(params['displayed_mutated_token'],{rules.encode_count(n) for n in range(1,260)})
        self.assertEqual(verify_image(render_display(display,self.layout),self.layout)['classification'],'INVALID')

    def test_near_identical_raster_shades_cannot_manufacture_wrong_tier(self):
        sample=generate_talim('near_alias',1,30,['C','R'],1,2,max_run_length=259,
                             run_pattern='controlled_rows',controlled_row_pattern=[('C',11),('R',19)]).to_metadata()
        layout=layout_for(sample);display=displayed_rows(sample);valid=render_display(display,layout)
        self.assertIn('Y',role_candidates('C','lower',layout))
        row={k:list(v.ljust(layout['columns'])) for k,v in display[0].items()}
        row['upper'][0]=' ';row['lower'][0]='C'
        raw=[{k:''.join(v) for k,v in row.items()}];im=render_display(raw,layout)
        self.assertEqual(verify_matched_negative(valid,im,layout,raw)['status'],'EXCLUDED')

    def test_all_authentic_count_boundaries_stay_valid(self):
        for n in (1,9,10,11,99,250,259):
            s=generate_talim('count',1,n,['A'],3,4,max_run_length=259).to_metadata()
            l=layout_for(s)
            self.assertEqual(verify_image(render_display(displayed_rows(s),l),l)['classification'],'VALID',n)

    def test_range_errors_are_not_invented_as_visible_glyphs(self):
        for n in (0,-1,260,True,'11'):
            with self.assertRaises(ValueError):rules.encode_count(n)
        with self.assertRaises(ValueError):mutate(self.sample,'count_above_max',self.layout)

    def test_missing_count_requires_no_alternate_parse(self):
        display,_=mutate(self.sample,'missing_count',self.layout)
        proof=verify_image(render_display(display,self.layout),self.layout)
        self.assertEqual(proof['evidence']['legal_canonical_interpretations'],0)
        self.assertIn('CANON_REQUIRED_SYMBOL_COUNT',proof['reason_codes'])

    def test_simply_changing_to_another_valid_instruction_is_not_negative(self):
        s=generate_talim('other',1,20,['A','R'],3,4,max_run_length=259,
                        run_pattern='controlled_rows',controlled_row_pattern=[('A',11),('R',9)]).to_metadata()
        l=layout_for(s);d=displayed_rows(s);valid=render_display(d,l)
        # Canonical value11 ->12, still an authentic supported instruction.
        d[0]['count_line']=d[0]['count_line'].replace('ja','jb',1)
        changed=render_display(d,l)
        self.assertEqual(verify_image(changed,l)['classification'],'VALID')
        self.assertEqual(verify_matched_negative(valid,changed,l,d)['status'],'EXCLUDED')

    def test_alignment_mutation_is_not_approved(self):
        with self.assertRaises(ValueError):mutate(self.sample,'alignment',self.layout)

    def test_raw_annotation_agrees_with_image(self):
        display,_=mutate(self.sample,'missing_count',self.layout)
        im=render_display(display,self.layout)
        self.assertTrue(display_agrees(im,self.layout,display))
        self.assertFalse(display_agrees(im,self.layout,displayed_rows(self.sample)))

    def test_illumination_preserves_every_cell_and_label(self):
        display,_=mutate(self.sample,'wrong_tier',self.layout)
        for image,expected in [(self.valid,'VALID'),(render_display(display,self.layout),'INVALID')]:
            for seed in (2,3):
                _,scan,recipe=make_variant(image,seed)
                self.assertEqual(verify_image(scan,self.layout,recipe)['classification'],expected)

    def test_lossy_or_geometric_augmentation_not_silently_certified(self):
        with self.assertRaises(ValueError):verify_image(self.valid,self.layout,dict(padding=0,operations=[dict(type='rotation',degrees=1)]))

    def test_quality_failure_separate_from_semantic_invalidity(self):
        for color in (0,255):
            im=Image.new('L',self.valid.size,color)
            self.assertEqual(verify_image(im,self.layout)['classification'],'IMAGE_QUALITY_FAILURE')

    def test_unknown_raster_is_not_labeled_invalid(self):
        im=self.valid.copy();im.putpixel((self.layout['padding']+self.layout['extra_left'],self.layout['padding']),123)
        result=verify_image(im,self.layout)
        self.assertEqual(result['status'],'NEEDS_DOMAIN_REVIEW')
        self.assertEqual(result['classification'],'UNSUPPORTED_OR_AMBIGUOUS')

    def test_property_style_multiple_new_sources(self):
        for i in range(6):
            s,_=make_parent('unit_properties',i+30,123)
            l=layout_for(s);v=render_display(displayed_rows(s),l)
            self.assertEqual(verify_image(v,l)['classification'],'VALID')
            for category in RULES:
                d,_=mutate(s,category,l,i);m=render_display(d,l)
                self.assertEqual(verify_matched_negative(v,m,l,d,RULES[category])['status'],'AUTO_VERIFIED')


class OwnershipTests(unittest.TestCase):
    def test_parent_split_70_15_15(self):
        assignment=assign_splits([str(i) for i in range(100)],42)
        self.assertEqual(Counter(assignment.values()),Counter(development_design=70,validation_calibration=15,protected_test=15))

    def test_preflight_never_assigns_protected_test(self):
        self.assertEqual(set(assign_splits(['a','b'],42,True).values()),{'development_design'})

    def test_variants_cannot_cross_parent_split(self):
        rows=[dict(source_parent_id='p',mutation_family_id='p',image_sha256='a',dataset_split='development_design'),
              dict(source_parent_id='p',mutation_family_id='p',image_sha256='b',dataset_split='protected_test')]
        with self.assertRaises(ValueError):verify_splits(rows)

    def test_duplicate_bytes_cannot_cross_split(self):
        rows=[dict(source_parent_id='p',mutation_family_id='p',image_sha256='same',dataset_split='development_design'),
              dict(source_parent_id='q',mutation_family_id='q',image_sha256='same',dataset_split='protected_test')]
        with self.assertRaises(ValueError):verify_splits(rows)

    def test_new_sources_are_deterministic_and_not_mutation_siblings(self):
        a,_=make_parent('unit_sources',1,42);b,_=make_parent('unit_sources',1,42);c,_=make_parent('unit_sources',2,42)
        self.assertEqual(a,b);self.assertNotEqual(rules.semantic_key(a),rules.semantic_key(c))

    def test_history_collisions_are_excluded_not_backfilled(self):
        config=read_json(ROOT/'configs/generation_v2.yaml')
        s,_=make_parent(config['preflight_namespace'],0,config['seed'])
        plans,excluded=source_plan(config,'preflight',1,dict(semantic_keys=[rules.semantic_key(s)],mode='bounded'))
        self.assertFalse(plans);self.assertEqual(excluded[0]['status'],'EXCLUDED')

    def test_historical_inputs_and_frozen_design_unchanged(self):
        self.assertGreater(verify_historical(),20)
        self.assertGreater(verify_design(),5)

    def test_preflight_full_lineage_annotation_duplicates(self):
        dataset=ROOT/'dataset/preflight'
        if not (dataset/'completion.json').exists():self.skipTest('preflight not yet built')
        result=validate(dataset)
        self.assertTrue(result['parent_disjoint']);self.assertEqual(result['duplicates'],0)
        rows=read_json(dataset/'manifest.json')
        self.assertEqual(len({r['source_parent_id'] for r in rows}),30)
        self.assertTrue(all(r['human_review_status']=='NOT_REVIEWED' for r in rows))
        self.assertTrue(all(r['ctc_transcription'] is None for r in rows if r['validity_status']=='INVALID'))

    def test_sealed_protected_examples_are_hash_only(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);(d/'annotations').mkdir()
            image=d/'image.png';Image.new('L',(2,2),255).save(image)
            secret=d/'annotations/protected.json';secret.write_text('NOT_JSON_DO_NOT_PARSE')
            row=dict(image_id='protected',image_file_path='image.png',image_sha256=digest(image),
                source_parent_id='parent',mutation_family_id='parent',dataset_split='protected_test',
                augmentation_recipe=None,validity_status='VALID')
            write_json(d/'manifest.json',[row]);write_json(d/'parents.json',[dict(source_parent_id='parent')])
            write_json(d/'completion.json',dict(complete=True,manifest_sha256=digest(d/'manifest.json')))
            write_json(d/'protected_test_seal.json',dict(frozen=True,files={'image.png':digest(image),'annotations/protected.json':digest(secret)}))
            with patch('validate_dataset.verify_image',side_effect=AssertionError('Must not evaluate sealed test')):
                result=validate(d)
            self.assertEqual(result['protected_hash_only_images'],1)
            self.assertEqual(result['semantic_images_checked'],0)

    def test_quarantined_datasets_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);write_json(p/'QUARANTINED.json',dict(reason='unproven visual invalidity'))
            with self.assertRaisesRegex(ValueError,'Quarantined'):validate(p)


if __name__=='__main__':unittest.main()
