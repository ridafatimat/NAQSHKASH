"""Offline R3 development audit. Prints statistics only; never runs OCR or changes results.

--wrong-tier-pixels additionally verifies 38 development negatives and 38 controls
using the existing fixed-font inverse renderer, for DIAGNOSIS ONLY.
"""
import argparse,json,sys,csv
from pathlib import Path
from collections import Counter,defaultdict
R=Path(r'C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2')
sys.path.insert(0,str(R/'src'))
from integration import load_json,digest,rules,page_errors
from dataset_loader import DevelopmentDataset,safe_image_path
from evaluate_rejection import load_candidate
from observation_store import load_raw_run
from evaluate_r2 import verify_r2
from measurements import alignment_targets


def normalized(rows,ink=False,aliases=False):
    def text(s):
        s=s.replace(' ','') if ink else s.strip(' ')
        return s.replace('X','B') if aliases else s
    return [{('count' if k=='count_line' else k):text(v) for k,v in row.items() if text(v)} for row in rows]


def numbers(items):return dict(images=len(items),parents=len({i['parent'] for i in items}))


def main(pixel_check=False,include_cases=False):
    verify_r2();ds=DevelopmentDataset(verify_files=False);runs={};paths={}
    for p in sorted((R/'metrics').glob('*/results.json')):
        a=load_json(p);c=a.get('candidate')
        if c in ('R0','R1','R2'):
            if c in runs:raise ValueError('Multiple candidate runs; resolve selection explicitly')
            done=load_json(p.parent/'completion.json')
            assert done['complete'] and done['results_sha256']==digest(p) and done['candidate']==c
            assert {x['image_id'] for x in a['rows']}==set(ds.by_id) and len(a['rows'])==len(ds.rows)
            if c in ('R0','R1'):load_candidate(p.parent,ds)
            runs[c]=a;paths[c]=str(p.parent)
    assert set(runs)=={'R0','R1','R2'}
    assert runs['R0']['lineage']==runs['R1']['lineage']==runs['R2']['lineage']
    assert runs['R2']['r2_lock_sha256']==digest(R/'configs/r2_lock.json')
    raw_paths=[p for p in (R/'metrics').glob('*/observations.jsonl') if digest(p)==runs['R1']['lineage']['raw_observations_sha256']]
    assert len(raw_paths)==1
    raw,lineage=load_raw_run(raw_paths[0].parent,ds);raw={x['identity']['image_id']:x['observation'] for x in raw}
    byid={c:{x['image_id']:x for x in a['rows']} for c,a in runs.items()}
    exclusions=load_json(R.parent/'stage2f_negative_dataset_v2/configs/visual_alias_exclusions.json')
    font_audit=load_json(R.parent/'stage2f_tier_aware_decoding/reports/glyph_audit_summary.json')
    uncertain=set(c for group in exclusions['groups'] for c in group)
    facts=[];metrics=defaultdict(lambda:dict(pages=0,characters=0,raw_edits=0,t1_edits=0,raw_exact=0,t1_exact=0))
    schema=Counter();frames=0;frames_bad=0;perspective=0
    for id,o in raw.items():
        schema.update(o.keys());perspective+=bool(o['diagnostics'].get('perspective_applied'))
        for t in o['tiers']:
            frames+=1;frames_bad+=t['signals']['frames']!=t['resized_width']//4
    for m in ds.rows:
        if m['validity_status']!='INVALID':continue
        id=m['image_id'];r1=byid['R1'][id];r2=byid['R2'][id];o=raw[id]
        assert r1['raw_OCR_transcription']==o['decoded_rows']==r2['raw_OCR_transcription']
        assert r1['interpreted_transcription']==r2['interpreted_transcription']
        assert r2['image_sha256']==m['image_sha256'] and r2['split']=='development_design'
        p=m['mutation_parameters'];ri=p['row'];role='count' if p['target_tier']=='count_line' else p['target_tier']
        assert m['layout']['font_sha256']==font_audit['font_sha256']==exclusions['font_sha256'] and m['layout']['font_size_px']==48
        display=normalized(m['raw_display']);reference=normalized(m['raw_display'],ink=True)
        rawtext=normalized(o['decoded_rows']);t1text=normalized(r1['interpreted_transcription'])
        rawink=normalized(o['decoded_rows'],ink=True);t1ink=normalized(r1['interpreted_transcription'],ink=True)
        ref_alias=normalized(m['raw_display'],ink=True,aliases=True)
        no_ocr=not o['segmentation_accepted'] or o['ocr_status']!='ok'
        expected=m['raw_display'][ri][p['target_tier']];ct=m['raw_display'][ri]['count_line'].strip(' ')
        cp=rawtext[ri].get('count','') if ri<len(rawtext) else None
        local_status='spatial_only'
        local_options=[]
        if m['invalidity_category']!='missing_count' and not no_ocr:
            start=len(expected[:p['column']].replace(' ',''));token=p['original_symbol'] if role!='count' else p['displayed_mutated_token']
            ref=expected.replace(' ','');pred=rawink[ri].get(role,'') if ri<len(rawink) else ''
            local_options=[({ref[i]} if ref==pred else alignment_targets(ref,pred,i)) for i in range(start,start+len(token))]
            local_status=('faithful' if all(s=={c} for s,c in zip(local_options,token)) else
                'lost_or_changed' if any(c not in s for s,c in zip(local_options,token)) else 'ambiguous_edit_alignment')
        gap_loss=bool(' ' in ct and cp!=ct and cp==ct.replace(' ',''))
        faithful_raw=normalized(o['decoded_rows'],ink=True,aliases=True)==ref_alias
        faithful_t1=normalized(r1['interpreted_transcription'],ink=True,aliases=True)==ref_alias
        full_pattern_raw=normalized(o['decoded_rows'],aliases=True)==normalized(m['raw_display'],aliases=True)
        primary=('D' if no_ocr else 'B' if local_status=='lost_or_changed' or gap_loss else
                 'D' if local_status=='ambiguous_edit_alignment' else
                 'A' if local_status=='faithful' and r2['actual_outcome']=='ACCEPT' else
                 'C' if m['invalidity_category']=='missing_count' else 'reviewed_without_failure')
        f=dict(id=id,parent=m['source_parent_id'],category=m['invalidity_category'],variant='illuminated' if m['augmentation_recipe'] else 'clean',
            primary=primary,raw_glyph_faithful=faithful_raw,t1_glyph_faithful=faithful_t1,
            raw_pattern_match=full_pattern_raw,t1_pattern_match=normalized(r1['interpreted_transcription'],aliases=True)==normalized(m['raw_display'],aliases=True),
            raw_literal_pattern_match=rawtext==display,t1_literal_pattern_match=t1text==display,
            raw_glyph_discrepancy=not faithful_raw,t1_glyph_discrepancy=not faithful_t1,
            gap_loss=gap_loss,local_mutation_status=local_status,spatial_required=m['invalidity_category']!='wrong_tier',
            r1_outcome=r1['actual_outcome'],r2_outcome=r2['actual_outcome'],no_ocr=no_ocr,
            full_reference_unambiguous=not any(c in uncertain for row in reference for text in row.values() for c in text),
            affected_tier_exact=ri<len(rawtext) and rawtext[ri].get(role,'')==display[ri].get(role,''),
            raw_pair_identical=o['decoded_rows']==raw[next(z['image_id'] for z in ds.groups[m['source_parent_id']] if z['validity_status']=='VALID' and z['augmentation_recipe']==m['augmentation_recipe'])]['decoded_rows'],
            local_OCR_options=[sorted(s,key=str) for s in local_options])
        facts.append(f)
        # Visible INK glyph streams only: no whitespace-based CER, no CTC targets.
        if f['full_reference_unambiguous'] and not no_ocr:
            for cohort in ('all',m['invalidity_category']):
                metric=metrics[cohort];metric['pages']+=1
                for name,pred in [('raw',rawink),('t1',t1ink)]:
                    errors=page_errors(reference,pred);metric[name+'_edits']+=errors['edits'];metric[name+'_exact']+=errors['exact']
                metric['characters']+=sum(len(t) for row in reference for t in row.values())
        # Count-only visible glyph streams have unambiguous glyphs in this font.
        if no_ocr:continue
        cref=[dict(count=row['count']) for row in reference]
        for cohort in ('count_only_all','count_only_'+m['invalidity_category']):
            metric=metrics[cohort];metric['pages']+=1;metric['characters']+=sum(len(z['count']) for z in cref)
            for name,pred in [('raw',rawink),('t1',t1ink)]:
                errors=page_errors(cref,[dict(count=row.get('count','')) for row in pred]);metric[name+'_edits']+=errors['edits'];metric[name+'_exact']+=errors['exact']
    def summarize(group):
        ans=numbers(group);ans['primary']={k:dict(**numbers([x for x in group if x['primary']==k]),percent=100*sum(x['primary']==k for x in group)/len(group)) for k in ('A','B','C','D','reviewed_without_failure')}
        keys=('raw_glyph_faithful','t1_glyph_faithful','raw_pattern_match','t1_pattern_match','raw_literal_pattern_match','t1_literal_pattern_match','gap_loss','raw_glyph_discrepancy','t1_glyph_discrepancy','no_ocr','full_reference_unambiguous','affected_tier_exact','raw_pair_identical')
        ans.update({k:numbers([x for x in group if x[k]]) for k in keys})
        for key in ('local_mutation_status','r1_outcome','r2_outcome'):
            ans[key]={v:numbers([x for x in group if x[key]==v]) for v in sorted({x[key] for x in group})}
        for pred in ('raw','t1'):
            ans[pred+'_faithful_but_R2_accepts']=numbers([x for x in group if x[pred+'_glyph_faithful'] and x['r2_outcome']=='ACCEPT'])
            ans[pred+'_pattern_but_R2_accepts']=numbers([x for x in group if x[pred+'_pattern_match'] and x['r2_outcome']=='ACCEPT'])
        ans['examples']={k:[x['id'] for x in group if x['primary']==k][:3] for k in ('A','B','C','D')}
        return ans
    summary=dict(runs=paths,raw_run=str(raw_paths[0].parent),input_hashes={str(Path(p)/'results.json'):digest(Path(p)/'results.json') for p in paths.values()},
        all=summarize(facts),categories={c:summarize([x for x in facts if x['category']==c]) for c in sorted({x['category'] for x in facts})},
        variants={c:summarize([x for x in facts if x['variant']==c]) for c in ('clean','illuminated')},lexical_metrics=dict(metrics),
        raw_schema_keys=dict(schema),tiers_with_frame_counts=frames,frame_length_formula_mismatches=frames_bad,perspective_images=perspective)
    # B flags and reference availability overlap primary causes; do not sum them.
    summary['any_glyph_discrepancy_or_proven_gap_loss']=numbers([x for x in facts if not x['no_ocr'] and (x['raw_glyph_discrepancy'] or x['gap_loss'])])
    summary['wrong_tier_accepted']=[dict(id=x['id'],parent=x['parent'],local_status=x['local_mutation_status'],local_OCR_options=x['local_OCR_options']) for x in facts if x['category']=='wrong_tier' and x['r1_outcome']=='ACCEPT']
    if pixel_check:
        sys.path.insert(0,str(R.parent/'stage2f_negative_dataset_v2/src'))
        # Existing runtime loads project talim; the offline verifier also needs
        # renderer/generator modules from the original dataset-tools package.
        import talim
        talim.__path__.append(r'C:\FYP Dataset Generation\dataset_tools\talim')
        from independent_verify import verify_image,recover_cells,role_candidates,atlas
        from PIL import Image,ImageChops
        audit=[];count_read=0
        for f in summary['wrong_tier_accepted']:
            m=ds.by_id[f['id']];control=next(z for z in ds.groups[m['source_parent_id']] if z['validity_status']=='VALID' and z['augmentation_recipe']==m['augmentation_recipe']);layout=m['layout'];p=m['mutation_parameters']
            images=[]
            for z in (m,control):
                path=safe_image_path(ds.directory,z);assert digest(path)==z['image_sha256']
                with Image.open(path) as im:images.append(im.convert('L'))
                count_read+=1
            im,ctrl=images;v=verify_image(im,layout,m['augmentation_recipe']);assert v['status']=='AUTO_VERIFIED'
            cells,unknown=recover_cells(im,layout,m['augmentation_recipe']);glyphs=cells[p['row']][p['target_tier']][p['column']]
            allowed=rules.EXCLUSIVE_UP_SYMBOLS if p['target_tier']=='upper' else rules.EXCLUSIVE_DOWN_SYMBOLS
            legal=sorted(set(glyphs)&set(allowed));possible=role_candidates(p['original_symbol'],p['target_tier'],layout)
            aliases=sorted({c for group in exclusions['groups'] if p['original_symbol'] in group for c in group if c in allowed})
            # Strict visible-ink equivalence under each image's brightness transfer.
            templates=atlas(layout['cell_width'],layout['line_height']);recipe=m['augmentation_recipe']
            if recipe:templates={c:t.point([int(i*recipe['operations'][0]['factor']) for i in range(256)]) for c,t in templates.items()}
            def visible(t):
                ink=ImageChops.difference(templates[' '],t);b=ink.getbbox();return ink.crop(b) if b else None
            original=visible(templates[p['original_symbol']]);equal=[]
            for c in allowed:
                other=visible(templates[c])
                if original is not None and other is not None and original.size==other.size and original.tobytes()==other.tobytes():equal.append(c)
            diff=ImageChops.difference(im,ctrl)
            audit.append(dict(id=f['id'],parent=f['parent'],symbol=p['original_symbol'],target=p['target_tier'],
                exact_visible_legal_aliases=sorted(equal),conservative_legal_aliases=aliases,raster_role_candidates=possible,
                cell_matches=glyphs,legal_cell_candidates=legal,classification=v['classification'],reason_codes=v['reason_codes'],
                changed_pixels=sum(n for value,n in enumerate(diff.histogram()) if value),difference_bbox=diff.getbbox(),
                local_OCR_status=f['local_status'],local_OCR_options=f['local_OCR_options'],control_id=control['image_id'],recipe=recipe))
        summary['wrong_tier_pixel_audit']=dict(image_files_opened=count_read,rows=audit)
    if include_cases:summary['cases']=facts
    verify_r2();print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wrong-tier-pixels',action='store_true')
    parser.add_argument('--cases',action='store_true',help='Include all per-image primary causes and secondary flags in stdout JSON')
    args=parser.parse_args();main(args.wrong_tier_pixels,args.cases)
