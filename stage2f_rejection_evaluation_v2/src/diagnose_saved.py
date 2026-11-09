"""Offline development analysis only. Never used by the runtime adapter."""
import json,csv,hashlib
from collections import Counter
from pathlib import Path
import sys
R=Path(r'C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2')
sys.path.insert(0,str(R/'src'))
from dataset_loader import DevelopmentDataset,safe_image_path
from integration import verify_integrity,digest
from evaluate_rejection import load_candidate
from observation_store import load_raw_run

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ds=DevelopmentDataset(verify_files=False)
    runs={}
    for p in sorted((R/'metrics').glob('*/results.json')):
        a=json.loads(p.read_text());c=a.get('candidate')
        if c in ('R0','R1'):
            if c in runs:raise ValueError('Multiple runs: select explicitly')
            runs[c]=(p,a)
    assert set(runs)=={'R0','R1'}
    for p,a in runs.values():load_candidate(p.parent,ds)
    assert runs['R0'][1]['lineage']==runs['R1'][1]['lineage']
    raws=[]
    for p in (R/'metrics').glob('*/observations.jsonl'):
        if digest(p)==runs['R1'][1]['lineage']['raw_observations_sha256']:raws.append(p)
    assert len(raws)==1
    checked_raw,lineage=load_raw_run(raws[0].parent,ds)
    raw={x['identity']['image_id']:x['observation'] for x in checked_raw}
    rows=runs['R1'][1]['rows'];byid={x['image_id']:x for x in rows}
    assert set(raw)==set(byid)==set(ds.by_id)
    facts=[]
    for x in rows:
        if x['validity']!='INVALID':continue
        m=ds.by_id[x['image_id']];p=m['mutation_parameters'];control=byid[x['matched_valid_control_id']]
        ri=p['row'];role='count' if p['target_tier']=='count_line' else p['target_tier']
        # Diagnostic raster-content comparison trims blank canvas margins only.
        # This is not an invalid-image CTC target or CER reference.
        display=[{('count' if k=='count_line' else k):v.strip(' ') for k,v in z.items() if v.strip(' ')} for z in m['raw_display']]
        o=raw[x['image_id']];decoded=x['raw_OCR_transcription']
        target_exact=ri<len(decoded) and decoded[ri].get(role)==display[ri].get(role,'')
        count_gap=any(' ' in z.get('count','').strip() for z in decoded)
        same=decoded==control['raw_OCR_transcription']
        boxes=[z['box'] for z in o['segmentation_rows'][ri] if z['role']==role] if ri<len(o['segmentation_rows']) else []
        layout=m['layout'];xx=layout['padding']+layout['extra_left']+p['column']*layout['cell_width']
        yy=layout['padding']+ri*(3*layout['line_height']+layout['row_gap'])+({'upper':0,'count':1,'lower':2}[role])*layout['line_height']
        # Nominal source cell coverage is not a foreground-preservation oracle.
        covered=bool(boxes and boxes[0][0]<=xx+layout['cell_width']/2<boxes[0][2] and boxes[0][1]<=yy<boxes[0][3])
        f=dict(image_id=x['image_id'],parent=x['source_parent_id'],category=x['invalidity_category'],variant=x['variant'],
            outcome=x['actual_outcome'],matched_control_id=control['image_id'],control_outcome=control['actual_outcome'],
            raw_pair_identical=same,interpreted_pair_identical=x['interpreted_transcription']==control['interpreted_transcription'],
            entire_mutated_tier_exact=target_exact,entire_display_exact=decoded==display,
            count_internal_gap_survives=count_gap,segmentation_accepted=o['segmentation_accepted'],
            transformed=o.get('diagnostics',{}).get('perspective_applied',False),nominal_mutation_cell_in_role_crop=covered,
            mutated_row=ri,mutated_role=role,display_mutated_tier=display[ri].get(role,''),predicted_mutated_tier=decoded[ri].get(role) if ri<len(decoded) else None,
            reasons=x['triggered_reason_codes'])
        ct=display[ri].get('count','');cp=decoded[ri].get('count','') if ri<len(decoded) else None
        f['count_only_space_loss']=cp!=ct and cp==ct.replace(' ','')
        f['count_tier_identical_to_control']=ri<len(decoded) and ri<len(control['raw_OCR_transcription']) and cp==control['raw_OCR_transcription'][ri].get('count')
        f['cause']=('D1_UNCERTAIN' if not o['segmentation_accepted'] else
            'F1_NO_SPATIAL_COUNT_VALIDATION' if target_exact and x['invalidity_category']!='wrong_tier' else
            'D1_E1_MUTATED_TIER_DIFFERS_UNLOCALIZED' if not target_exact else 'WRONG_TIER_PRESERVED_REVIEW')
        facts.append(f)
    def counts(group):
        return dict(images=len(group),parents=len({x['parent'] for x in group}),
                    **{k:dict(images=sum(bool(x[k]) for x in group),parents=len({x['parent'] for x in group if x[k]})) for k in
                    ('raw_pair_identical','interpreted_pair_identical','entire_mutated_tier_exact','entire_display_exact','count_internal_gap_survives','count_only_space_loss','count_tier_identical_to_control','segmentation_accepted','transformed','nominal_mutation_cell_in_role_crop')})
    summary={cat:{outcome:counts([x for x in facts if x['category']==cat and x['outcome']==outcome]) for outcome in ('ACCEPT','REVIEW_UNCERTAIN','REJECT_INVALID')} for cat in ('wrong_tier','invalid_count_token','missing_count')}
    summary['all_invalid']=counts(facts)
    summary['reasons']={v:dict(Counter('|'.join(x['triggered_reason_codes']) for x in rows if x['validity']==v and x['actual_outcome']=='REVIEW_UNCERTAIN')) for v in ('VALID','INVALID')}
    summary['valid_reviews']=[dict(id=x['image_id'],parent=x['source_parent_id'],reasons=x['triggered_reason_codes'],raw=x['raw_OCR_transcription'],expected=x['valid_reference_transcription']) for x in rows if x['validity']=='VALID' and x['actual_outcome']=='REVIEW_UNCERTAIN']
    summary['valid_count_gaps']=sum(any(' ' in z.get('count','').strip() for z in x['interpreted_transcription']) for x in rows if x['validity']=='VALID')
    summary['runs']={c:str(p.parent) for c,(p,a) in runs.items()};summary['raw_run']=str(raws[0].parent)
    summary['input_hashes']={str(p):digest(p) for p in [raws[0],*[p for p,a in runs.values()]]}
    # At most six representative pairs, limited to development.
    from PIL import Image,ImageChops,ImageDraw
    selections=[]
    for cat in ('wrong_tier','invalid_count_token','missing_count'):
        for exact in (True,False):
            candidates=[f for f in facts if f['category']==cat and f['outcome']=='ACCEPT' and f['variant']=='clean' and f['entire_mutated_tier_exact']==exact]
            if candidates:selections.append(min(candidates,key=lambda f:ds.by_id[f['image_id']]['layout']['columns']))
    # Reuse the already inspected sample record on rerun; do not reopen images.
    cached=out/'diagnosis_counts.json'
    if cached.exists():
        previous=json.loads(cached.read_text());samples=previous['pixel_samples']
        assert {s['image_id'] for s in samples}=={s['image_id'] for s in selections}
    else:samples=[]
    panel=Image.new('RGB',(1100,max(1,len(selections))*220),'white');draw=ImageDraw.Draw(panel)
    for i,f in enumerate(selections):
        if cached.exists():break
        m=ds.by_id[f['image_id']];ctrl=ds.by_id[f['matched_control_id']]
        ims=[Image.open(safe_image_path(ds.directory,z)).convert('L') for z in (ctrl,m)]
        for z,im in zip((ctrl,m),ims):assert digest(safe_image_path(ds.directory,z))==z['image_sha256']
        bbox=ImageChops.difference(*ims).getbbox();assert bbox
        l=m['layout'];p=m['mutation_parameters'];xx=l['padding']+l['extra_left']+p['column']*l['cell_width'];yy=l['padding']+p['row']*(3*l['line_height']+l['row_gap'])
        crop=(max(0,xx-60),max(0,yy-5),min(ims[0].width,xx+max(3,p['width'])*50+60),min(ims[0].height,yy+155))
        draw.text((5,i*220),f['image_id']+' exact tier='+str(f['entire_mutated_tier_exact']),fill='black')
        for j,im in enumerate(ims):
            tile=im.crop(crop);tile.thumbnail((520,180));panel.paste(tile,(j*550+5,i*220+35));draw.text((j*550+5,i*220+20),'VALID CONTROL' if j==0 else 'INVALID',fill='black')
        o=raw[f['image_id']];boxes=[z['box'] for row in o['segmentation_rows'] for z in row if z['row']==p['row'] and z['role']==f['mutated_role']]
        # Changed pixels within target tier crop; wrong-tier also changes source tier.
        diff=ImageChops.difference(*ims);inside=bool(boxes and diff.crop(tuple(boxes[0])).getbbox())
        samples.append(dict(image_id=f['image_id'],control_id=f['matched_control_id'],difference_bbox=bbox,changed_pixels=sum(v!=0 for v in diff.tobytes()),changed_pixels_inside_mutated_role_crop=inside,perspective_applied=o['diagnostics'].get('perspective_applied'),crop=crop))
    if not cached.exists():panel.save(out/'representative_pairs.png')
    summary['pixel_samples']=samples
    with (out/'failure_causes.csv').open('w',newline='',encoding='utf-8') as fp:
        w=csv.DictWriter(fp,fieldnames=list(facts[0]));w.writeheader();w.writerows(facts)
    (out/'diagnosis_counts.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('valid_reviews',)},indent=2))
    print(verify_integrity())

if __name__=='__main__':main(sys.argv[1])
