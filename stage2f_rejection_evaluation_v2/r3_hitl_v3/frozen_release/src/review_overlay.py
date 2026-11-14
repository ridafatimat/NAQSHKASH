"""Optional offline review regions. Never maps text columns to image pixels."""
import math,copy,html,io
from pathlib import Path
from policy_v3 import WORKSPACE,load_json,digest,content_hash

LABEL='REVIEW REGION — precise invalid glyph location not verified'


def validate_geometry(g):
    import numpy as np
    if not g or g.get('verified') is not True:raise ValueError('No verified processed/source geometry')
    M=np.asarray(g['source_to_processed_matrix'],dtype=float)
    if M.shape!=(3,3) or not np.isfinite(M).all() or abs(np.linalg.det(M))<1e-12:raise ValueError('Invalid coordinate transform')
    if len(g['source_dimensions'])!=2 or len(g['processed_dimensions'])!=2 or any(type(v) is not int or v<=0 for v in g['source_dimensions']+g['processed_dimensions']):raise ValueError('Invalid image dimensions')
    return M


def validate_box(box,size):
    if len(box)!=4 or not all(type(v) in (float,int) and math.isfinite(v) for v in box):raise ValueError('Invalid D1 box')
    a,b,c,d=box
    if not(0<=a<c<=size[0] and 0<=b<d<=size[1]):raise ValueError('Box outside verified image')
    return box


def plan_regions(findings,g):
    """Geometry metadata only, AFTER a decision. No image bytes or annotations."""
    import numpy as np
    if not findings:return dict(regions=[],unavailable_reason='No new R3 findings')
    try:M=validate_geometry(g)
    except (ValueError,TypeError,KeyError) as exc:return dict(regions=[],unavailable_reason=str(exc))
    inverse=np.linalg.inv(M);result=[]
    for fi,f in enumerate(findings):
        indices=[f['logical_row_index']] if f['logical_row_index'] is not None else f.get('candidate_logical_rows',[])
        for ri in indices:
            if type(ri) is not int or not 0<=ri<len(g['rows']):continue
            tiers=[t for t in g['rows'][ri] if f.get('tier') is None or t['role']==f['tier']]
            if not tiers:continue
            try:
                for t in tiers:validate_box(t['box'],g['processed_dimensions'])
                box=[min(t['box'][0] for t in tiers),min(t['box'][1] for t in tiers),max(t['box'][2] for t in tiers),max(t['box'][3] for t in tiers)]
                a,b,c,d=box;points=np.array([[a-.5,b-.5,1],[c-.5,b-.5,1],[c-.5,d-.5,1],[a-.5,d-.5,1]])@inverse.T
                if not np.isfinite(points).all() or np.any(np.abs(points[:,2])<1e-10):raise ValueError('Unmappable region')
                q=points[:,:2]/points[:,2:]
                source=[max(0.,float(q[:,0].min()+.5)),max(0.,float(q[:,1].min()+.5)),min(float(g['source_dimensions'][0]),float(q[:,0].max()+.5)),min(float(g['source_dimensions'][1]),float(q[:,1].max()+.5))]
                validate_box(source,g['source_dimensions'])
                result.append(dict(finding_index=fi,logical_row_index=ri,tier=f.get('tier'),processed_box=box,source_box=source,
                    source_image_dimensions=g['source_dimensions'],processed_image_dimensions=g['processed_dimensions'],
                    source_polygon_cv2=q.tolist(),region_scope='tier' if f.get('tier') else 'logical_row',
                    approximate=True,image_geometry_verified=True,precise_invalid_glyph_verified=False,label=LABEL))
            except (ValueError,TypeError):continue
    return dict(regions=result,unavailable_reason=None if result else 'No supported region for these findings')


def geometry_index(dataset,raw,baseline_raw_sha):
    """Use only independently cached D1 transforms whose identities match raw.

    Perspective=False is deliberately insufficient. No image bytes are opened.
    """
    index={};sources={};issues=[];root=WORKSPACE/'r3_alignment_experiment/runs'
    if not root.exists():return index,dict(sources=sources,issues=['No geometry cache'])
    for run in sorted(p for p in root.iterdir() if p.is_dir() and (p/'observations.jsonl').is_file()):
        done=load_json(run/'completion.json');plan=load_json(run/'plan.json')
        if plan.get('baseline_raw_sha256')!=baseline_raw_sha:continue
        if not done.get('complete') or done['raw_sha256']!=digest(run/'observations.jsonl') or done['plan_sha256']!=digest(run/'plan.json'):raise ValueError('Geometry cache hash drift')
        sources[str(run/'observations.jsonl')]=done['raw_sha256']
        for line in (run/'observations.jsonl').read_text().splitlines():
            item=__import__('json').loads(line);id=item['identity']['image_id']
            if id not in dataset.by_id:raise ValueError('Geometry outside selected development IDs')
            if item['identity']!=dataset.identity(dataset.by_id[id]) or item['payload_sha256']!=content_hash({k:v for k,v in item.items() if k!='payload_sha256'}):raise ValueError('Geometry identity drift')
            cap=item['observation']
            if cap['full_segmentation_rows']!=raw[id]['segmentation_rows'] or not cap.get('reproduction',{}).get('all_match'):
                issues.append(dict(image_id=id,reason='D1/OCR reproduction mismatch'));continue
            g=dict(verified=True,source_image_sha256=item['identity']['image_sha256'],rows=cap['full_segmentation_rows'],
                source_dimensions=cap['original_dimensions'],processed_dimensions=cap['processed_dimensions'],
                source_to_processed_matrix=cap['source_to_processed_matrix'],source_record_sha256=item['payload_sha256'])
            try:validate_geometry(g)
            except (ValueError,TypeError,KeyError) as exc:issues.append(dict(image_id=id,reason=str(exc)));continue
            if id in index and {k:v for k,v in index[id].items() if k!='source_record_sha256'}!={k:v for k,v in g.items() if k!='source_record_sha256'}:raise ValueError('Conflicting verified geometry caches')
            index[id]=g
    return index,dict(sources=sources,issues=issues,images_with_verified_geometry=len(index))


def render_image(raw,regions,destination):
    """Called only after decisions were sealed. No GT or text-column input."""
    from PIL import Image,ImageDraw
    if not regions:raise ValueError('No region: do not fabricate an overlay')
    with Image.open(io.BytesIO(raw)) as im:image=im.convert('RGB')
    for r in regions:
        if not r.get('image_geometry_verified') or r.get('precise_invalid_glyph_verified'):raise ValueError('Unsupported overlay claim')
        if list(image.size)!=r['source_image_dimensions']:raise ValueError('Overlay image dimensions differ from verified source')
        validate_box(r['source_box'],list(image.size))
    # Draw at original coordinates, then make one small annotated thumbnail.
    draw=ImageDraw.Draw(image)
    for r in regions:draw.rectangle(r['source_box'],outline='orange',width=3)
    image.thumbnail((1400,1200));canvas=Image.new('RGB',(max(image.width,700),image.height+50),'white');canvas.paste(image,(0,50))
    ImageDraw.Draw(canvas).text((8,8),'REVIEW REGION - precise invalid glyph location not verified',fill='black')
    canvas.save(destination)


def generate_gallery(records,dataset,output,limit=6,image_ids=None):
    """Small development-only offline gallery; missing geometry -> text only."""
    from dataset_loader import image_bytes
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if not 1<=limit<=12:raise ValueError('Small gallery limit must be 1..12')
    byid={r['image_id']:r for r in records}
    if image_ids:
        if len(set(image_ids))!=len(image_ids) or len(image_ids)>limit or any(i not in byid for i in image_ids):raise ValueError('Invalid selected development IDs')
        selected=[byid[i] for i in image_ids]
    else:selected=sorted((r for r in records if r['new_review']),key=lambda r:(not bool(r['review_regions']['regions']),r['image_id']))[:limit]
    entries=[];rendered=0
    for r in selected:
        if r['R3']['outcome']!='REVIEW_UNCERTAIN':raise ValueError('Only review records may be highlighted')
        regions=r['review_regions']['regions'];picture=None
        if regions:
            source=dataset.by_id[r['image_id']]
            if source['dataset_split']!='development_design' or source['image_sha256']!=r['image_sha256']:raise ValueError('Unauthorized image identity')
            raw=image_bytes(dataset.directory,source)  # AFTER all decisions; selected IDs only
            picture=r['image_id']+'.png';render_image(raw,regions,output/picture);rendered+=1
        text='\n'.join(f['explanation'] for f in r['findings']) or r['R3'].get('explanation','Review original R2 evidence.')
        entries.append('<section><h2>'+html.escape(r['image_id'])+'</h2><pre>'+html.escape(text)+'</pre>'+('<img src="'+picture+'">' if picture else '<p>No verified image-coordinate region. Text findings only.</p>')+'</section>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>R3 review regions</title><style>body{font:15px system-ui}pre{white-space:pre-wrap}img{max-width:100%}</style><h1>Approximate review regions</h1><p>'+LABEL+'</p>'+''.join(entries),encoding='utf-8')
    return dict(selected=len(selected),rendered=rendered,text_only=len(selected)-rendered,precise_glyph_localization_claimed=False)
