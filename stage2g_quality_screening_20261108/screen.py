"""Conservative local screening. No OCR, model imports or automatic acceptance."""
import argparse,csv,hashlib,html,json,unittest,zipfile
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from PIL import Image

AUDIT=Path(r'C:\NAQSHKASH\stage2\stage2g_handwritten_dataset_preparation_20261108')
def digest(b):return hashlib.sha256(b).hexdigest()
def readcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvout(p,rows,fields=None):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def decode(b):
    # Pillow verifies structure; OpenCV supplies consistent BGR analysis pixels.
    import io
    with Image.open(io.BytesIO(b)) as im:im.verify()
    with Image.open(io.BytesIO(b)) as im:im.load()
    a=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR)
    if a is None:raise ValueError('image decode failed')
    return a
def gray(a):
    a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
    scale=min(1,900/max(a.shape));return cv2.resize(a,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale<1 else a
def metrics(a):
    g=gray(a);return {'pixel_range':int(a.max())-int(a.min()),'gray_std':float(g.std()),'contrast_p95_p5':float(np.percentile(g,95)-np.percentile(g,5)),'laplacian_variance':float(cv2.Laplacian(g,cv2.CV_64F).var()),'near_extreme_fraction':float(np.mean((g<=2)|(g>=253)))}
def objective(a):
    # Only spatially constant (within one level/channel) images qualify as blank.
    return bool(all(np.ptp(a[:,:,c])<=1 for c in range(3)))
def features(g):
    det=cv2.ORB_create(nfeatures=1800);return det.detectAndCompute(g,None)
def coverage(g,H,target_shape):
    # Reference foreground is a proxy including ruling/shadows, never a transcription.
    mask=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,12)
    h,w=target_shape;yy,xx=np.nonzero(mask)
    if not len(xx):return 0.,0.
    pts=np.stack([xx,yy],axis=1).astype(np.float32).reshape(-1,1,2)
    q=cv2.perspectiveTransform(pts,H).reshape(-1,2)
    lost=np.mean((q[:,0]<0)|(q[:,0]>=w)|(q[:,1]<0)|(q[:,1]>=h))
    warped=cv2.warpPerspective(mask,H,(w,h),flags=cv2.INTER_NEAREST)
    return float(lost),warped
def compare(ref,g,rf,gf):
    result={'registration':'UNRESOLVED','inliers':0,'inlier_fraction':0.,'reference_feature_coverage':0.,'reference_foreground_outside_fraction':'','unmatched_foreground_fraction':''}
    k1,d1=rf;k2,d2=gf
    if d1 is None or d2 is None:return result
    pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1,d2,k=2)
    good=[m[0] for m in pairs if len(m)==2 and m[0].distance<.72*m[1].distance]
    if len(good)<16:return result
    p=np.float32([k1[m.queryIdx].pt for m in good]);q=np.float32([k2[m.trainIdx].pt for m in good])
    H,mask=cv2.findHomography(p,q,cv2.RANSAC,3)
    if H is None:return result
    keep=mask.ravel().astype(bool);n=int(keep.sum());frac=n/len(good)
    area=cv2.contourArea(cv2.convexHull(p[keep]))/(ref.shape[0]*ref.shape[1]) if n>=3 else 0
    result.update(inliers=n,inlier_fraction=frac,reference_feature_coverage=area)
    corners=np.float32([[0,0],[ref.shape[1],0],[ref.shape[1],ref.shape[0]],[0,ref.shape[0]]]).reshape(-1,1,2)
    transformed=cv2.perspectiveTransform(corners,H)
    ratio=abs(cv2.contourArea(transformed))/(g.shape[0]*g.shape[1])
    if n<16 or frac<.45 or area<.12 or not .15<ratio<6 or not cv2.isContourConvex(transformed):return result
    lost,warped=coverage(ref,H,g.shape)
    fg=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,12)
    nearby=cv2.dilate(fg,np.ones((7,7),np.uint8))
    unmatched=float(np.sum((warped>0)&(nearby==0))/max(1,np.sum(warped>0)))
    result.update(registration='REGISTERED_PROXY',reference_foreground_outside_fraction=lost,unmatched_foreground_fraction=unmatched)
    return result
def esc(s):return html.escape(str(s),quote=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False);(out/'references').mkdir()
    snapshot={};
    def load(p):
        p=Path(p);b=p.read_bytes();snapshot[str(p)]=digest(b);return b
    # Reuse existing inventory; never rescan dataset directories.
    for p in [AUDIT/'manifest.csv',AUDIT/'source_split.csv',AUDIT/'source_image_inventory.json',AUDIT/'annotation_issues.json']:load(p)
    rows=readcsv(AUDIT/'manifest.csv');sources=readcsv(AUDIT/'source_split.csv');inventory=json.loads((AUDIT/'source_image_inventory.json').read_text());refs={};synthetics={};refmeta=[]
    for s in sources:
        source=s['source_id'];original=next((r for r in rows if r['source_id']==source and r['augmentation_type']=='original_photo'),None)
        if original:
            origin=original['image_path'];b=load(origin);relationship='archive original_photo; source lineage requires verification'
        else:
            candidate=next((x for x in inventory[source] if Path(x['origin']).suffix.lower() in ['.jpg','.jpeg']),None)
            if not candidate:raise ValueError('no inventoried handwritten reference for '+source)
            origin=candidate['origin'];archive,member=origin.split('!',1)
            with zipfile.ZipFile(archive) as z:b=z.read(member)
            if digest(b)!=candidate['sha256']:raise ValueError('inventoried source changed')
            snapshot[origin]=digest(b);relationship='inventoried remaining.zip photograph; parent relationship provisional'
        ref=decode(b);g=gray(ref);refs[source]=(g,features(g),metrics(ref));dest=out/'references'/(source+'_handwriting'+Path(origin).suffix.lower());dest.write_bytes(b)
        synthetic=next((x for x in inventory[source] if x['origin'].lower().endswith('.png') and '!' not in x['origin']),None)
        syn=''
        if synthetic:
            sb=load(synthetic['origin']);syn=str(out/'references'/(source+'_synthetic.png'));Path(syn).write_bytes(sb)
        synthetics[source]=syn;refmeta.append({'source_id':source,'handwriting_reference':str(dest),'reference_origin':origin,'relationship':relationship,'synthetic_reference':syn,'split':s['split']})
    results=[]
    for index,r in enumerate(rows):
        source=r['source_id'];flags=[];failure='';m={};c={};b=load(r['image_path']);a=None
        if digest(b)!=r['sha256']:raise ValueError('audit image hash changed: '+r['image_path'])
        try:
            a=decode(b);m=metrics(a)
            if objective(a):failure='SPATIALLY_CONSTANT_BLANK'
            g=gray(a);ref,rf,rm=refs[source]
            c=compare(ref,g,rf,features(g))
            if m['laplacian_variance']<20 or (m['laplacian_variance']<80 and m['laplacian_variance']/max(1,rm['laplacian_variance'])<.15):flags.append('SEVERE_BLUR_PROXY')
            if m['contrast_p95_p5']<20:flags.append('VERY_LOW_CONTRAST_PROXY')
            if m['near_extreme_fraction']>.70:flags.append('SATURATION_OR_BACKGROUND_PROXY')
            if 'crop' in r['augmentation_type']:flags.append('PARTIAL_CROP_RECIPE_RISK')
            if any(x in r['augmentation_type'] for x in ['perspective','rotation','mixed_hard']):flags.append('GEOMETRIC_RECIPE_RISK')
            if c['registration']=='UNRESOLVED':flags.append('PARENT_REGISTRATION_UNRESOLVED')
            else:
                if c['reference_foreground_outside_fraction']>.01:flags.append('LIKELY_CLIPPING_PROXY')
                if c['unmatched_foreground_fraction']>.20:flags.append('MISSING_OR_UNREADABLE_FOREGROUND_PROXY')
        except Exception as e:
            if a is None:failure='UNREADABLE_OR_CORRUPT:'+str(e)
            else:flags.append('ANALYSIS_UNRESOLVED:'+str(e))
        if failure:flags.append(failure)
        result=dict(r,**m,**c,automatic_flags=';'.join(flags),quality_result='OBJECTIVE_EXCLUDE' if failure else ('FLAGGED' if flags else 'PASS_AUTOMATED_CHECKS'),decision='OBJECTIVE_EXCLUDE' if failure else 'NEEDS_REVIEW',verified_transcription_error='NOT_ESTABLISHED')
        results.append(result)
        if (index+1)%100==0:print('Screened',index+1,flush=True)
    fields=list(dict.fromkeys(k for r in results for k in r));csvout(out/'automated_quality_report.csv',results,fields);csvout(out/'flagged_images.csv',[r for r in results if r['automatic_flags']],fields);csvout(out/'reference_mapping.csv',refmeta)
    queue=[]
    for s in sources:
        queue.append({'priority':0,'review_scope':'ISOLATED_TEST' if s['split']=='test' else 'DEVELOPMENT','review_type':'SOURCE_LABEL_AND_PROVENANCE','source_id':s['source_id'],'split':s['split'],'image_path':next(x['handwriting_reference'] for x in refmeta if x['source_id']==s['source_id']),'reason':'verify original source against synthetic rendering/TXT: every tier, symbol and count; confirm lineage','decision':'NEEDS_REVIEW','reviewer':'','notes':''})
    for r in results:
        if r['decision']=='OBJECTIVE_EXCLUDE':continue
        queue.append({'priority':1 if any(x in r['automatic_flags'] for x in ['CLIPPING','MISSING','PARTIAL_CROP']) else (2 if r['automatic_flags'] else 3),'review_scope':'ISOLATED_TEST' if r['split']=='test' else 'DEVELOPMENT','review_type':'AUGMENTATION_CONTENT' if r['augmentation_type']!='original_photo' else 'ORIGINAL_IMAGE','source_id':r['source_id'],'split':r['split'],'image_path':r['image_path'],'reason':r['automatic_flags'] or 'quality checks pass; human content verification still required','decision':'NEEDS_REVIEW','reviewer':'','notes':''})
    queue.sort(key=lambda q:(q['review_scope']=='ISOLATED_TEST',q['priority'],q['source_id'],q['image_path']));csvout(out/'prioritized_review_queue.csv',queue)
    def dashboard(test=False):
        selected=[s for s in sources if (s['split']=='test')==test]
        parts=['<!doctype html><meta charset="utf-8"><title>Stage 2G review</title><style>body{font:16px system-ui;margin:24px;background:#f2f4f7}article{background:white;padding:18px;margin:24px 0}img{max-width:100%;max-height:440px;object-fit:contain} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}figure{margin:0;border:1px solid #ccc;padding:10px}pre{white-space:pre-wrap;font:18px monospace}summary{cursor:pointer;padding:12px}b{color:#912d16}</style>']
        parts+=['<h1>'+('Isolated test data review' if test else 'Train/validation handwritten review')+'</h1><p>No automatic acceptance. Flags are image-analysis proxies, not verified transcription errors. Review source labels first, then priority 1 augmentations. Click images for full resolution. Decisions must be recorded in a copy of the CSV queue.</p>']
        if test:parts+=['<b>Held-out test data: annotation/provenance QA only by an independent custodian. Do not use these images, scores or labels to tune models, screening thresholds or development choices.</b>']
        for s in selected:
            source=s['source_id'];meta=next(x for x in refmeta if x['source_id']==source);rr=[r for r in results if r['source_id']==source];rr.sort(key=lambda r:('CLIPPING' not in r['automatic_flags'] and 'PARTIAL_CROP' not in r['automatic_flags'] and 'MISSING' not in r['automatic_flags'],not bool(r['automatic_flags']),r['augmentation_type']))
            text=load(AUDIT/'annotations'/(source+'.txt')).decode('utf-8-sig');load(AUDIT/'annotations'/(source+'.json'))
            def img(p,label):
                return '<figure><figcaption>'+esc(label)+'</figcaption><a target="_blank" href="'+esc(Path(p).as_uri())+'"><img loading="lazy" src="'+esc(Path(p).as_uri())+'"></a></figure>'
            parts+=['<article><p><a href="'+esc((AUDIT/'review'/(source+'.jpg')).as_uri())+'">Existing audit contact sheet</a></p><h2>'+esc(source)+' — '+esc(s['split'])+'</h2><b>Priority 0: verify source-label correctness</b><div class="grid">',img(meta['handwriting_reference'],'Handwritten reference: '+meta['relationship'])]
            parts+=[img(meta['synthetic_reference'],'Synthetic annotation reference — NOT handwriting') if meta['synthetic_reference'] else '<figure>Synthetic raster unavailable in existing inventory. Use TXT and JSON; none fabricated.</figure>','<figure><h3>TXT transcription (spaces preserved)</h3><pre>'+esc(text)+'</pre><a href="'+esc((AUDIT/'annotations'/(source+'.json')).as_uri())+'">JSON annotation</a></figure></div>','<details><summary>All '+str(len(rr))+' associated images — high risk first</summary><div class="grid">']
            for r in rr:parts.append(img(r['image_path'],r['augmentation_type']+' | '+r['quality_result']+' | '+(r['automatic_flags'] or 'No automatic flags')+' | '+r['decision']))
            parts+=['</div></details></article>']
        return '\n'.join(parts)
    (out/'review_dashboard.html').write_text(dashboard(),encoding='utf-8');(out/'isolated_test_review.html').write_text(dashboard(True),encoding='utf-8')
    # Verify every read input, including ZIP members, remained unchanged.
    for origin,h in snapshot.items():
        if '!' in origin:
            p,n=origin.split('!',1)
            with zipfile.ZipFile(p) as z:b=z.read(n)
        else:b=Path(origin).read_bytes()
        if digest(b)!=h:raise ValueError('input changed during screening: '+origin)
    (out/'input_integrity.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
    summary={'total_images':len(results),'quality_results':dict(Counter(r['quality_result'] for r in results)),'human_verification_images':sum(r['decision']=='NEEDS_REVIEW' for r in results),'accepted_images':0,'objectively_excluded_images':sum(r['decision']=='OBJECTIVE_EXCLUDE' for r in results),'source_label_reviews':30,'source_label_corrections':'NOT_ESTABLISHED; exact TXT/JSON consistency passed prior audit, physical source-label correctness unverified','source_split':dict(Counter(s['split'] for s in sources)),'automatic_flags':dict(Counter(f for r in results for f in r['automatic_flags'].split(';') if f))}
    (out/'readiness_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    report='''# Stage 2G automated screening and readiness

Existing audit reused; no discovery repeated. Original audit and E1/Stage 2F remain unchanged. No OCR, learned models, training, fine-tuning, Git or protected-test model evaluation.

## Results
```json
'''+json.dumps(summary,indent=2)+'''
```

PASS_AUTOMATED_CHECKS is a quality-screen result, never ACCEPT. Only decode/structure failure or spatially constant images (each color channel range <=1) are objectively excluded. Blur, contrast, clipping, registration and unreadability proxies are heuristic NEEDS_REVIEW flags. No verified transcription errors or required original-label corrections have been established. All 30 source-to-label mappings still need human verification; the prior exact TXT/JSON agreement is not proof that handwriting matches those labels.

## Methods and thresholds
Pillow structure verification and full decode, OpenCV decode. Metrics computed at <=900px longest side. Blur flag: Laplacian variance <20, or <80 with <0.15 of source reference variance. Contrast flag: grayscale p95-p5 <20. Extreme-pixel flag: >70% pixels <=2 or >=253 (can be legitimate white paper). These are documented conservative triage thresholds, not calibrated acceptance criteria.

ORB features, ratio test 0.72, RANSAC 3px homography; require >=16 inliers, >=45% inlier fraction, >=12% reference spatial feature coverage, convex mapped corners and mapped area ratio 0.15–6. Otherwise parent registration is unresolved, not a content error. Foreground proxy is adaptive Gaussian threshold (31px/C12), including notebook ruling/shadow. >1% reference foreground mapped outside image flags likely clipping; >20% mapped foreground unmatched within a 7x7 dilation flags likely missing/unreadable foreground. These cannot identify Talim symbols, recover clipped material, distinguish handwriting from ruling or prove exact parent identity. Wrong-parent/repeated-feature registration remains possible. Partial-crop and geometry recipe names add risk flags only. A comparison against an already cropped reference cannot detect content lost before that reference was captured.

20 archive original_photo PNGs are reused as photographed references. For the other 10 sources, existing source_image_inventory.json identifies remaining.zip JPG references; extracted copies are stored separately. Reference mapping is explicit and provisional, not an authenticated parent chain. Synthetic reference PNGs are copied only where present in the existing inventory; unavailable references are labelled unavailable. No raster rendering has been invented. TXT/JSON remain untouched.

## Review and split isolation
Priorities: 0 source-label/provenance check; 1 likely clipping/missing foreground/partial crop; 2 other quality/registration/geometry flags; 3 quality-pass images. Queue has 30 source review tasks plus per-image tasks for non-excluded images. All image decisions require a human; no automatic acceptance. Source approval can be reused to avoid repeatedly comparing the same label for every augmentation, but each derivative still needs content-preservation checking. Source-label correction requires observed, documented discrepancy, not an automatic proxy.

The existing 24/3/3 source split is preserved verbatim. Main dashboard includes train/validation only. Test images are in a separate isolated_test_review.html, not linked or embedded in the development dashboard. Queue marks ISOLATED_TEST rows and places them after development rows; aggregate quality CSVs contain split markers. A separate custodian may perform test annotation/provenance QA using the isolated page. Do not use held-out images/metrics to develop models or tune thresholds. No writer-independent evaluation is claimed.

Input hashes were checked unchanged after screening; input_integrity.json records them. Review all flags in automated_quality_report.csv / flagged_images.csv; original metrics and evidence are preserved. No images are deleted. The dataset is screening-complete, human-validation-pending, NOT ready for automatic E1 fine-tuning. Run test_screening.py for lightweight algorithm and output checks.
'''
    (out/'dataset_readiness_report.md').write_text(report,encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
