"""Validation-only baseline adapter. Segmentation never receives labels."""
import sys,csv,json,hashlib
from pathlib import Path
import cv2,numpy as np
sys.dont_write_bytecode=True
PROJECT=Path(r'C:\NAQSHKASH');OUT=PROJECT/'stage2/stage2g_e1_handwritten_baseline_complete_20261108'
AUDIT=PROJECT/'stage2/stage2g_handwritten_dataset_preparation_20261108'
IDS=('all_003191','all_003556','all_009197')
sys.path[:0]=[str(PROJECT),str(PROJECT/'stage2/stage2e_finetuning/src'),str(PROJECT/'stage2/stage2d_experiments')]
def handwritten_candidates(bgr):
    """Spatial candidates only, no certified roles or semantic alignment."""
    h,w=bgr.shape[:2];blue=bgr[:,:,0].astype(float)-bgr[:,:,2].astype(float)
    if np.percentile(blue,99)>18:
        mask=(blue>max(18,float(np.percentile(blue,90)))) .astype(np.uint8)*255
        method='blue_chromatic_ink'
    else:
        g=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        mask=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,51,12);method='local_dark_ink'
    n,cc,st,_=cv2.connectedComponentsWithStats(mask)
    comps=[];filtered=np.zeros_like(mask)
    for i,(x,y,cw,ch,area) in enumerate(st[1:],1):
        if area<max(8,h*w*.000025) or (cw>w*.35 and cw/ch>10) or ch>h*.6 or cw>w*.8:continue
        if x<=1 or y<=1 or x+cw>=w-1 or y+ch>=h-1:continue
        comps.append((int(x),int(y),int(cw),int(ch)));filtered[cc==i]=255
    if not comps:return [],filtered,{'method':method,'failure':'no_components'}
    scale=float(np.percentile([c[3] for c in comps],95))
    large=sorted([c for c in comps if c[3]>=max(12,.55*scale) and c[2]>=.35*scale],key=lambda c:c[1]+c[3]/2)
    groups=[]
    for c in large:
        cy=c[1]+c[3]/2
        if groups and abs(cy-np.median([v[1]+v[3]/2 for v in groups[-1]]))<.85*scale:groups[-1].append(c)
        else:groups.append([c])
    candidates=[];diag=[]
    for row,g in enumerate(groups):
        top=min(c[1] for c in g);bottom=max(c[1]+c[3] for c in g)
        center=(top+bottom)/2
        prev=(max(v[1]+v[3] for v in groups[row-1])+top)/2 if row else 0
        nxt=(bottom+min(v[1] for v in groups[row+1]))/2 if row+1<len(groups) else h
        gx0=min(c[0] for c in g);gx1=max(c[0]+c[2] for c in g)
        associated=[c for c in comps if prev<=c[1]+c[3]/2<nxt and gx0-.03*w<=c[0]+c[2]/2<=gx1+.03*w]
        x0=max(0,min(c[0] for c in associated)-6);x1=min(w,max(c[0]+c[2] for c in associated)+6)
        for role,a,b in [('upper',int(prev),top),('count',top,bottom),('lower',bottom,int(nxt))]:
            ys,xs=np.where(filtered[a:b,x0:x1]>0)
            if len(ys)<4:continue
            box=[max(0,x0+int(xs.min())-4),max(a,a+int(ys.min())-4),min(w,x0+int(xs.max())+5),min(b,a+int(ys.max())+5)]
            candidates.append({'row':row,'role':role,'box':box,'alignment':'UNVERIFIED_IMAGE_ONLY_CANDIDATE'})
        diag.append({'row':row,'count_band':[top,bottom],'component_count':len(g),'row_region':[int(prev),int(nxt)]})
    return candidates,filtered,{'method':method,'component_scale':scale,'groups':diag,'warning':'large-component heuristic; may mix tier marks or clip count descenders; image-only candidates require visual verification'}
def writecsv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
def main():
    import context as ctx
    import pipeline_d1
    ctx.seed();checkpoint=PROJECT/'stage2/stage2e_finetuning/checkpoints/selected_model/best.pt'
    expected='27f2da50795e2fca8f327afefdce10645f3e60e5320060a8cee93994fabdcdf1'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==expected
    model,vocab,_=ctx.model_load(checkpoint);model.eval()
    from data.collate import resize_preserve_aspect_ratio
    from data.ctc_decode import greedy_ctc_decode
    import torch
    for folder in ['segmentation_previews','error_examples']:(OUT/folder).mkdir(exist_ok=True)
    refs=list(csv.DictReader((PROJECT/'stage2/stage2g_quality_screening_20261108/reference_mapping.csv').open(encoding='utf-8-sig')))
    config=json.loads((PROJECT/'stage2/stage2d_experiments/configs/D1.json').read_text())
    predictions=[];pages=[]
    for source in IDS:
        r=next(r for r in refs if r['source_id']==source);assert r['split']=='validation'
        bgr=cv2.imread(r['handwriting_reference']);gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        d1=pipeline_d1.process_image(gray,config)
        candidates,ink,diag=handwritten_candidates(bgr)
        preview=bgr.copy()
        for t in candidates:
            x0,y0,x1,y1=t['box'];color={'upper':(0,130,0),'count':(0,0,255),'lower':(255,0,0)}[t['role']]
            cv2.rectangle(preview,(x0,y0),(x1,y1),color,2);cv2.putText(preview,f"r{t['row']} {t['role']}",(x0,max(16,y0)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
        cv2.imwrite(str(OUT/'segmentation_previews'/f'{source}_candidates.jpg'),preview)
        cv2.imwrite(str(OUT/'segmentation_previews'/f'{source}_ink.png'),ink)
        dp=cv2.cvtColor(d1['image'],cv2.COLOR_GRAY2BGR)
        for rr in d1['rows']:
            for t in rr:
                x0,y0,x1,y1=t['box'];cv2.rectangle(dp,(x0,y0),(x1,y1),(0,0,255),2);cv2.putText(dp,f"r{t['row']} {t['role']}",(x0,max(14,y0)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
        cv2.imwrite(str(OUT/'segmentation_previews'/f'{source}_D1.jpg'),dp)
        # Labels enter only AFTER image-only coordinates are finalized.
        annotation=json.loads((AUDIT/'annotations'/(source+'.json')).read_text())
        truth={(i,role):row[key].lstrip(' ') for i,row in enumerate(annotation['logical_rows']) for role,key in [('upper','upper'),('count','count_line'),('lower','lower')] if row[key]}
        pages.append({'source_id':source,'input':r['handwriting_reference'],'expected_nonempty_tiers':len(truth),'d1_accepted':d1['accepted'],'d1_reason':d1['reason'],'d1_rows':d1['rows'],'d1_diagnostics':d1['diagnostics'],'candidate_tiers':candidates,'adaptation_diagnostics':diag})
        if not candidates:predictions.append({'source_id':source,'row':'','role':'','box':'','prediction':'','candidate_expected':'','status':'NO_VALID_TIERS','scored':False,'inference_error':'','crop_path':''})
        for t in candidates:
            x0,y0,x1,y1=t['box'];crop=gray[y0:y1,x0:x1];name=f"{source}_r{t['row']}_{t['role']}.png";path=OUT/'error_examples'/name;cv2.imwrite(str(path),crop)
            error='';prediction=''
            try:
                tensor=resize_preserve_aspect_ratio(ctx.tensor(crop)).unsqueeze(0)
                with torch.inference_mode():
                    logits=model(tensor);assert torch.isfinite(logits).all();prediction=greedy_ctc_decode(logits.argmax(2)[:,0].tolist(),vocab)
            except Exception as exc:error=repr(exc)
            predictions.append({'source_id':source,'row':t['row'],'role':t['role'],'box':json.dumps(t['box']),'prediction':prediction,'candidate_expected':truth.get((t['row'],t['role']),''),'status':'INFERENCE_FAILED' if error else 'UNVERIFIED_ALIGNMENT','scored':False,'inference_error':error,'crop_path':str(path)})
        print(source,'candidates',len(candidates),'D1',d1['accepted'],d1['reason'],flush=True)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==expected
    writecsv(OUT/'predictions.csv',predictions)
    (OUT/'segmentation_previews'/'coordinates.json').write_text(json.dumps(pages,indent=2),encoding='utf-8')
    summary={'status':'PROVISIONAL_ALIGNMENT_UNVERIFIED','checkpoint':str(checkpoint),'checkpoint_sha256':expected,'source_photographs_attempted':3,'source_photographs_with_candidate_inference':len({r['source_id'] for r in predictions if r['status']=='UNVERIFIED_ALIGNMENT'}),'sources_scored':0,'expected_nonempty_tiers':sum(p['expected_nonempty_tiers'] for p in pages),'segmented_candidate_tiers':sum(len(p['candidate_tiers']) for p in pages),'verified_segmented_tiers':0,'successfully_inferred_candidate_tiers':sum(r['status']=='UNVERIFIED_ALIGNMENT' for r in predictions),'ocr_inference_failures':sum(r['status']=='INFERENCE_FAILED' for r in predictions),'tiers_scored':0,'cer':None,'exact_tier_accuracy':None,'substitutions':None,'deletions':None,'insertions':None,'per_source':[{'source_id':p['source_id'],'expected_tiers':p['expected_nonempty_tiers'],'candidate_tiers':len(p['candidate_tiers']),'d1_rejected':not p['d1_accepted'],'d1_reason':p['d1_reason'],'cer':None,'exact_tier_accuracy':None,'scoring_exclusion':'candidate tier boundaries/role and physical label correspondence require verification'} for p in pages]}
    (OUT/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
