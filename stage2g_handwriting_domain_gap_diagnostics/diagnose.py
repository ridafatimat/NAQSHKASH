"""Inference-only frozen E1 domain-gap diagnosis. Writes only to OUT."""
import sys,json,csv,hashlib,re,math,collections,platform,time,zipfile
from pathlib import Path
sys.dont_write_bytecode=True
P=Path('C:/NAQSHKASH');PREV=P/'stage2/stage2g_final_handwritten_e1_20261109'
OUT=P/'stage2/stage2g_handwriting_domain_gap_diagnostics'
ROOT=PREV/'dataset/naqshkash_stage2g_handwritten_release_v1'
sys.path[:0]=[str(P),str(P/'stage2/stage2e_finetuning/src')]
import context as c
import metrics as em
from pipeline_d1 import process_image
from PIL import Image,ImageDraw
np=c.np;cv=c.cv2;torch=c.torch
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(name,rows):
 assert rows,name
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
def ns(s):return re.sub(r'\s+','',s)
def bbox(mask):
 y,x=np.where(mask)
 return (int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1) if len(x) else None
def events(a,b):
 d=np.zeros((len(a)+1,len(b)+1),dtype=np.int32);d[:,0]=np.arange(len(a)+1);d[0]=np.arange(len(b)+1)
 for i in range(1,len(a)+1):
  for j in range(1,len(b)+1):d[i,j]=d[i-1,j-1] if a[i-1]==b[j-1] else min(d[i-1,j-1],d[i-1,j],d[i,j-1])+1
 out=[];i=len(a);j=len(b)
 while i or j:
  if i and j and a[i-1]==b[j-1]:i-=1;j-=1
  elif i and j and d[i,j]==d[i-1,j-1]+1:out.append(('substitution',a[i-1],b[j-1]));i-=1;j-=1
  elif i and d[i,j]==d[i-1,j]+1:out.append(('deletion',a[i-1],''));i-=1
  else:out.append(('insertion','',b[j-1]));j-=1
 return out[::-1]
def aggregate(rr):
 chars=sum(len(r['truth']) for r in rr);nc=sum(len(ns(r['truth'])) for r in rr)
 counts={k:sum(r[k] for r in rr) for k in ['substitutions','deletions','insertions']};no={k:sum(r['no_space_'+k] for r in rr) for k in counts}
 return dict(tiers=len(rr),characters=chars,cer=sum(counts.values())/chars,cer_no_whitespace=sum(no.values())/nc,exact_tiers=sum(r['truth']==r['prediction'] for r in rr),exact_accuracy=sum(r['truth']==r['prediction'] for r in rr)/len(rr),**counts,no_space_substitutions=no['substitutions'],no_space_deletions=no['deletions'],no_space_insertions=no['insertions'],mean_blank_probability=float(np.mean([r['blank_probability_mean'] for r in rr])),mean_blank_argmax_fraction=float(np.mean([r['blank_argmax_fraction'] for r in rr])),empty_predictions=sum(not r['prediction'] for r in rr),mean_entropy=float(np.mean([r['entropy_mean_nats'] for r in rr])),mean_target_ctc_nll_per_char=float(np.mean([r['target_ctc_nll_per_char'] for r in rr])),insufficient_ctc_samples=sum(not r['ctc_feasible'] for r in rr),repeated_path_frames=sum(r['repeated_nonblank_path_frames'] for r in rr),collapsed_repeat_characters=sum(r['adjacent_repeated_decoded_chars'] for r in rr))
def geometry(im,truth):
 h,w=im.shape;mask=im<250;box=bbox(mask);x0,y0,x1,y1=box
 n,l,st,_=cv.connectedComponentsWithStats(mask.astype(np.uint8),8);st=st[1:];st=st[st[:,4]>=2]
 dist=cv.distanceTransform(mask.astype(np.uint8),cv.DIST_L2,5);ridge=mask&(dist>=cv.dilate(dist,np.ones((3,3),np.uint8))-.01)
 ink=im[mask];bg=im[~mask];resized=round(w*64/h)
 return dict(height=h,width=w,aspect_ratio=w/h,ink_span_height_fraction=(y1-y0)/h,median_component_height_fraction=float(np.median(st[:,3])/h) if len(st) else 0,median_component_height_e1_pixels=float(np.median(st[:,3])*64/h) if len(st) else 0,nonspace_characters=len(ns(truth)),characters_per_100_e1_pixels=len(ns(truth))*100/resized,ink_fraction=float(mask.mean()),stroke_width_native_proxy=float(np.median(2*dist[ridge])) if ridge.any() else 0,stroke_width_e1_proxy=float(np.median(2*dist[ridge])*64/h) if ridge.any() else 0,ink_intensity_median=float(np.median(ink)),ink_intensity_p90=float(np.percentile(ink,90)),background_intensity_median=float(np.median(bg)) if len(bg) else 255,background_contrast=float((np.median(bg) if len(bg) else 255)-np.median(ink)),faint_ink_fraction=float(np.mean(ink>200)),top_padding_fraction=y0/h,bottom_padding_fraction=(h-y1)/h,left_padding_fraction=x0/w,right_padding_fraction=(w-x1)/w,e1_resized_width=resized,target_length=len(truth),required_ctc_steps=len(truth)+sum(a==b for a,b in zip(truth,truth[1:])),nominal_ctc_steps=resized//4,ctc_steps_per_target=(resized//4)/len(truth))
def beam_decode(prob,v,width=10):
 # Prefix beam without language model. Diagnostic only; all primary runs remain greedy.
 beams={(): (0.,-np.inf)}
 def add(d,key,slot,value):
  prev=d.setdefault(key,[-np.inf,-np.inf]);prev[slot]=float(np.logaddexp(prev[slot],value))
 for frame in np.log(np.clip(prob,1e-30,1)):
  new={}
  for prefix,(pb,pnb) in beams.items():
   total=float(np.logaddexp(pb,pnb));add(new,prefix,0,total+frame[0])
   for k in range(1,len(frame)):
    if prefix and prefix[-1]==k:
     add(new,prefix,1,pnb+frame[k]);add(new,prefix+(k,),1,pb+frame[k])
    else:add(new,prefix+(k,),1,total+frame[k])
  beams=dict(sorted(new.items(),key=lambda kv:np.logaddexp(*kv[1]),reverse=True)[:width])
 return v.decode(list(max(beams,key=lambda k:np.logaddexp(*beams[k]))))
def infer(model,v,im,truth,sid,role,domain,variant,beam=False):
 sample=dict(image=c.tensor(im),label_text=truth,sample_id=sid,tier_name=role)
 b=c.talim_collate_fn([sample]);enc=c.CTCLabelEncoder(v).encode_batch(b['label_texts'])
 with torch.inference_mode():
  logits=model(b['images']);lengths,_=c.runtime_ctc_lengths(model,b,enc,logits.shape[0]);t=int(lengths[0]);lp=logits[:t,0].float().log_softmax(1);prob=lp.exp().numpy();ids=prob.argmax(1);pred=c.greedy_ctc_decode(ids.tolist(),v)
  loss=torch.nn.functional.ctc_loss(lp[:,None,:],enc['targets'],lengths,enc['target_lengths'],blank=0,reduction='sum',zero_infinity=False)
 required=len(truth)+sum(a==b for a,b in zip(truth,truth[1:]));counter=c.dc.old.edit_counts(truth,pred);noc=c.dc.old.edit_counts(ns(truth),ns(pred))
 safe=re.sub('[^a-zA-Z0-9_-]','_',domain+'__'+variant+'__'+sid);np.savez_compressed(OUT/'posteriors'/(safe+'.npz'),probabilities=prob,argmax_indices=ids)
 targetidx=np.asarray(v.encode(truth),dtype=int);uniq=np.unique(targetidx)
 record=dict(domain=domain,variant=variant,id=sid,role=role,truth=truth,prediction=pred,**counter,**{'no_space_'+k:x for k,x in noc.items()},characters=len(truth),e1_resized_width=int(b['image_widths'][0]),ctc_time_steps=t,target_length=len(truth),ctc_required_steps=required,ctc_feasible=t>=required,ctc_time_target_ratio=t/len(truth),blank_probability_mean=float(prob[:,0].mean()),blank_argmax_fraction=float(np.mean(ids==0)),blank_argmax_frames=int((ids==0).sum()),nonblank_path_frames=int((ids!=0).sum()),repeated_nonblank_path_frames=int(np.sum((ids[1:]==ids[:-1])&(ids[1:]!=0))),nonblank_runs=int(np.sum((ids!=0)&np.r_[True,ids[1:]!=ids[:-1]])),adjacent_repeated_decoded_chars=sum(a==b for a,b in zip(pred,pred[1:])),max_identical_path_run=max(len(list(g)) for _,g in __import__('itertools').groupby(ids.tolist())),entropy_mean_nats=float(-(prob*np.log(np.clip(prob,1e-30,1))).sum(1).mean()),target_class_mass_mean=float(prob[:,uniq].sum(1).mean()),target_ctc_nll=float(loss),target_ctc_nll_per_char=float(loss)/len(truth),blank_only_log_probability=float(lp[:,0].sum()),posterior_file='posteriors/'+safe+'.npz',beam_width10_prediction=beam_decode(prob,v) if beam else '')
 assert np.isfinite(float(loss)) and record['ctc_feasible'];return record
def transformed(im,which,role,target):
 if which=='A_current':return im.copy()
 if which=='B_contrast':
  # Robust foreground-only black point; white background stays white.
  vals=im[im<250];black=float(np.percentile(vals,5));return np.clip((im.astype(np.float32)-black)*255/max(255-black,1),0,255).astype(np.uint8)
 if which=='C_otsu':return cv.threshold(im,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
 if which=='D_dilation':
  # One output-pixel radius at E1 scale. No binarization or contrast change.
  resized=c.resize_preserve_aspect_ratio(c.tensor(im))[0].numpy();base=np.clip(np.round(resized*255),0,255).astype(np.uint8)
  return cv.erode(base,cv.getStructuringElement(cv.MORPH_CROSS,(3,3)),iterations=1)
 x0,y0,x1,y1=bbox(im<255)
 if which=='E_tight_vertical':
  # Preserve all nonwhite pixels, no horizontal crop; pad 10% of ink span, >=2px.
  band=im[y0:y1];pad=max(2,round(.10*(y1-y0)));return cv.copyMakeBorder(band,pad,pad,0,0,cv.BORDER_CONSTANT,value=255)
 if which=='F_glyph_height':
  # Vertical-only mapping to synthetic per-role ink-span fraction, fixed canvas/width.
  h,w=im.shape;want=max(1,min(h-2,round(target[role]*h)));band=cv.resize(im[y0:y1],(w,want),interpolation=cv.INTER_LINEAR);out=np.full_like(im,255);top=(h-want)//2;out[top:top+want]=band;return out
 raise ValueError(which)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 assert not (OUT/'diagnostic_report.md').exists(),'Preserve completed diagnosis; select a new OUT for repeat'
 for name in ['examples','posteriors','controls']:(OUT/name).mkdir(exist_ok=True)
 c.seed(42)
 dataset_zip=Path.home()/'Downloads'/'naqshkash_stage2g_handwritten_release_v1.zip';assert dataset_zip.is_file(),f'Dataset not found: {dataset_zip}'
 assert sha(dataset_zip)==json.loads((PREV/'dataset_import_audit.json').read_text())['dataset_zip_sha256']
 frozen=json.loads((c.E/'reports/selected_checkpoint.json').read_text());cp=Path(frozen['selected_checkpoint']);assert frozen['selected']=='E1' and sha(cp)==frozen['sha256']
 # Snapshot previous experiment and directly used E/F reference artifacts; no writes there.
 protected={str(p):sha(p) for p in PREV.rglob('*') if p.is_file()};protected[str(cp)]=sha(cp)
 for folder in [c.E/'reports',P/'stage2/stage2f_rejection_evaluation_v2']:
  # Prior experiment is fully protected; large Stage2F trees remain unopened for inference.
  if folder==c.E/'reports':
   for p in folder.glob('*'):
    if p.is_file():protected[str(p)]=sha(p)
 dump('protected_input_hashes.json',protected)
 model,v,state=c.model_load(cp);model.eval();model.requires_grad_(False)
 def weight_hash():
  h=hashlib.sha256()
  for k,a in model.state_dict().items():h.update(k.encode());h.update(a.cpu().numpy().tobytes())
  return h.hexdigest()
 wh=weight_hash();manifest=read(ROOT/'validation_manifest.csv');assert len(manifest)==14
 split={r['source_id']:r['split'] for r in read(ROOT/'evidence/split_assignment_authoritative.csv')}
 allowed={'all_003191','all_003556','all_009197'};assert {r['source_id'] for r in manifest}==allowed and all(split[s]=='validation' for s in allowed)
 hand=[];locks={};audit=[];distributions=[];predictions=[];originals={};references={}
 for r in manifest:
  sid=r['tier_id'];assert r['crop_path']==f'validation/tier_crops/{sid}.png' and r['label_path']==f'validation/tier_labels/{sid}.txt'
  assert sha(ROOT/r['crop_path'])==r['crop_sha256'] and sha(ROOT/r['label_path'])==r['label_sha256'];raw=(ROOT/r['label_path']).read_text(encoding='utf-8-sig');assert raw==r['raw_label'] and raw.lstrip(' ')==r['ocr_text']
  im=cv.imread(str(ROOT/r['crop_path']),0);assert im.shape==(int(r['crop_height']),int(r['crop_width']))
  hand.append(dict(id=sid,role=r['tier'],truth=r['ocr_text'],source=r['source_id'],row=int(r['row_index']),im=im));originals[sid]=im
  locks[str(ROOT/r['crop_path'])]=r['crop_sha256'];locks[str(ROOT/r['label_path'])]=r['label_sha256']
 # Exact synthetic references to all 14 handwritten targets, never protected sources.
 cfg=json.loads((c.D/'configs/D1.json').read_text());synthetic=[];rendered=[]
 for s in sorted(allowed):
  folder=P/'synthetic_30_references'/s;pp=folder/(s+'.png');jp=folder/(s+'.json');tp=folder/(s+'.txt')
  for q in [pp,jp,tp]:locks[str(q)]=sha(q)
  image=cv.imread(str(pp),0);data=json.loads(jp.read_text());jprev=json.loads((P/'stage2/stage2g_handwritten_dataset_preparation_20261108/annotations'/(s+'.json')).read_text());assert data['logical_rows']==jprev['logical_rows']
  txt='\n\n'.join('\n'.join(r[k] for k in ['upper','count_line','lower']) for r in data['logical_rows']);assert tp.read_text(encoding='utf-8-sig').rstrip('\n')==txt.rstrip('\n')
  labels,entries=c.dc.reference.clean_reference(image,data['logical_rows']);out=process_image(image,cfg);assess=c.dc.reference.assess(out,labels,entries)
  audit.append(dict(source=s,pipeline='D1',correct_segmentation=assess['correct'],reason=assess['reason'],expected_tiers=len(entries),proposed_tiers=sum(len(r) for r in out['rows'])))
  pitch=round((image.shape[0]-48-(len(data['logical_rows'])-1)*20)/(3*len(data['logical_rows'])))
  masks={}
  for e in entries:
   # Include antialiased pixels as ink for clean renderer control, with slot identity.
   ti=['upper','count','lower'].index(e['role']);y=24+e['row']*(3*pitch+20)+ti*pitch;mask=np.zeros_like(image,dtype=bool);mask[y:y+pitch]=image[y:y+pitch]<255;masks[(e['row'],e['role'])]=mask
  for e in entries:
   sid=f"{s}_row_{e['row']:02d}_{e['role']}";hr=next(r for r in hand if r['id']==sid);assert hr['truth']==e['truth']
   if assess['correct']:
    t=next(x for x in assess['tiers'] if x['row']==e['row'] and x['role']==e['role']);x0,y0,x1,y1=t['box'];im=out['image'][y0:y1,x0:x1].copy()
    synthetic.append(dict(id=sid,role=e['role'],truth=e['truth'],source=s,im=im));references[sid]=im
    audit.append(dict(source=sid,pipeline='D1_tier',correct_segmentation=t['geometry_ok'],reason=f"retention={t['retention']}; contamination={t['contamination']}",expected_tiers=1,proposed_tiers=1))
   same=[bbox(mask) for (row,role),mask in masks.items() if row==e['row']];rx0=min(b[0] for b in same);rx1=max(b[2] for b in same)
   cb=bbox(masks[(e['row'],'count')]);H=cb[3]-cb[1];pix=masks[(e['row'],e['role'])];stroke=2*cv.distanceTransform(pix.astype(np.uint8),cv.DIST_L2,5).max();kernel=max(3,int(stroke))|1
   keep=cv.dilate(pix.astype(np.uint8),cv.getStructuringElement(cv.MORPH_ELLIPSE,(kernel,kernel))).astype(bool);canvas=np.full_like(image,255);canvas[keep]=image[keep];canvas[(canvas>200)&keep&~pix]=255
   b=bbox(pix);px=round(.25*H);py=round(max(6,.15*H));x0=max(0,rx0-px);x1=min(image.shape[1],rx1+px);y0=max(0,b[1]-py);y1=min(image.shape[0],b[3]+py);im=canvas[y0:y1,x0:x1]
   # Require every assigned ink pixel retained and no foreign dark ink; otherwise no CER.
   complete=int(pix[y0:y1,x0:x1].sum())==int(pix.sum());foreign=int(((im<128)&~pix[y0:y1,x0:x1]).sum());ok=complete and foreign==0
   audit.append(dict(source=sid,pipeline='handwriting_convention_oracle',correct_segmentation=ok,reason=f"all assigned pixels retained={complete}; foreign dark pixels={foreign}; H={H}",expected_tiers=1,proposed_tiers=1))
   if ok:rendered.append(dict(id=sid,role=e['role'],truth=e['truth'],source=s,im=im))
   cv.imwrite(str(OUT/'controls'/(sid+'_handwriting_convention.png')),im)
 write('segmentation_control_audit.csv',audit)
 # Additional representative established Stage2E crops: 20 per role, deterministic IDs.
 existing=read(c.E/'manifests/evaluation.csv');standard=[]
 for role in ['upper','count','lower']:
  rr=sorted([r for r in existing if r['cohort']=='clean_historical_regression' and r['role']==role],key=lambda r:r['id'])[:20]
  for r in rr:
   assert sha(r['path'])==r['file_sha256'];locks[r['path']]=r['file_sha256'];standard.append(dict(id=r['id'],role=role,truth=r['truth'],source=r['parent_id'],im=cv.imread(r['path'],0)))
 # Derive height targets from synthetic reference images only, never validation outcomes.
 target={role:float(np.median([geometry(r['im'],r['truth'])['ink_span_height_fraction'] for r in synthetic if r['role']==role])) for role in ['upper','count','lower']}
 dump('configuration.json',dict(status='PROVISIONAL',model_checkpoint=str(cp),sha256=frozen['sha256'],seed=42,torch=torch.__version__,python=sys.version,platform=platform.platform(),no_training=True,decoder='unchanged Stage2E greedy CTC; optional baseline-only width10 prefix beam is separate diagnostic',glyph_height_targets=target,synthetic_convention='Known renderer-slot masks, zero deskew, row-union width, xpad=0.25H, ypad=max(6,0.15H), keep_faint_v1. H=count ink span. Oracle isolation, not a replay of photograph-specific segmentation; no ruling present. Stroke halo kernel=odd max(3,int(2*max distance transform)).',transforms={'A_current':'unchanged release bytes','B_contrast':'foreground p5 blackpoint mapped to 0; white=255','C_otsu':'single global Otsu threshold','D_dilation':'grayscale one-pixel cross erosion at model height64 (dark ink dilation), no threshold','E_tight_vertical':'all nonwhite vertical bbox + 10% span white pad >=2px, full width','F_glyph_height':'vertical ink span remapped to synthetic per-role median fraction on unchanged canvas, no horizontal change'},distribution_note='Ink threshold<250; component area>=2px; stroke width proxy=2*distance on local medial ridges; multipart notation means components are not semantic glyphs. Padding and occupied span are threshold-dependent proxies, not reader verification.'))
 controls={}
 for domain,rr in [('synthetic_standard',standard),('synthetic_paired_D1',synthetic),('synthetic_handwriting_convention',rendered)]:
  results=[]
  for r in rr:
   im=r['im'];distributions.append(dict(domain=domain,variant='original',id=r['id'],role=r['role'],**geometry(im,r['truth'])));pr=infer(model,v,im,r['truth'],r['id'],r['role'],domain,'original');results.append(pr);predictions.append(pr)
  controls[domain]=aggregate(results) if results else None;print('CONTROL',domain,controls[domain],flush=True)
 dump('control_results.json',controls)
 variants=['A_current','B_contrast','C_otsu','D_dilation','E_tight_vertical','F_glyph_height'];ablation=[];rolewise=[];conf=[];example_paths=[]
 for variant in variants:
  results=[]
  for r in hand:
   im=transformed(r['im'],variant,r['role'],target);cv.imwrite(str(OUT/'examples'/(r['id']+'_'+variant+'.png')),im)
   pr=infer(model,v,im,r['truth'],r['id'],r['role'],'handwritten',variant,beam=variant=='A_current');results.append(pr);predictions.append(pr)
   distributions.append(dict(domain='handwritten',variant=variant,id=r['id'],role=r['role'],**geometry(im,r['truth'])))
  ablation.append(dict(variant=variant,**aggregate(results)))
  for role in ['upper','count','lower']:rolewise.append(dict(domain='handwritten',variant=variant,role=role,**aggregate([r for r in results if r['role']==role])))
  print('ABLATION',variant,ablation[-1],flush=True)
 for domain in controls:
  for role in ['upper','count','lower']:
   rr=[r for r in predictions if r['domain']==domain and r['role']==role]
   if rr:rolewise.append(dict(domain=domain,variant='original',role=role,**aggregate(rr)))
 for domain,variant in sorted({(r['domain'],r['variant']) for r in predictions}):
  for role in ['upper','count','lower']:
   cc=collections.Counter(e for r in predictions if r['domain']==domain and r['variant']==variant and r['role']==role for e in events(r['truth'],r['prediction']))
   conf.extend(dict(domain=domain,variant=variant,role=role,operation=op,reference=a,prediction=b,count=n) for (op,a,b),n in cc.most_common())
 write('preprocessing_ablation_results.csv',ablation);write('rolewise_error_analysis.csv',rolewise);write('input_distribution_comparison.csv',distributions);write('predictions_and_ctc_behavior.csv',predictions);write('character_confusions.csv',conf)
 summaries=[]
 features=[k for k in distributions[0] if k not in ['domain','variant','id','role']]
 for domain,variant,role in sorted({(r['domain'],r['variant'],r['role']) for r in distributions}):
  rr=[r for r in distributions if (r['domain'],r['variant'],r['role'])==(domain,variant,role)]
  summaries.append(dict(domain=domain,variant=variant,role=role,n=len(rr),**{k:float(np.median([r[k] for r in rr])) for k in features}))
 write('input_distribution_summary.csv',summaries)
 prior={r['id']:r['prediction'] for r in read(PREV/'baseline_predictions.csv')};baseline=[r for r in predictions if r['domain']=='handwritten' and r['variant']=='A_current'];assert len(baseline)==14 and all(prior[r['id']]==r['prediction'] for r in baseline)
 beamrows=[]
 for r in baseline:
  x=dict(r);x['prediction']=r['beam_width10_prediction'];x.update(c.dc.old.edit_counts(x['truth'],x['prediction']));x.update({'no_space_'+k:z for k,z in c.dc.old.edit_counts(ns(x['truth']),ns(x['prediction'])).items()});beamrows.append(x)
 dump('beam_decoder_diagnostic.json',dict(note='Width10 CTC prefix beam, no LM; separate from fixed greedy ablations, not tuned',overall=aggregate(beamrows),predictions=[{'id':r['id'],'truth':r['truth'],'prediction':r['prediction']} for r in beamrows]))
 # Human-readable before/after panels; all 14, row order fixed across variants.
 for r in hand:
  sheet=Image.new('RGB',(1500,7*120),'#eeeeee');draw=ImageDraw.Draw(sheet)
  panels=[('synthetic_D1',references[r['id']])]+[(variant,cv.imread(str(OUT/'examples'/(r['id']+'_'+variant+'.png')),0)) for variant in variants]
  for i,(label,im) in enumerate(panels):
   pic=Image.fromarray(im).convert('RGB');pic.thumbnail((1480,90));draw.text((10,i*120+3),label+f' native {im.shape[1]}x{im.shape[0]}',fill='black');sheet.paste(pic,(10,i*120+24))
  sheet.save(OUT/'examples'/(r['id']+'_comparison.png'))
 checks=dict(no_training_or_optimizer=True,model_weights_unchanged=weight_hash()==wh,no_parameter_gradients=all(p.grad is None for p in model.parameters()),previous_experiment_and_selected_E1_unchanged=all(sha(p)==h for p,h in protected.items()),read_inputs_unchanged=all(sha(p)==h for p,h in locks.items()),baseline_predictions_exactly_reproduced=True,handwritten_samples_per_ablation=all(r['tiers']==14 for r in ablation),no_protected_test_inference=True,all_ctc_lengths_sufficient=all(r['ctc_feasible'] for r in predictions),all_paired_reference_targets_equal=True)
 assert all(checks.values());dump('integrity_checks.json',checks);dump('input_locks.json',locks)
 sourcefiles=[Path(__file__),P/'data/collate.py',P/'data/ctc_decode.py',P/'models/crnn.py',c.E/'src/context.py',c.D/'pipeline_d1.py',c.D/'configs/D1.json',c.dc.C/'evaluation/reference.py',ROOT/'evidence/scripts/engine.py']
 dump('software_hashes.json',{str(p):sha(p) for p in sourcefiles})
 report(controls,ablation,rolewise,predictions,summaries,audit)
 supplement()
 print('DIAGNOSIS COMPLETE',str(OUT),flush=True)
def report(controls,abl,roles,preds,dist,audit):
 text='# Stage 2G handwriting domain-gap diagnosis — PROVISIONAL\n\nFrozen E1 only. No training, model-weight changes, label edits, protected-test inference or Git operations. All 14 final validation tiers from three sources are fixed. Independent Talim-reader semantic verification remains incomplete.\n\n## Matched preprocessing controls\n\n| Control | Tiers | CER | CER without whitespace | Exact |\n|---|---:|---:|---:|---:|\n'
 for name,m in controls.items():
  if m:text+=f"| {name} | {m['tiers']} | {m['cer']:.4%} | {m['cer_no_whitespace']:.4%} | {m['exact_tiers']}/{m['tiers']} |\n"
 text+='\nThe paired synthetic references have exactly the same source/row/role targets as the 14 handwriting tiers. D1 is the frozen Stage2E page preprocessing; renderer-slot masks independently check retention and contamination. Segmentation is assessed before scoring. The handwriting-convention control uses known synthetic masks and the documented row-union/padding/keep-faint conventions with H=count-ink height and zero deskew. It isolates crop/render geometry and is not an end-to-end replay of unavailable photograph masks or segmentation heuristics. No synthetic reference from protected sources is opened. The 60 additional established Stage2E clean crops are deterministic 20-per-role controls, not a random population estimate.\n\n'
 text+='## Frozen-E1 handwritten ablations\n\n| Transformation | CER | No-whitespace CER | Exact tiers | Mean blank argmax fraction | Mean CTC NLL/target char |\n|---|---:|---:|---:|---:|---:|\n'
 for m in abl:text+=f"| {m['variant']} | {m['cer']:.4%} | {m['cer_no_whitespace']:.4%} | {m['exact_tiers']}/14 | {m['mean_blank_argmax_fraction']:.3f} | {m['mean_target_ctc_nll_per_char']:.3f} |\n"
 text+='\nEach B–F starts independently from A. No transforms are stacked. Fixed frozen weights, vocabulary, normalization, CTC lengths and greedy decoder. Transformation recipes and synthetic-only height targets are in configuration.json. Otsu can remove faint ink; dilation can merge dots; vertical remapping changes glyph aspect. These are diagnostic ablations, not automatically safe production changes.\n\n## Rolewise current-input errors\n\n| Role | Tiers | CER | No-whitespace CER | S / D / I | Exact |\n|---|---:|---:|---:|---:|---:|\n'
 for r in roles:
  if r['domain']=='handwritten' and r['variant']=='A_current':text+=f"| {r['role']} | {r['tiers']} | {r['cer']:.4%} | {r['cer_no_whitespace']:.4%} | {r['substitutions']} / {r['deletions']} / {r['insertions']} | {r['exact_tiers']}/{r['tiers']} |\n"
 text+='\nAll transformations and synthetic controls have rolewise rows in rolewise_error_analysis.csv. Minimum-edit ties use the Stage2E substitution/deletion/insertion preference. Confusion counts are alignment diagnostics, not a Talim semantic judgment.\n\n## CTC evidence\n\n'
 a=[r for r in preds if r['domain']=='handwritten' and r['variant']=='A_current'];s=[r for r in preds if r['domain']=='synthetic_paired_D1']
 for name,rr in [('handwriting A',a),('matched synthetic D1',s)]:
  text+=f"- {name}: time steps {min(r['ctc_time_steps'] for r in rr)}–{max(r['ctc_time_steps'] for r in rr)}; required steps {min(r['ctc_required_steps'] for r in rr)}–{max(r['ctc_required_steps'] for r in rr)}; minimum T/target {min(r['ctc_time_target_ratio'] for r in rr):.2f}; mean blank posterior {np.mean([r['blank_probability_mean'] for r in rr]):.3f}; mean blank argmax fraction {np.mean([r['blank_argmax_fraction'] for r in rr]):.3f}; empty predictions {sum(not r['prediction'] for r in rr)}/{len(rr)}.\n"
 beam=json.loads((OUT/'beam_decoder_diagnostic.json').read_text())['overall'];text+=f"\nAll scored samples have enough true time steps, including repeated-target blank separation. Insufficient CTC length is ruled out for these inputs. Blank dominance alone is expected in CTC and cannot establish deletion causality. Full per-frame 66-class probabilities and argmax paths are saved in posteriors/, alongside entropy, target-class mass, target-sequence negative log likelihood, repeated nonblank frames, run counts and decoded repeats in predictions_and_ctc_behavior.csv. A width-10 prefix beam without a language model gives {beam['cer']:.4%} CER / {beam['cer_no_whitespace']:.4%} no-whitespace CER / {beam['exact_tiers']}/14 exact. This separate untuned decoder check does not change any primary ablation. Beam width is finite and cannot rule out every decoder improvement.\n"
 text+='\n## Distribution measurement definitions\n\ninput_distribution_comparison.csv contains per-image geometry, padding, nonspace-target density, grayscale intensity, faint-ink fraction, contrast, stroke-width proxy, E1 width and nominal CTC feasibility; summary CSV gives per-role medians. Ink is <250; component area >=2 pixels; stroke proxy is twice distance-transform radius at local medial ridges. These proxies depend on threshold and do not equate multipart connected components to semantic glyphs. Blank background contrast is measured on already rendered crops, not original paper photos. Faint values 250–254 are excluded from metric masks but preserved in geometry transforms. E uses the complete <255 vertical span so no retained nonwhite pixel is deliberately cropped.\n\n## Reproduction and integrity\n\nRun `python -B diagnose.py` using the recorded runtime and the existing project paths. Writes are confined to this diagnostics directory. A completed diagnostic_report.md causes refusal to overwrite: for a fresh repeat, copy script to a new isolated output folder and change OUT to that folder. This script reads the exact Downloads release hash and uses its already verified imported final dataset. It never loads earlier handwritten ZIPs. Python dependencies are the existing CPU PyTorch/OpenCV/NumPy/Pillow environment. No install or network access is needed.\n\nintegrity_checks.json records weight-state equality, no parameter gradients, exact reproduction of the previous 14 baseline predictions, sufficient CTC lengths, fixed labels/samples, and before/after hashes for every previous experiment file plus selected E1 and directly used labels/images/software. Source hash and runtime/config records are included. Earlier Stage2E/F files are used read-only; Stage2F images are not loaded.\n'
 (OUT/'diagnostic_report.md').write_text(text,encoding='utf-8')
def supplement():
 """Quantization equivalence and clean dilation safety controls; no model updates."""
 config=json.loads((OUT/'configuration.json').read_text());cp=Path(config['model_checkpoint']);assert sha(cp)==config['sha256']
 c.seed(42);m,v,_=c.model_load(cp);m.eval();m.requires_grad_(False)
 rows=read(OUT/'predictions_and_ctc_behavior.csv');current=[r for r in rows if r['variant']=='A_current'];extra=[]
 for r in current:
  mr=next(x for x in read(ROOT/'validation_manifest.csv') if x['tier_id']==r['id']);im=cv.imread(str(ROOT/mr['crop_path']),0)
  # D's uint8 handoff by itself, no morphological change.
  q=np.clip(np.round(c.resize_preserve_aspect_ratio(c.tensor(im))[0].numpy()*255),0,255).astype(np.uint8)
  extra.append(infer(m,v,q,r['truth'],r['id'],r['role'],'handwritten','Q_resize_quantize_identity'))
 standard={r['id']:r for r in read(c.E/'manifests/evaluation.csv')};baseline=[r for r in rows if r['domain']=='synthetic_standard']
 for r in baseline:
  sr=standard[r['id']];assert sha(sr['path'])==sr['file_sha256'];im=cv.imread(sr['path'],0);im=transformed(im,'D_dilation',r['role'],{})
  extra.append(infer(m,v,im,r['truth'],r['id'],r['role'],'synthetic_standard','D_dilation'))
 write('supplementary_control_predictions.csv',extra)
 q=[r for r in extra if r['variant']=='Q_resize_quantize_identity'];d=[r for r in extra if r['domain']=='synthetic_standard'];qa=aggregate(q);da=aggregate(d)
 check=dict(quantized_handoff_matches_all_14_baseline_predictions=all(a['prediction']==b['prediction'] for a,b in zip(current,q)),quantized_handoff_metrics=qa,synthetic_dilation_metrics=da,checkpoint_unchanged=sha(cp)==config['sha256'])
 assert check['checkpoint_unchanged'];dump('supplementary_control_results.json',check)
 runs=[];posterior=[]
 for r in current+[x for x in rows if x['domain']=='synthetic_paired_D1']:
  z=np.load(OUT/r['posterior_file']);prob=z['probabilities'];ids=z['argmax_indices'];start=0;nonblank=[]
  for k,g in __import__('itertools').groupby(ids.tolist()):
   run=list(g);end=start+len(run)
   if k:
    char=v.decode([k]);nonblank.append((char,start,end));runs.append(dict(domain=r['domain'],id=r['id'],role=r['role'],character=char,start_step=start,end_step_exclusive=end,frames=end-start,mean_character_probability=float(prob[start:end,k].mean()),peak_character_probability=float(prob[start:end,k].max())))
   start=end
  mass=prob[:,1:].sum();ci=v.encode('i')[0]
  repeated=[(a,b) for a,b in zip(nonblank,nonblank[1:]) if a[0]==b[0]]
  posterior.append(dict(domain=r['domain'],id=r['id'],role=r['role'],decoded_i=r['prediction'].count('i'),decoded_length=len(r['prediction']),truth_i=r['truth'].count('i'),i_share_nonblank_probability=float(prob[:,ci].sum()/mass),mean_nonblank_peak_probability=float(np.mean([prob[t,k] for t,k in enumerate(ids) if k])) if (ids!=0).any() else 0,repeated_decoded_pairs=len(repeated),all_repeated_pairs_blank_separated=all(b[1]>a[2] and np.all(ids[a[2]:b[1]]==0) for a,b in repeated)))
 write('ctc_nonblank_runs.csv',runs);write('ctc_repeat_and_class_mass.csv',posterior)
 # Same model-scale panel for representative faint marks and a repeated-count failure.
 for sid in ['all_003191_row_01_count','all_003556_row_00_upper','all_009197_row_00_lower']:
  grid=Image.new('RGB',(1600,6*110),'#eeeeee');draw=ImageDraw.Draw(grid)
  for i,variant in enumerate(['A_current','B_contrast','C_otsu','D_dilation','E_tight_vertical','F_glyph_height']):
   im=cv.imread(str(OUT/'examples'/(sid+'_'+variant+'.png')),0);t=c.resize_preserve_aspect_ratio(c.tensor(im))[0].numpy();im=np.round(t*255).astype(np.uint8);pic=Image.fromarray(im).convert('RGB');pic.thumbnail((1580,64));draw.text((10,i*110+4),variant+f' E1:64x{im.shape[1]}',fill='black');grid.paste(pic,(10,i*110+26))
  grid.save(OUT/'examples'/(sid+'_model_scale.png'))
 write_conclusions()
 protected=json.loads((OUT/'protected_input_hashes.json').read_text());assert all(sha(p)==h for p,h in protected.items())
 checks=json.loads((OUT/'integrity_checks.json').read_text());checks.update(supplementary_model_no_gradients=all(p.grad is None for p in m.parameters()),previous_hashes_reverified_after_supplement=True);dump('integrity_checks.json',checks)
 hashes=json.loads((OUT/'software_hashes.json').read_text());hashes[str(Path(__file__))]=sha(__file__);dump('software_hashes.json',hashes)
 print('SUPPLEMENT',check,flush=True)
def write_conclusions():
 abl=read(OUT/'preprocessing_ablation_results.csv');roles=read(OUT/'rolewise_error_analysis.csv');preds=read(OUT/'predictions_and_ctc_behavior.csv');summ=read(OUT/'input_distribution_summary.csv');extra=json.loads((OUT/'supplementary_control_results.json').read_text())
 a={r['id']:r for r in preds if r['variant']=='A_current'};d={r['id']:r for r in preds if r['variant']=='D_dilation'}
 errors=lambda r:sum(int(r[k]) for k in ['substitutions','deletions','insertions'])
 delta={k:errors(d[k])-errors(a[k]) for k in a};best=min(abl,key=lambda r:float(r['cer']));repeat=read(OUT/'ctc_repeat_and_class_mass.csv');count=[r for r in repeat if r['domain']=='handwritten' and r['role']=='count']
 text='\n## Diagnostic conclusions and next-experiment decision\n\n'
 text+='**Primary diagnosis: an appearance-domain mismatch (stroke shape, thickness and faintness), with a secondary preprocessing contribution; the tested geometry conventions and decoder do not explain most of the failure.** This is an inference from matched controls, not a causal partition of error percentages. Exact-label synthetic references remain at 0% CER under D1 and 3.09% under the documented handwriting-convention emulator, while actual handwriting is 74.23%. Synthetic glyph identity is therefore recognized on these targets; crop/render convention alone produces a much smaller failure. Contrast alone and vertical normalization do not close the gap. Unverified writer copying/target semantics remain a confounder that image-label file agreement cannot resolve. No original handwriting page segmenter is evaluated, so this does not establish end-to-end page segmentation quality.\n\n'
 text+='### Measured appearance and geometry differences\n\n| Role | Synthetic / handwriting median stroke-width proxy at E1 scale | Synthetic / handwriting ink median | Synthetic / handwriting component-height proxy at E1 scale | Synthetic / handwriting resized width |\n|---|---:|---:|---:|---:|\n'
 for role in ['upper','count','lower']:
  s=next(r for r in summ if r['domain']=='synthetic_paired_D1' and r['role']==role);h=next(r for r in summ if r['domain']=='handwritten' and r['variant']=='A_current' and r['role']==role)
  text+=f"| {role} | {float(s['stroke_width_e1_proxy']):.2f} / {float(h['stroke_width_e1_proxy']):.2f} | {float(s['ink_intensity_median']):.0f} / {float(h['ink_intensity_median']):.0f} | {float(s['median_component_height_e1_pixels']):.2f} / {float(h['median_component_height_e1_pixels']):.2f} | {float(s['e1_resized_width']):.1f} / {float(h['e1_resized_width']):.1f} |\n"
 text+='\nStroke widths are geometric proxies, not calibrated pen widths; component heights are not glyph heights because multi-dot/line notation can have several components. Overall vertical occupied fractions and E1 widths are broadly similar on these matched samples, while darkness, local thickness and component morphology differ substantially. Handwriting has larger native pixels but E1 normalizes those away. Its foreground is much lighter; already-rendered backgrounds are essentially white in both domains. Full dimensions, aspect ratios, density and padding are in the distribution CSVs. Shape variants and the persistence of errors after contrast/height changes support an appearance gap; this experiment cannot fully separate shape from grayscale effects or label correctness.\n\n'
 text+=f"### Which preprocessing helps?\n\nDilation is the only tested change with a notable aggregate gain: 74.23% to {float(best['cer']):.2%} CER (72 to 67 edits; five fewer edits), and 88.73% to 80.28% no-whitespace CER. It improves {sum(x<0 for x in delta.values())} tiers, worsens {sum(x>0 for x in delta.values())}, and leaves {sum(x==0 for x in delta.values())} unchanged. Still 0/14 exact; this is exploratory and not evidence of generalization. Its mean blank fraction rises slightly rather than falls, so the improvement cannot be attributed simply to reducing blanks. Contrast yields only one fewer edit and no no-whitespace gain. Otsu, vertical tightening and synthetic-role ink-height normalization yield one more space-sensitive edit than baseline. No combined transforms were tested.\n\n"
 text+=f"The resize/uint8-handoff identity control {'reproduces all 14 baseline predictions' if extra['quantized_handoff_matches_all_14_baseline_predictions'] else 'does not reproduce all baseline predictions; see supplementary control details'}; thus the observed dilation gain is {'not explained by its uint8 handoff on these samples' if extra['quantized_handoff_matches_all_14_baseline_predictions'] else 'potentially confounded by quantization'}. Applying the same dilation to 60 previously perfect synthetic controls yields {extra['synthetic_dilation_metrics']['cer']:.4%} CER and {extra['synthetic_dilation_metrics']['exact_tiers']}/60 exact. Dilation must therefore be calibrated on training data and preserve dots/gaps; it is not automatically a universal inference default.\n\n"
 text+='### Most affected roles and character behavior\n\nCount is worst by primary CER: 92.68% versus upper 61.29% and lower 60.00%. Without whitespace, lower is almost equally poor (92.31%; upper 76.47%), so the lower headline benefits from easier alignment spaces. Repeated count confusion is concentrated on `i`: target a→i five times, d→i four, b→i three and c→i twice under the deterministic alignment. These are OCR-class confusions, not claims about the Talim meaning of handwriting. Whitespace accounts for 26/97 targets; removing it does not remove the domain gap.\n\n'
 text+=f"Across the five count outputs, {sum(int(r['decoded_i']) for r in count)}/{sum(int(r['decoded_length']) for r in count)} decoded characters are i, against {sum(int(r['truth_i']) for r in count)} target i characters. The repeated decoded labels inspected are all separated by blank runs: {all(r['all_repeated_pairs_blank_separated']=='True' for r in repeat)}. ctc_nonblank_runs.csv records their exact positions and posterior confidence. This is repeated character emission by the model, not failure to collapse adjacent identical CTC frames. Original mean nonblank-frame confidence for count is {np.mean([float(r['mean_nonblank_peak_probability']) for r in count]):.3f}; high confidence can still be wrong out of domain.\n\n"
 text+='The model assigns very low probability to the correct handwriting sequence (mean per-tier target CTC NLL/character 8.106, versus the much smaller matched-synthetic control value in control_results.json). All original targets have enough time steps; no outputs are empty. A blank-heavy posterior occurs on correctly recognized synthetic tiers too. The separate width-10 beam check recovers only one edit, preserving 24 aligned deletions, so greedy search is not the main demonstrated bottleneck. This does not prove a cause for each deletion: target probability mass may be spread across wrong identities, positions or blank, and multipart handwriting and copying errors remain possible.\n\n'
 text+='### Is a second fine-tuning run justified now?\n\n**Not yet as another unconstrained fine-tuning attempt.** The evidence is sufficient to design an appearance-focused pilot, but not to certify labels, select a universal preprocessing change, or claim generalization from three validation sources. Before executing it: obtain independent Talim-reader checks of the handwritten count copying and notation targets (prioritize all five validation counts and the 57 training counts, then all upper/lower targets); verify the 165 training-only crops for dilation-induced merging at model scale; calibrate any stroke policy using training images only. Keep the 14 validation labels and sources fixed and the protected test closed. This diagnosis does not create a new training set or authorize a test evaluation.\n\n'
 text+='### Exact next strategy to test after those checks (proposal only)\n\nOne predeclared appearance-adaptation pilot, initialized anew from frozen E1; do not continue epoch 2 of the rejected model. Use only the verified 165 handwritten training tiers, distinguishable synthetic replay, no validation training, seed 42 and a fixed two-epoch ceiling. Preserve original ocr_text/internal whitespace. Use effective batch four (two handwritten plus two replay, gradient accumulation, frozen BatchNorm statistics). Role-stratify handwritten ordering without oversampling count until the first pass covers all 165 tiers.\n\nUse AdamW with weight decay 1e-4 and gradient clipping 1.0; freeze the BiLSTM and classifier for this first appearance-focused probe, and train the CNN at 1e-5. This is a hypothesis derived from the appearance mismatch, not a validated optimal schedule. Use a 50/50 identity versus one-output-pixel cross dilation augmentation for handwritten TRAINING images only if visual QA confirms dots/gaps survive; keep validation inference at unchanged A for the primary comparison and report fixed D as a separate diagnostic. Do not adopt Otsu, vertical warps or contrast+dilation stacks. Replay uses original synthetic tiers; require original synthetic regression cohorts after every epoch. Do not generate new labels or pretend the current threshold proxy yields verified glyph segmentation.\n\nInclude a frozen-E1 distillation anchor on replay: add 0.1 times the mean-over-true-time-step KL divergence KL(teacher probabilities || student probabilities) to CTC loss, temperature 1.0. The unmodified frozen E1 is the teacher; no teacher updates. Keep exact time widths and blank index unchanged. This is a proposed forgetting-control mechanism, not a result of this diagnosis. Checkpoint selection should retain E1 unless handwritten CER improves while every clean cohort increases by <=0.001 absolute, clean exact drops <=0.01, positive-edge CER increases <=0.005 absolute, and long-positive CER increases <=0.005. Report every epoch, role, no-whitespace score and sample exposure; reject any candidate violating a gate. Do not expand epochs or search hyperparameters on the 14 tiers.\n\nNo second training run was started. Frozen E1 remains selected.\n'
 p=OUT/'diagnostic_report.md';existing=p.read_text(encoding='utf-8').split('\n## Diagnostic conclusions and next-experiment decision')[0];p.write_text(existing+text,encoding='utf-8')
if __name__=='__main__':
 if '--supplement-only' in sys.argv:supplement()
 else:main()
