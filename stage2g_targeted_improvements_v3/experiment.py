"""Inference-only Stage 2G V3; no synthetic generation or SGD."""
import sys,json,csv,hashlib,collections,math,importlib.util,re,time
from pathlib import Path
sys.dont_write_bytecode=True
P=Path('C:/NAQSHKASH');V=P/'stage2/stage2g_writer_independent_finetune_v2';O=P/'stage2/stage2g_targeted_improvements_v3'
spec=importlib.util.spec_from_file_location('read_only_v2_helpers',V/'experiment.py');v2=importlib.util.module_from_spec(spec);spec.loader.exec_module(v2)
c=v2.c;torch=c.torch;np=c.np;cv=c.cv2
from talim import talim_rules as rules
from PIL import Image,ImageDraw
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(n,x):
 p=O/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def write(n,rr):
 if not rr:return
 p=O/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def allowed(role):return set(rules.VALID_COUNT_GLYPHS) if role=='count' else set(rules.UP_SYMBOLS if role=='upper' else rules.DOWN_SYMBOLS)|{' '}
def align(a,b):
 d=np.zeros((len(a)+1,len(b)+1),np.int32);d[:,0]=np.arange(len(a)+1);d[0,:]=np.arange(len(b)+1)
 for i in range(1,len(a)+1):
  for j in range(1,len(b)+1):d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(a[i-1]!=b[j-1]))
 out=[];i=len(a);j=len(b)
 while i or j:
  if i and j and a[i-1]==b[j-1]:out.append(('match',i-1,j-1,a[i-1],b[j-1]));i-=1;j-=1
  elif i and j and d[i,j]==d[i-1,j-1]+1:out.append(('substitution',i-1,j-1,a[i-1],b[j-1]));i-=1;j-=1
  elif i and d[i,j]==d[i-1,j]+1:out.append(('deletion',i-1,j,a[i-1],''));i-=1
  else:out.append(('insertion',i,j-1,'',b[j-1]));j-=1
 return out[::-1]
def count_events(rr):
 cc=collections.Counter((op,a,b) for r in rr for op,i,j,a,b in align(r['truth'],r['prediction']) if op!='match')
 return cc
def inputs():
 rows=read(V/'audited_handwritten_inputs.csv');tr=[r for r in rows if r['split']=='train' and r['sample_type']=='original'];va=[r for r in rows if r['split']=='validation' and r['sample_type']=='original'];dev=read(V/'regression_inputs.csv')
 assert len(tr)==108 and len(va)==14 and {r['writer_id'] for r in tr}=={'W01','W03','W04','W06'} and {r['writer_id'] for r in va}=={'W05'}
 assert all(r['source_id']!='all_006767' and not (r['source_id']=='all_003191' and r['row_index']=='1') for r in tr+va)
 for r in tr+va:assert sha(r['path'])==r['sha256'] and Path(r['label_path']).read_text(encoding='utf-8-sig')==r['raw_label']
 for r in dev:assert sha(r['path'])==r['file_sha256']
 return tr,va,dev
def setup():
 O.mkdir(parents=True,exist_ok=True)
 for folder in ['posteriors','visual_failure_examples','candidate_models']:(O/folder).mkdir(exist_ok=True)
 result=load(V/'final_comparison.json');models={k:result['approaches'][k]['checkpoint'] for k in ['A','C','D']};hashes={k:sha(p) for k,p in models.items()}
 assert all(hashes[k]==result['approaches'][k]['sha256'] for k in models)
 protected={str(p):sha(p) for p in V.rglob('*') if p.is_file()};protected[models['A']]=hashes['A']
 dump('protected_previous_hashes.json',protected)
 dump('protocol.json',dict(status='PROVISIONAL',training=False,synthetic_generation=False,models=models,model_hashes=hashes,fixes=['F1 frozen E1 + existing contrast + authentic role-alphabet CTC mask; preserve raw output/flags','F2 inference-only CNN delta shrinkage from V2 D toward E1; existing contrast; alpha grid .25/.5/.75'],calibration='108 training-writer originals and existing synthetic development cohorts only; W05 never selects parameters',gates=load(V/'training_config.json')['gates'],alpha_selection='Among full-synthetic-gate-safe alphas with training macro-writer CER improvement over B, lowest training macro-writer CER; tie prefer smaller alpha. If none pass, reject without W05 tuning.',role_fix_selection='One fixed authentic role mask, no parameter search. Require improvement in training macro-writer CER, no synthetic calibration-control error increase, and all truths supported.',validation='14 W05 original tiers only, no augmented validation',reuse='Saved V2 A/B/C/D and synthetic predictions; only three existing W05 row0 frozen-A posterior traces from the old source-split diagnosis are eligible. Do not load W02/W07 or quarantine traces.',scope='No generic training, AdaBN, image generation, source changes or Git operations. Feature-class correspondences are approved isolated-single-symbol targets, not independently reader-verified.'))
def model(path):
 m,v,s=c.model_load(Path(path));m.eval();m.requires_grad_(False);return m,v,s
IDENTITY={'name':'identity'};CONTRAST={'name':'contrast','contrast':True}
def probability(m,v,r,conf,condition,save=True):
 name=re.sub('[^a-zA-Z0-9_-]','_',condition+'__'+r['id'])+'.npz';path=O/'posteriors'/name
 if path.exists():return np.load(path)['probabilities'],name,'v3_cache'
 if condition=='A' and r['id'].startswith('all_003191_row_00_'):
  old=P/'stage2/stage2g_handwriting_domain_gap_diagnostics/posteriors'/('handwritten__A_current__'+r['id']+'.npz')
  if old.exists():
   prob=np.load(old)['probabilities'];np.savez_compressed(path,probabilities=prob);return prob,name,'existing_verified_W05_posterior'
 t,_=v2.normalize(v2.tensors(r),r['role'],conf)
 with torch.inference_mode():prob=m(t.unsqueeze(0))[:,0].float().softmax(1).numpy()
 assert np.allclose(prob.sum(1),1,atol=1e-5)
 if save:np.savez_compressed(path,probabilities=prob)
 return prob,name,'new_trace_previously_unavailable'
def decode(prob,v,role,restricted=False):
 if restricted:
  ok=[0]+[v.encode(ch)[0] for ch in allowed(role) if ch in v.char_to_index];p=np.zeros_like(prob);p[:,ok]=prob[:,ok]
 else:p=prob
 return c.greedy_ctc_decode(p.argmax(1).tolist(),v)
def result_row(r,pred,prob,v):
 ct=c.dc.old.edit_counts(r['truth'],pred);no=c.dc.old.edit_counts(v2.ns(r['truth']),v2.ns(pred));targets=torch.tensor(v.encode(r['truth']),dtype=torch.long)
 with torch.no_grad():loss=torch.nn.functional.ctc_loss(torch.from_numpy(np.log(np.clip(prob,1e-30,1)))[:,None,:],targets,torch.tensor([len(prob)]),torch.tensor([len(targets)]),blank=0,reduction='mean')
 return dict(id=r['id'],source_id=r.get('source_id',r.get('parent_id','')),writer_id=r.get('writer_id','synthetic'),role=r['role'],truth=r['truth'],prediction=pred,**ct,**{'no_space_'+k:x for k,x in no.items()},edit_distance=sum(ct.values()),loss=float(loss),exact=pred==r['truth'])
def evaluate(m,v,rows,conf,condition,name,restricted=False,save=False):
 rr=[]
 for i,r in enumerate(rows):
  assert set(r['truth'])<=allowed(r['role']),'Truth incompatible with authentic role; do not remap/drop'
  pr,_,_=probability(m,v,r,conf,condition,save=save);record=result_row(r,decode(pr,v,r['role'],restricted),pr,v)
  if restricted:
   raw=decode(pr,v,r['role'],False);record.update(raw_prediction=raw,raw_out_of_role_characters=''.join(ch for ch in raw if ch not in allowed(r['role'])),role_restriction_changed_output=raw!=record['prediction'])
  rr.append(record)
  if (i+1)%250==0:print('EVAL',name,i+1,'/',len(rows),flush=True)
 summary=v2.summary(rr);dump('metrics/'+name+'.json',summary);write('predictions/'+name+'.csv',rr);return summary,rr
def regression(rr,dev):
 assert len(rr)==len(dev)
 out={cohort:v2.agg([r for r,x in zip(rr,dev) if x['cohort']==cohort]) for cohort in sorted({x['cohort'] for x in dev})}
 out['long_positive_development']=v2.agg([r for r,x in zip(rr,dev) if x['cohort']=='positive_development' and len(x['truth'])>=65]);return out
def constrained_alignment(prob,truth,v):
 labels=[0]
 for k in v.encode(truth):labels.extend([k,0])
 labels=np.array(labels);lp=np.log(np.clip(prob,1e-30,1));T,S=len(prob),len(labels);prev=np.full(S,-np.inf);prev[0]=lp[0,0]
 if S>1:prev[1]=lp[0,labels[1]]
 back=np.zeros((T,S),np.int16)
 for t in range(1,T):
  cur=np.full(S,-np.inf)
  for s in range(S):
   choices=[s]
   if s:choices.append(s-1)
   if s>1 and labels[s] and labels[s]!=labels[s-2]:choices.append(s-2)
   k=max(choices,key=lambda j:prev[j]);cur[s]=prev[k]+lp[t,labels[s]];back[t,s]=k
  prev=cur
 s=max([S-1,S-2],key=lambda j:prev[j]);states=np.zeros(T,np.int16)
 for t in range(T-1,-1,-1):states[t]=s;s=int(back[t,s])
 return {i:np.where(states==2*i+1)[0] for i in range(len(truth))}
def emission_runs(prob,v):
 ids=prob.argmax(1);runs=[];start=0
 import itertools
 for k,g in itertools.groupby(ids.tolist()):
  frames=list(g);end=start+len(frames)
  if k:runs.append(dict(character=v.decode([k]),start=start,end=end,peak=start+int(prob[start:end,k].argmax()),class_id=k))
  start=end
 return runs
def spacing(r):
 t=v2.tensors(r)[0].numpy();black=v2.inkinfo(t)['blackpoint'];mask=t<(1+black)/2;x=np.any(mask,axis=0).astype(np.uint8);x=cv.morphologyEx(x[None],cv.MORPH_CLOSE,np.ones((1,2),np.uint8))[0]
 changes=np.diff(np.r_[0,x,0].astype(int));st=np.where(changes==1)[0];en=np.where(changes==-1)[0];centers=(st+en-1)/2;positions=np.array([i for i,ch in enumerate(r['truth']) if ch!=' ']);ok=len(centers)==len(positions) and len(positions)>=2
 residual=float('nan');cvpitch=float('nan')
 if ok:
  fit=np.polyfit(positions,centers,1);residual=float(np.sqrt(np.mean((np.polyval(fit,positions)-centers)**2)));pitch=np.diff(centers)/np.diff(positions);cvpitch=float(pitch.std()/pitch.mean())
 return dict(id=r['id'],domain='handwritten_'+r['split'] if 'writer_id' in r else 'synthetic_development',role=r['role'],target_nonspace=len(positions),detected_horizontal_groups=len(centers),group_count_matches=ok,layout_residual_e1_pixels=residual,local_cell_pitch_cv=cvpitch,qualification='Offline label-position comparison, not verified per-symbol segmentation; mismatched group counts not fitted')
def retention(r):
 im=cv.imread(r['path'],0).astype(np.float32)/255;t=v2.tensors(r)[0].numpy();black=np.percentile(im[im<.98],5);mask=(im<(1+black)/2).astype(np.uint8);n,l,stats,_=cv.connectedComponentsWithStats(mask,8);h,w=im.shape;out=[]
 # At most six smallest meaningful components per crop; threshold/projection proxies only.
 ids=sorted([k for k in range(1,n) if stats[k,4]>=2],key=lambda k:stats[k,4])[:6]
 for k in ids:
  x,y,bw,bh,area=map(int,stats[k]);xx0=max(0,math.floor(x*t.shape[1]/w));xx1=min(t.shape[1],max(xx0+1,math.ceil((x+bw)*t.shape[1]/w)));yy0=max(0,math.floor(y*64/h));yy1=min(64,max(yy0+1,math.ceil((y+bh)*64/h)));roi=t[yy0:yy1,xx0:xx1];before=1-float(im[l==k].min());after=1-float(roi.min())
  out.append(dict(id=r['id'],role=r['role'],writer_id=r['writer_id'],split=r['split'],native_component_area=area,native_component_width=bw,native_component_height=bh,expected_e1_height=bh*64/h,native_peak_darkness=before,e1_roi_peak_darkness=after,below_1_over_255_after=after<1/255,below_original_midcontrast_after=after<(1-black)/2,note='Small-component bbox proxy; neighbouring ink may share ROI. No semantic dot identity or original-photo extraction claim.'))
 return out
def analysis():
 assert not (O/'analysis_complete.json').exists(),'Preserve completed analysis'
 setup();tr,va,dev=inputs();old=load(V/'final_comparison.json');names={'A':'A_frozen_original','B':'B_frozen_normalized','C':'C_raw_epoch_1','D':'D_normalized_epoch_2'};saved={k:read(V/'predictions'/(n+'.csv')) for k,n in names.items()};reg={k:read(V/'predictions'/('regression_'+n+'.csv')) for k,n in names.items()}
 compare=[];errorrows=[];counts=[]
 for i,r in enumerate(va):
  by={k:next(x for x in rr if x['id']==r['id']) for k,rr in saved.items()};assert all(x['truth']==r['truth'] for x in by.values())
  compare.append(dict(id=r['id'],writer_id=r['writer_id'],source_id=r['source_id'],role=r['role'],truth=r['truth'],**{k+'_prediction':x['prediction'] for k,x in by.items()}))
  for k,x in by.items():
   for op,ri,hj,a,b in align(r['truth'],x['prediction']):
    if op!='match':errorrows.append(dict(approach=k,id=r['id'],source_id=r['source_id'],writer_id=r['writer_id'],role=r['role'],operation=op,reference_position=ri,prediction_position=hj,reference=a,predicted=b,whitespace_involved=a.isspace() or b.isspace(),predicted_outside_authentic_role=bool(b and b not in allowed(r['role']))))
  # Actual crop + exact text, monospace-style escaped labels preserve visible spaces.
  sheet=Image.new('RGB',(1800,400),'#eeeeee');draw=ImageDraw.Draw(sheet);im=Image.open(r['path']).convert('RGB');im.thumbnail((1780,120));sheet.paste(im,(10,25));draw.text((10,4),r['id']+' W05',fill='black');draw.text((10,155),'truth: '+repr(r['truth']),fill='black')
  for j,(k,x) in enumerate(by.items()):draw.text((10,195+j*42),k+': '+repr(x['prediction']),fill='black')
  sheet.save(O/'visual_failure_examples'/(r['id']+'.png'))
 for k,rr in saved.items():
  for role in ['upper','count','lower']:
   g=[x for x in rr if x['role']==role];ct=count_events(g)
   counts.extend(dict(approach=k,role=role,operation=op,reference=a,predicted=b,count=n) for (op,a,b),n in ct.most_common())
 write('per_tier_comparison.csv',compare);write('character_error_analysis.csv',errorrows);write('character_confusions.csv',counts)
 regrows=[];changed=[]
 for ra,rb,rc,rd,x in zip(reg['A'],reg['B'],reg['C'],reg['D'],dev):
  assert ra['id']==rb['id']==rc['id']==rd['id']==x['id']
  if any(z['prediction']!=ra['prediction'] for z in [rb,rc,rd]):changed.append(x)
  if ra['prediction']!=rd['prediction'] or ra['prediction']!=rb['prediction']:
   regrows.append(dict(id=x['id'],source_id=x['parent_id'],cohort=x['cohort'],role=x['role'],long_sequence=len(ra['truth'])>=65,truth=ra['truth'],A=ra['prediction'],B=rb['prediction'],C=rc['prediction'],D=rd['prediction'],errors_A=int(ra['edit_distance']),errors_B=int(rb['edit_distance']),errors_C=int(rc['edit_distance']),errors_D=int(rd['edit_distance']),preprocessing_delta=int(rb['edit_distance'])-int(ra['edit_distance']),fixed_normalization_weight_delta=int(rd['edit_distance'])-int(rb['edit_distance']),total_delta=int(rd['edit_distance'])-int(ra['edit_distance']),nonspace_A_D_identical=v2.ns(ra['prediction'])==v2.ns(rd['prediction']),spaces_A=ra['prediction'].count(' '),spaces_B=rb['prediction'].count(' '),spaces_D=rd['prediction'].count(' ')))
 write('regression_failure_analysis.csv',regrows)
 event_delta=[]
 for role in ['upper','count','lower']:
  ac=count_events([r for r in reg['A'] if r['role']==role]);bc=count_events([r for r in reg['B'] if r['role']==role]);dc=count_events([r for r in reg['D'] if r['role']==role])
  event_delta.extend(dict(role=role,operation=key[0],reference=key[1],predicted=key[2],count_A=ac[key],count_B=bc[key],count_D=dc[key],weight_delta=dc[key]-bc[key],total_delta=dc[key]-ac[key]) for key in sorted(ac.keys()|bc.keys()|dc.keys()) if dc[key]!=ac[key] or bc[key]!=ac[key])
 write('regression_character_changes.csv',event_delta)
 # New traces are needed for 11 previously untraced W05 tiers and changed V2 model/normalization conditions.
 posterior_events=[];trace_stats=[];trace_reuse=[];state=load(O/'protocol.json');loaded={}
 for k in ['A','B','C','D','D_raw']:
  key='A' if k=='B' else 'D' if k=='D_raw' else k
  if key not in loaded:loaded[key]=model(state['models'][key])
  m,v,_=loaded[key];conf=CONTRAST if k in ['B','D'] else IDENTITY
  for r in va:
   prob,file,origin=probability(m,v,r,conf,k);pred=decode(prob,v,r['role']);trace_reuse.append(dict(condition=k,id=r['id'],origin=origin,file=file))
   if k in saved:assert pred==next(x['prediction'] for x in saved[k] if x['id']==r['id'])
   runs=emission_runs(prob,v);forced=constrained_alignment(prob,r['truth'],v);iidx=v.encode('i')[0]
   trace_stats.append(dict(condition=k,id=r['id'],role=r['role'],truth=r['truth'],prediction=pred,blank_mean=float(prob[:,0].mean()),blank_argmax_fraction=float(np.mean(prob.argmax(1)==0)),space_emissions=pred.count(' '),i_emissions=pred.count('i'),target_i=r['truth'].count('i'),i_nonblank_mass=float(prob[:,iidx].sum()/prob[:,1:].sum()),time_steps=len(prob),target_length=len(r['truth']),outside_role_emissions=sum(ch not in allowed(r['role']) for ch in pred)))
   for op,ri,hj,a,b in align(r['truth'],pred):
    if op=='match':continue
    span=forced[ri] if op=='deletion' else np.arange(runs[hj]['start'],runs[hj]['end']);peak=int(span[np.argmax(prob[span,v.encode(b)[0]])]) if b else int(span[np.argmax(prob[span,v.encode(a)[0]])]);target=v.encode(a)[0] if a else None
    posterior_events.append(dict(condition=k,id=r['id'],role=r['role'],operation=op,reference=a,predicted=b,reference_index=ri,hypothesis_index=hj,frame_start=int(span.min()),frame_end=int(span.max())+1,peak_frame=peak,reference_probability_at_peak=float(prob[peak,target]) if a else '',predicted_probability_at_peak=float(prob[peak,v.encode(b)[0]]) if b else '',blank_mean_on_span=float(prob[span,0].mean()),reference_rank_at_peak=int(np.sum(prob[peak]>prob[peak,target]))+1 if a else '',span_type='hypothetical truth-constrained Viterbi span; not a verified glyph box' if op=='deletion' else 'observed greedy emission run'))
 write('ctc_error_posteriors.csv',posterior_events);write('ctc_tier_statistics.csv',trace_stats);write('trace_reuse_log.csv',trace_reuse)
 # Changed synthetic examples: fill missing posteriors and D/raw counterfactual, no repeated full-cohort inference.
 counter=[]
 for k in ['A','B','D','D_raw']:
  key='A' if k=='B' else 'D' if k=='D_raw' else k;m,v,_=loaded[key];conf=CONTRAST if k in ['B','D'] else IDENTITY
  for r in changed:
   prob,_,_=probability(m,v,r,conf,'reg_'+k);prediction=decode(prob,v,r['role'])
   if k in reg:assert prediction==next(z['prediction'] for z in reg[k] if z['id']==r['id'])
   row=result_row(r,prediction,prob,v);counter.append(dict(condition=k,**row,space_argmax_frames=int((prob.argmax(1)==v.encode(' ')[0]).sum()),blank_argmax_frames=int((prob.argmax(1)==0).sum())))
 write('changed_regression_counterfactuals.csv',counter)
 write('small_mark_retention.csv',[x for r in tr+va for x in retention(r)])
 controls=[]
 for cohort in ['clean_historical_regression','positive_development']:
  for role in ['upper','count','lower']:controls+=sorted([r for r in dev if r['cohort']==cohort and r['role']==role],key=lambda x:x['id'])[:20]
 write('spacing_consistency.csv',[spacing(r) for r in tr+va+controls if r['role']!='count'])
 features(loaded,tr,dev)
 # No new role-aware normalization: training evidence says contrast benefits every role, reproducing V2 B.
 rolecal=[]
 for role in ['upper','count','lower']:
  for variant in ['identity','contrast']:
   rr=[r for r in read(V/'predictions'/('calibration_train_'+variant+'.csv')) if r['role']==role];writers={r['writer_id'] for r in rr};macro=np.mean([sum(int(r['edit_distance']) for r in rr if r['writer_id']==w)/sum(len(r['truth']) for r in rr if r['writer_id']==w) for w in writers]);rolecal.append(dict(role=role,normalization=variant,macro_writer_CER=float(macro)))
 write('reused_role_normalization_calibration.csv',rolecal)
 dump('analysis_complete.json',dict(status='PROVISIONAL',new_training=False,primary_tiers=14,source_of_old_metrics=str(V),protected_writers_images_used=False,role_constraints_directly_motivated_by_invalid_role_emissions=True,weight_shrinkage_directly_motivated_by_B_D_regression=True,all_saved_W05_predictions_reproduced=True))
 print('ANALYSIS COMPLETE',flush=True)
def features(loaded,tr,dev):
 hand=[r for r in tr if len(r['truth'])==1];classes=sorted({r['truth'] for r in hand}&{r['truth'] for r in dev if len(r['truth'])==1});hand=[r for r in hand if r['truth'] in classes];synth=[]
 for ch in classes:synth+=sorted([r for r in dev if r['truth']==ch],key=lambda r:r['id'])[:3]
 results=[];vectors={}
 for condition,key,conf in [('A','A',IDENTITY),('B','A',CONTRAST),('C','C',IDENTITY),('D','D',CONTRAST)]:
  m=loaded[key][0]
  for r in hand+synth:
   t,_=v2.normalize(v2.tensors(r),r['role'],conf);arr=t[0].numpy();ys,xs=np.where(arr<.98);box=(xs.min(),ys.min(),xs.max()+1,ys.max()+1);captured={};hooks=[]
   for name,module in m.cnn.named_children():
    if isinstance(module,(torch.nn.MaxPool2d,torch.nn.AdaptiveAvgPool2d)):
     def capture(mod,inp,out,name=name):
      z=out[0];H,W=z.shape[1:];x0=max(0,int(box[0]*W/arr.shape[1]));x1=min(W,max(x0+1,int(math.ceil(box[2]*W/arr.shape[1]))));y0=max(0,int(box[1]*H/64));y1=min(H,max(y0+1,int(math.ceil(box[3]*H/64))));roi=z[:,y0:y1,x0:x1].flatten(1);vec=torch.cat([roi.mean(1),roi.std(1,unbiased=False)]).cpu().numpy();captured[name]=vec
     hooks.append(module.register_forward_hook(capture))
   with torch.inference_mode():m(t.unsqueeze(0))
   for h in hooks:h.remove()
   vectors[(condition,r['id'])]=captured
  layers=next(iter(vectors[(condition,hand[0]['id'])])) if hand else None
  for r in hand:
   for layer,vec in vectors[(condition,r['id'])].items():
    proto={ch:np.mean([vectors[(condition,s['id'])][layer] for s in synth if s['truth']==ch],axis=0) for ch in classes}
    cos={ch:float(np.dot(vec,z)/(np.linalg.norm(vec)*np.linalg.norm(z)+1e-12)) for ch,z in proto.items()};best=max(cos,key=cos.get);wrong=max(v for k,v in cos.items() if k!=r['truth'])
    results.append(dict(condition=condition,handwritten_id=r['id'],writer_id=r['writer_id'],role=r['role'],putative_label=r['truth'],cnn_pool_layer=layer,matched_cosine=cos[r['truth']],best_other_cosine=wrong,matched_margin=cos[r['truth']]-wrong,nearest_synthetic_label=best,nearest_matches_label=best==r['truth'],reference_classes=len(classes),qualification='Whole-tier single-symbol labels; no fabricated per-glyph boxes; not independent reader verification. ROI pooled feature retrieval is descriptive, not OCR accuracy or causal localization.'))
 write('cnn_feature_comparison.csv',results)
 from PIL import Image,ImageDraw
 sheet=Image.new('RGB',(1600,len(classes)*130),'#eeeeee');d=ImageDraw.Draw(sheet)
 for i,ch in enumerate(classes):
  hr=next(r for r in hand if r['truth']==ch);sr=next(r for r in synth if r['truth']==ch)
  for j,(label,r) in enumerate([('existing synthetic',sr),('training handwriting',hr)]):
   t=v2.tensors(r)[0].numpy();im=Image.fromarray(np.round(t*255).astype(np.uint8)).convert('RGB');im.thumbnail((770,85));d.text((10+j*800,i*130+3),repr(ch)+' '+label+' '+r['id'],fill='black');sheet.paste(im,(10+j*800,i*130+26))
 sheet.save(O/'visual_failure_examples/isolated_symbol_feature_pairs.png')
 dump('feature_probe_scope.json',dict(training_tiers=len(hand),synthetic_tiers=len(synth),classes=classes,scope='Only original training-writer isolated-single-character tiers and existing synthetic development tiers. H lacks a matched single-class control and is excluded from this descriptive probe, not from OCR evaluation. No W05 class centroid fitting. No count-tier isolated glyph correspondence is available; count localization uses posterior/whole-tier evidence instead.'))
def macro_writer(rr):
 return float(np.mean([sum(r['edit_distance'] for r in rr if r['writer_id']==w)/sum(len(r['truth']) for r in rr if r['writer_id']==w) for w in sorted({r['writer_id'] for r in rr})]))
def synthetic_safe(reg,base,gate):
 checks={}
 for cohort,b in base.items():
  lim=gate['clean_cer_delta'] if cohort.startswith('clean') else gate['long_cer_delta'] if cohort.startswith('long') else gate['positive_cer_delta']
  checks[cohort+'_cer']=reg[cohort]['cer']-b['cer']<=lim+1e-12
  if cohort.startswith('clean'):checks[cohort+'_exact']=reg[cohort]['exact_accuracy']>=b['exact_accuracy']-gate['clean_exact_drop']-1e-12
 return dict(passed=all(checks.values()),checks=checks)
def blend(e1,d,alpha):
 state={}
 for k,x in e1.items():
  if k.startswith('cnn.') and x.is_floating_point() and not torch.equal(x,d[k]):state[k]=(1-alpha)*x+alpha*d[k]
  else:state[k]=x.clone()
 return state
def fixes():
 assert (O/'analysis_complete.json').exists() and not (O/'final_comparison_report.md').exists()
 tr,va,dev=inputs();protocol=load(O/'protocol.json');old=load(V/'final_comparison.json');gate=protocol['gates'];baseline=old['approaches']['A']['validation'];basereg=old['approaches']['A']['regression'];m,v,e1=model(protocol['models']['A']);dm,_,ds=model(protocol['models']['D'])
 controls=[]
 for cohort in ['clean_historical_regression','positive_development']:
  for role in ['upper','count','lower']:controls+=sorted([r for r in dev if r['cohort']==cohort and r['role']==role],key=lambda x:x['id'])[:20]
 # Reuse existing B training/control predictions to avoid redundant raw inference.
 bp=read(V/'predictions/calibration_train_contrast.csv');base_macro=float(np.mean([sum(int(r['edit_distance']) for r in bp if r['writer_id']==w)/sum(len(r['truth']) for r in bp if r['writer_id']==w) for w in {'W01','W03','W04','W06'}]));cal=[];experiments=[]
 _,ftr=evaluate(m,v,tr,CONTRAST,'B','F1_role_mask_training',restricted=True,save=True)
 _,fcontrol=evaluate(m,v,controls,CONTRAST,'B','F1_role_mask_synthetic_controls',restricted=True,save=True)
 controlB={r['id']:r for r in read(V/'predictions/calibration_synthetic_contrast.csv')};role_train_pass=macro_writer(ftr)<base_macro and all(r['edit_distance']<=int(controlB[r['id']]['edit_distance']) for r in fcontrol)
 selection=dict(name='F1_authentic_role_mask',training_macro_CER=macro_writer(ftr),baseline_B_training_macro_CER=base_macro,training_and_synthetic_control_pass=role_train_pass,definition={role:sorted(allowed(role)&set(v.char_to_index)) for role in ['upper','count','lower']},authentic_symbols_not_in_original_model_vocabulary=sorted(set.union(*(allowed(role) for role in ['upper','count','lower']))-set(v.char_to_index)),blank_index=0,space_only_in_upper_lower=True,limitations='Authentic role alphabets intersected with unchanged E1 vocabulary (ambiguous O has no model class). Requires trusted tier role on valid crops; preserve unmasked text and role-violation flags. Does not establish a safe replacement for Stage2F invalid-notation rejection. No class i suppression: i is a legal count class.')
 dump('F1_selection_before_W05.json',selection)
 if role_train_pass:
  val,_=evaluate(m,v,va,CONTRAST,'B','F1_role_mask_W05',restricted=True,save=True);_,rp=evaluate(m,v,dev,CONTRAST,'B','F1_role_mask_regression',restricted=True,save=False);reg=regression(rp,dev);test=v2.gates(val,reg,baseline,basereg,gate);testB=v2.gates(val,reg,old['approaches']['B']['validation'],old['approaches']['B']['regression'],gate)
  experiments.append(dict(name='F1_authentic_role_mask',validation=val,regression=reg,gates_vs_A=test,gates_vs_B=testB,passed=test['passed'] and testB['passed'],model=protocol['models']['A'],model_sha256=protocol['model_hashes']['A'],normalization=CONTRAST,decoder='authentic-role-masked greedy CTC'))
 else:experiments.append(dict(name='F1_authentic_role_mask',passed=False,not_evaluated_on_W05=True,reason='Rejected by training-only improvement/synthetic-control requirement',selection=selection))
 # A single scalar shrinks only already-adapted CNN tensors. No optimizer, backward or BN updates.
 alpha_results=[]
 for alpha in [.25,.5,.75]:
  candidate,_,_=model(protocol['models']['A']);candidate.load_state_dict(blend(e1['model_state_dict'],ds['model_state_dict'],alpha));condition='blend_'+str(int(alpha*100));train,tp=evaluate(candidate,v,tr,CONTRAST,condition,'F2_alpha_'+str(alpha)+'_training',save=False);_,rp=evaluate(candidate,v,dev,CONTRAST,condition,'F2_alpha_'+str(alpha)+'_regression',save=False);reg=regression(rp,dev);safety=synthetic_safe(reg,basereg,gate);macro=macro_writer(tp)
  item=dict(alpha=alpha,training_macro_CER=macro,training=train,regression=reg,synthetic_gates=safety,eligible_for_selection=safety['passed'] and macro<base_macro);alpha_results.append(item);dump('F2_calibration.json',alpha_results);print('ALPHA',alpha,'train_macro',macro,'safe',safety,flush=True)
 good=[x for x in alpha_results if x['eligible_for_selection']]
 if good:
  selected=min(good,key=lambda x:(x['training_macro_CER'],x['alpha']));alpha=selected['alpha'];dump('F2_selection_before_W05.json',dict(selected_alpha=alpha,training_macro_CER=selected['training_macro_CER'],source='Training writers + full existing synthetic development only; no W05 alpha comparison',safety=selected['synthetic_gates'],no_training=True));candidate,_,_=model(protocol['models']['A']);candidate.load_state_dict(blend(e1['model_state_dict'],ds['model_state_dict'],alpha));val,_=evaluate(candidate,v,va,CONTRAST,'blend_'+str(int(alpha*100)),'F2_shrunk_CNN_W05',save=True);reg=selected['regression'];test=v2.gates(val,reg,baseline,basereg,gate);testB=v2.gates(val,reg,old['approaches']['B']['validation'],old['approaches']['B']['regression'],gate)
  path=O/'candidate_models/inference_only_shrunk_CNN.pt';torch.save(dict(model_state_dict=candidate.state_dict(),char_to_idx=e1['char_to_idx'],vocabulary_hash=e1['vocabulary_hash'],format_version='stage2g_v3_inference_only_CNN_shrinkage',alpha=alpha,base_E1_sha256=protocol['model_hashes']['A'],V2_D_sha256=protocol['model_hashes']['D'],normalization=CONTRAST,training_performed=False),path)
  experiments.append(dict(name='F2_shrunk_CNN',alpha=alpha,validation=val,regression=reg,gates_vs_A=test,gates_vs_B=testB,passed=test['passed'] and testB['passed'],model=str(path),model_sha256=sha(path),normalization=CONTRAST,decoder='original greedy CTC',training=False))
 else:experiments.append(dict(name='F2_shrunk_CNN',passed=False,not_evaluated_on_W05=True,reason='No train-selected scalar passed synthetic development gates and training improvement; no W05 alpha sweep'))
 dump('targeted_fix_results.json',experiments);flat=[]
 for x in experiments:
  if 'validation' in x:flat.append(dict(name=x['name'],evaluated=True,errors=x['validation']['overall']['errors'],cer=x['validation']['overall']['cer'],cer_no_whitespace=x['validation']['overall']['cer_no_whitespace'],exact_tiers=x['validation']['overall']['exact_tiers'],passed=x['passed'],failed_gates=';'.join(sorted({k for g in ['gates_vs_A','gates_vs_B'] for k,vv in x[g]['checks'].items() if not vv}))))
  else:flat.append(dict(name=x['name'],evaluated=False,errors='',cer='',cer_no_whitespace='',exact_tiers='',passed=False,failed_gates=x['reason']))
 write('targeted_fix_experiments.csv',flat);make_reports(experiments)
 protected=load(O/'protected_previous_hashes.json');assert all(sha(p)==h for p,h in protected.items());assert all(not p.requires_grad and p.grad is None for z in [m,dm] for p in z.parameters())
 dump('integrity_checks.json',dict(previous_V2_files_and_E1_unchanged=True,no_optimizer_or_backward=True,no_synthetic_generation=True,no_test_or_quarantined_images_loaded=True,normalization_reused_without_tuning=True,all_new_parameters_selected_before_W05=True,original_labels_unchanged=True))
 print('V3 COMPLETE',flat,flush=True)
def make_reports(experiments):
 old=load(V/'final_comparison.json');errors=read(O/'character_error_analysis.csv');reg=read(O/'regression_failure_analysis.csv');post=read(O/'ctc_tier_statistics.csv');events=read(O/'ctc_error_posteriors.csv');feat=read(O/'cnn_feature_comparison.csv');ret=read(O/'small_mark_retention.csv');sp=read(O/'spacing_consistency.csv')
 text='# Stage 2G V3 root-cause analysis — PROVISIONAL\n\nNo training, synthetic generation, source-label change, dataset restructuring or protected-test/quarantine inference. The analysis reuses V2 predictions and metrics; new inference fills missing posteriors/features and tests two narrowly scoped inference-only interventions. W05 was already exposed in V2 and motivates these hypotheses, so this is exploratory development, not a fresh independent validation. Writer provenance and Talim copying/transcription remain incompletely verified.\n\n## What the observed errors show\n\n| Approach | Errors | Symbol-only errors | Whitespace-involving edit events | Out-of-role predicted edit events |\n|---|---:|---:|---:|---:|\n'
 for k in ['A','B','C','D']:
  ee=[r for r in errors if r['approach']==k];m=old['approaches'][k]['validation']['overall'];text+=f"| {k} | {m['errors']} | {sum(m['no_space_'+z] for z in ['substitutions','deletions','insertions'])} | {sum(r['whitespace_involved']=='True' for r in ee)} | {sum(r['predicted_outside_authentic_role']=='True' for r in ee)} |\n"
 text+='\nThe symbol-only and whitespace-involving counts are different edit alignments and cannot be added as an error decomposition. per_tier_comparison.csv shows all 14 exact truth/A/B/C/D strings. character_error_analysis.csv records reference/hypothesis indices for each deterministic substitution, deletion and insertion; character_confusions.csv aggregates by role. Whitespace positions are left intact; no target relabeling or trimming beyond the established leading-space rule occurs.\n\n### Roles and concentration\n\n| Approach | Role | S / D / I | CER | No-whitespace CER |\n|---|---|---:|---:|---:|\n'
 for k in ['A','B','C','D']:
  for role,m in old['approaches'][k]['validation']['per_role'].items():text+=f"| {k} | {role} | {m['substitutions']} / {m['deletions']} / {m['insertions']} | {m['cer']:.4%} | {m['cer_no_whitespace']:.4%} |\n"
 text+='\nAll primary errors belong to W05, two sources. Single-writer data cannot identify a writer-population effect. Source errors and macro/source metrics are retained from V2. The count tier has substantial within-alphabet errors, including over-emission of the legitimate encoded count class `i`; a role prior cannot eliminate a legal class. Upper/lower errors include wrong-role outputs (e.g. upper E/C predicted Z/Y, lower [ predicted G) and wrong same-role identities. File-consistent labels do not prove the writer copied every symbol correctly.\n\n## Posterior and sequence evidence\n\n'
 for k in ['A','B','C','D']:
  rr=[r for r in post if r['condition']==k and r['role']=='count'];text+=f"- {k} count: {sum(int(r['i_emissions']) for r in rr)} decoded i versus {sum(int(r['target_i']) for r in rr)} target i; {sum(int(r['outside_role_emissions']) for r in rr)} out-of-role emitted characters; mean blank argmax fraction {np.mean([float(r['blank_argmax_fraction']) for r in rr]):.3f}.\n"
 text+='\nctc_error_posteriors.csv links substitution/insertion edits to actual greedy emission runs, with reference/predicted posterior and truth rank at the observed peak. Deletion spans are best paths constrained to the reference, explicitly hypothetical rather than verified glyph boxes. Low truth posterior/rank is evidence of poor model likelihood, not proof of why a particular mark was missed. The earlier diagnosis already established adequate CTC sequence lengths and correct repeat/blank collapse; those analyses are not rerun generically. New writer-split traces reproduce all saved V2 outputs. Only three compatible old frozen-A W05 row0 traces are reused; no W07/quarantine traces are loaded.\n\nRole leakage is directly observable at the output distribution. The CNN-versus-RNN attribution is not uniquely identifiable from a wrong character alone: frozen E1 can misrepresent ink, recurrent context can alter scores, and labels may be semantically unverified. A greedy-search-only explanation is inconsistent with the previous limited beam improvement.\n\n## Crop extraction, small marks, and appearance\n\n'
 text+=f"Measured {len(ret)} selected small native components across 108 training originals and 14 W05 crops; {sum(r['below_1_over_255_after']=='True' for r in ret)} projected model-scale ROIs fall below 1/255 peak darkness and {sum(r['below_original_midcontrast_after']=='True' for r in ret)} fall below original midcontrast. These are smallest-component bbox proxies, not verified Talim dots. Neighbouring ink can share an ROI; threshold changes can alter detectability. Therefore these counts cannot be interpreted as semantic mark-deletion rates or assigned to OCR errors without reader/photo-mask correspondence. Existing crop hashes and whole-tier mappings remain consistent. Original segmentation masks/photos are not fully packaged, so extraction-stage missing strokes cannot be independently reconstructed or ruled out. No segmentation-failure CER is invented.\n"
 text+='\nThe retained original crops and their actual height64 inputs are shown in visual_failure_examples/. Prior V2 contrast/thickness/geometry measurements and the earlier domain-gap report are reused: thin, light, irregular appearance remains a supported mismatch; generic height normalization did not close it. New spacing checks and single-class feature probes address the writer-split-specific questions instead of repeating those generic ablations.\n\n## Intermediate CNN feature probe\n\n'+json.dumps(load(O/'feature_probe_scope.json'),indent=2)+'\n\n| Condition | Last-pool mean matched cosine | Last-pool mean matched-minus-best-other margin | Matched nearest prototype |\n|---|---:|---:|---:|\n'
 layer=max({int(r['cnn_pool_layer']) for r in feat})
 for k in ['A','B','C','D']:
  rr=[r for r in feat if r['condition']==k and int(r['cnn_pool_layer'])==layer];text+=f"| {k} | {np.mean([float(r['matched_cosine']) for r in rr]):.4f} | {np.mean([float(r['matched_margin']) for r in rr]):+.4f} | {sum(r['nearest_matches_label']=='True' for r in rr)}/{len(rr)} |\n"
 text+='\nCNN activations are pooled within projected visible-ink ROI at each existing pooling block; no new glyph crops, learned probes or training are produced. Prototype comparisons use existing synthetic single-symbol development tiers and isolated training-writer one-character tiers. These small-class-set similarity/retrieval values are descriptive, not OCR accuracies or evidence that the CNN alone causes every error. Receptive fields extend beyond the approximate ROI. Class labels are putative on handwriting until reader verification. There are no reliable isolated count-glyph targets here; no forced segmentation is used to fabricate count correspondences.\n\n## Whitespace and geometry\n\n'
 for domain in ['handwritten_train','handwritten_validation','synthetic_development']:
  rr=[r for r in sp if r['domain']==domain];matched=[r for r in rr if r['group_count_matches']=='True'];text+=f"- {domain}: {len(matched)}/{len(rr)} upper/lower tiers admit a count-matched horizontal-group/target-position fit; median local cell-pitch CV {np.median([float(r['local_cell_pitch_cv']) for r in matched]):.3f} (descriptive).\n"
 text+='\nThe fit uses target character positions offline to inspect consistency, never to repair inference or fit W05 parameters. Multi-component marks can break grouping; unmatched group counts are left unfit, not silently aligned. Spatial gaps are not sufficient evidence to insert a particular number of label spaces on irregular handwriting. Geometry-aware spacing recovery is therefore not implemented. Blank/space emissions and internal label-grid spacing can disagree even when symbol identities are preserved.\n\n## Exact synthetic regression attribution\n\n| Cohort | A errors | B errors | D errors | A→B preprocessing delta | B→D fixed-preprocessing weight delta |\n|---|---:|---:|---:|---:|---:|\n'
 for cohort in old['approaches']['A']['regression']:
  a=old['approaches']['A']['regression'][cohort]['errors'];b=old['approaches']['B']['regression'][cohort]['errors'];d=old['approaches']['D']['regression'][cohort]['errors'];text+=f"| {cohort} | {a} | {b} | {d} | {b-a:+d} | {d-b:+d} |\n"
 text+='\nLong-positive is a subset of positive development, never extra independent samples. regression_failure_analysis.csv identifies the exact examples and source parents. Three variants of positive_edge_00323 lower retain identical nonspace text while D loses 12 more spaces in total than A: one additional space deletion appears after contrast and eleven after the CNN updates at fixed contrast. Across the entire long cohort, the total edit-count decomposition is +2 preprocessing and +10 weight updates; this also includes a B-only upper I deletion that D corrects. These are correlated variants of one source, not three independent mechanisms. Two short A→U upper errors are already present in B, so they are preprocessing-associated rather than newly caused by D. At fixed normalization B→D, v→u count substitutions increase by four and m→u by two, alongside punctuation/count-token substitutions. The CNN is the only weight block that changed in V2; BiLSTM/classifier weights did not learn new parameters, although their outputs respond to changed CNN features.\n\nchanged_regression_counterfactuals.csv adds D with raw preprocessing only on changed development examples, completing a diagnostic 2×2 comparison for those cases. It is a selected-case counterfactual, not a full-cohort D/raw score. Correlation between improved handwriting and worsened synthetic classes does not prove a shared per-class cause; compare regression_character_changes.csv with handwritten confusions, noting unverified handwriting targets.\n\n## Confirmed versus uncertain mechanisms\n\n| Evidence level | Finding | Limit |\n|---|---|---|\n| Direct | Many wrong identities remain after whitespace removal; count i over-emission includes legal-class errors | Does not distinguish visual shape from copying error for every glyph |\n| Direct | Wrong-role emissions contradict authentic role alphabets on these intended-valid tiers | Role restriction can hide invalid notation; preserve raw evidence |\n| Direct | D adds space deletions on unchanged synthetic symbol strings and adds other count confusions | One source supplies the long-space variants |\n| Direct intervention | A/B isolates the existing normalization; B/D isolates changed CNN weights | No unique neural-stage diagnosis for every handwritten error |\n| Descriptive | Single-symbol CNN similarities and small-component/pitch measures show representation/layout differences | Proxy measurements; no independent count-glyph correspondence |\n| Uncertain | Missing semantic dots during original extraction, writer copying mistakes, exact per-error CNN/RNN causation | Need original masks/photos and independent Talim-reader localized annotations |\n\n## Targeted decisions\n\nRole-aware contrast calibration is not a third fix: reused training-only results show contrast benefits every role, so it collapses to the already evaluated B. No new thickness or glyph-height setting is tuned. F1 tests the authentic role alphabet while leaving blank available and preserving raw output; it does not suppress the legitimate count class i. F2 shrinks only D\'s existing CNN delta toward E1, testing one predetermined three-point scalar grid on training writers plus synthetic development. It performs no gradient updates and never tunes alpha on W05. These are separate interventions, not an uncontrolled stack.\n'
 if (O/'source_font_evidence.json').exists():
  font=load(O/'source_font_evidence.json');pairs=read(O/'cross_role_font_outline_evidence.csv');profile=load(P/'stage2/stage2f_tier_aware_decoding/configs/rendering_profile.json');assert font['sha256']==profile['font_sha256']
  section='## Verified cross-role shape ambiguity\n\nThe inspected source font SHA-256 matches the existing frozen rendering profile. Its upper encoded C and lower encoded Y have exactly the same ordered contour points/on-curve flags after translation; only their font position differs (450 font units vertically). No new glyph was rendered. This verifies a real role-dependent encoding ambiguity, not merely a visual resemblance. CRNN.forward receives only the image, with no role embedding or classifier conditioning. Whole-page baseline position can distinguish the roles, but tier cropping/height normalization and irregular handwriting do not provide a reliable global baseline. Inferring that this is a cause of some cross-role confusions is supported by the direct outline equivalence and observed C→Y error; it does not prove identical raster inputs or explain all errors.\n\n| Upper / lower pair | Exact ordered outline equivalence after translation |\n|---|---|\n'
  for q in pairs:section+=f"| {q['upper']} / {q['lower']} | {q['ordered_translation_equivalent']} |\n"
  section+='\nThe other tested pairs are visually similar but are not asserted to have identical outlines. The fixed role mask corrects some E/Z and K/^ confusions, but in the inspected E-space-C tier it changes Z-space-Y to E followed by spaces, still deleting C. Thus a hard role prior helps but cannot recover all likelihood already assigned to the wrong class. No additional alias substitution/probability-folding fix was fitted after observing this result. Reliable role-conditioned recognition is an evidence-supported direction for later controlled work; missing count-glyph correspondence and target semantics remain unresolved. See font_outline_check.py, cross_role_font_outline_evidence.csv and source_font_evidence.json.\n\n'
  text=text.replace('## Posterior and sequence evidence',section+'## Posterior and sequence evidence')
 (O/'root_cause_report.md').write_text(text,encoding='utf-8')
 final='# Stage 2G V3 controlled targeted-fix comparison — PROVISIONAL\n\nNo new training or synthetic data generation. Original E1 and all V2 files remain unchanged. One W05 writer/two sources/14 tiers cannot establish population generalization; no IID-tier confidence interval is reported. Sourcewise results provide sensitivity only.\n\n| Approach | Errors / chars | CER | No-whitespace CER | Exact tiers | Gates |\n|---|---:|---:|---:|---:|---|\n'
 for k in ['A','B','C','D']:
  x=old['approaches'][k];m=x['validation']['overall'];final+=f"| V2 {k} | {m['errors']}/{m['characters']} | {m['cer']:.4%} | {m['cer_no_whitespace']:.4%} | {m['exact_tiers']}/14 | {'reference' if k=='A' else str(x['eligible'])} |\n"
 for x in experiments:
  if 'validation' in x:
   m=x['validation']['overall'];final+=f"| {x['name']} | {m['errors']}/{m['characters']} | {m['cer']:.4%} | {m['cer_no_whitespace']:.4%} | {m['exact_tiers']}/14 | {x['passed']} |\n"
  else:final+=f"| {x['name']} | not evaluated on W05 | — | — | — | training/development calibration rejection |\n"
 final+='\n## Targeted changes and calibration\n\nF1 applies the existing fixed contrast normalization to frozen E1 and masks frame-level nonblank classes outside the authentic known tier alphabet; blank remains legal and internal spaces remain legal only for upper/lower. It uses the original greedy collapse. Its raw unmasked counterpart is retained, not overwritten. This is conditional valid-tier recognition, not a safe replacement for Stage2F invalid-notation rejection.\n\nF2 uses (1−alpha)×E1 CNN + alpha×V2-D CNN, with all unmodified/equal tensors and recurrent/classifier parameters preserved exactly. The existing contrast recipe remains fixed. Only .25/.5/.75 are calibrated, solely on training writers and full existing synthetic development. selected alpha is fixed before its single W05 evaluation. It is inference-only shrinkage, not fine-tuning. The separate targets are motivated by observed illegal-role predictions and D\'s fixed-preprocessing regression respectively; neither is guaranteed to solve legal-class count confusion.\n\n'
 for x in experiments:
  final+='### '+x['name']+'\n\n'
  if 'validation' not in x:final+=x['reason']+'\n\n';continue
  final+='| Cohort | Errors / chars | CER | No-whitespace CER | Exact tiers |\n|---|---:|---:|---:|---:|\n'
  for cohort,m in x['regression'].items():final+=f"| {cohort} | {m['errors']}/{m['characters']} | {m['cer']:.4%} | {m['cer_no_whitespace']:.4%} | {m['exact_tiers']}/{m['tiers']} |\n"
  final+='\n| Role | S / D / I | CER | No-whitespace CER |\n|---|---:|---:|---:|\n'
  for role,m in x['validation']['per_role'].items():final+=f"| {role} | {m['substitutions']} / {m['deletions']} / {m['insertions']} | {m['cer']:.4%} | {m['cer_no_whitespace']:.4%} |\n"
  final+='\nSourcewise outcomes: '+json.dumps(x['validation']['per_source'])+'\n\n'
  fail=sorted({k for g in ['gates_vs_A','gates_vs_B'] for k,b in x[g]['checks'].items() if not b});final+='Failed fixed gates: '+(', '.join(fail) or 'none')+'. Model: '+x['model']+'; SHA-256: '+x['model_sha256']+'.\n\n'
 eligible=[x for x in experiments if x.get('passed')];best=min(eligible,key=lambda x:(x['validation']['overall']['cer'],x['validation']['overall']['cer_no_whitespace'])) if eligible else None
 dump('selection_decision.json',dict(status='PROVISIONAL',eligible_targeted_fix=best['name'] if best else None,operational_default_unchanged=str(c.E/'checkpoints/selected_model/best.pt'),no_automatic_promotion=True,reason='Any eligible result is conditional development evidence. Role-mask inference requires reliable roles on valid tiers and must not sanitize invalid-notation evidence for Stage2F. New-writer and reader verification remain outstanding.'))
 final+='## Retained model and further evidence\n\n'+('Best gate-passing targeted development configuration: '+best['name']+'. ' if best else 'Neither targeted proposal qualifies for promotion under all existing gates. ')+'The operational frozen E1 file remains selected and unchanged; no production policy or entry point is modified. Keep raw model text, role-violation signals and provenance available. A role-masked result must not be passed to Stage2F as if it were the untouched OCR evidence. Stage2F rejection regression is not established here.\n\nBefore more fine-tuning: independent Talim-reader checks and localized mark/count annotations; original crop masks/photos to verify missing dots; more real writers for independent evaluation; valid/invalid role reliability and rejection-policy tests; and train-only calibration of any spacing rule on verified glyph correspondences. These are evidence needs, not additional experiments performed now. The demonstrated improvement/regression mechanisms are observations; claims that any one feature block or every faint dot explains all errors remain hypotheses.\n\n## Reproduction\n\nUse experiment.py with --stage analysis, then --stage fixes in a new isolated O directory (change only O for a new run). Existing V2 hashes and paths must match. No package installation or network is required. The script reuses saved predictions and compatible posterior traces, fills unavailable traces, and records all calibration choices before W05 evaluation. Earlier 74.23% source-split results are not pooled with this writer-based set. All outputs are in this directory.\n'
 if (O/'source_font_evidence.json').exists():final+='\nA further read-only font check confirms an exact upper-C/lower-Y outline alias under translation in the existing source font. This supports explicit tier-role conditioning rather than expecting generic handwriting fine-tuning to resolve every role-dependent identity from stroke appearance alone. The check does not change either tested fix or its calibration. Run font_outline_check.py before --stage reports to reproduce this additional evidence; no synthetic glyph images are generated.\n'
 (O/'final_comparison_report.md').write_text(final,encoding='utf-8')
 # Actual model-scale comparisons preserve originals and never enter a dataset.
 _,va,_=inputs()
 for r in va:
  sheet=Image.new('RGB',(1800,360),'#eeeeee');draw=ImageDraw.Draw(sheet)
  for i,(label,arr) in enumerate([('original native crop',cv.imread(r['path'],0)),('actual E1 height64 input',np.round(v2.tensors(r)[0].numpy()*255).astype(np.uint8)),('fixed contrast height64 input',np.round(v2.normalize(v2.tensors(r),r['role'],CONTRAST)[0][0].numpy()*255).astype(np.uint8))]):
   im=Image.fromarray(arr).convert('RGB');im.thumbnail((1780,85));draw.text((10,i*120+2),r['id']+' '+label,fill='black');sheet.paste(im,(10,i*120+25))
  sheet.save(O/'visual_failure_examples'/(r['id']+'_model_scale.png'))
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['analysis','fixes','reports'],required=True);args=ap.parse_args();c.seed(42)
 {'analysis':analysis,'fixes':fixes,'reports':lambda:make_reports(load(O/'targeted_fix_results.json'))}[args.stage]()
