"""Isolated V5 experiments. Historical modules are imported read-only."""
import sys, csv, json, hashlib, zipfile, io, collections, importlib.util, time, random
from pathlib import Path
sys.dont_write_bytecode = True
P=Path('C:/NAQSHKASH'); V=P/'stage2/stage2g_writer_independent_finetune_v2'
O=P/'stage2/stage2g_handwriting_adaptation_v5'
spec=importlib.util.spec_from_file_location('v3readonly',P/'stage2/stage2g_targeted_improvements_v3/experiment.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
v2=h.v2;c=h.c;torch=h.torch;np=h.np;cv=h.cv
def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 for r in rows:
  if 'loss' in r:r['loss']=float(r['loss'])
 return rows
def put(n,x):
 p=O/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def write(n,rows):
 if not rows:return
 p=O/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def textfile(n,s):(O/n).write_text(s,encoding='utf-8')
def integrity():
 assert not (O/'training_integrity_report.md').exists(),'Completed integrity audit already exists'
 zpath=Path.home()/'Downloads/naqshkash_stage2g_v3_ground_truth_audit.zip'
 with zipfile.ZipFile(zpath) as z:
  prefix='naqshkash_stage2g_v3_ground_truth_audit/'
  zr=lambda n:list(csv.DictReader(io.StringIO(z.read(prefix+n).decode('utf-8-sig'),newline='')))
  excl=zr('recommended_temporary_exclusions.csv');review=zr('training_label_review.csv')
  for r in zr('checksums.csv'):
   name=r.get('path',r.get('file',''))
   if name and prefix+name in z.namelist():assert hashlib.sha256(z.read(prefix+name)).hexdigest()==r['sha256']
 rows=read(V/'audited_handwritten_inputs.csv')
 assert {r['writer_id'] for r in rows}<={'W01','W03','W04','W06','W05'}
 exclusions={r['sample_or_tier'] for r in excl};tiers={r['sample_or_tier'] for r in excl if r['kind']=='original tier'}
 removed=[r for r in rows if r['split']=='train' and (r['tier_id'] in tiers or r['id'] in exclusions)]
 tr=[r for r in rows if r['split']=='train' and r not in removed]
 counts=collections.Counter(r['sample_type'] for r in tr)
 assert counts=={'original':105,'augmented':2413},counts
 assert collections.Counter(r['sample_type'] for r in removed)=={'original':3,'augmented':56}
 needs=[r for r in review if r['status']=='NEEDS_TALIM_READER_REVIEW'];assert len(needs)==9
 assert {r['tier_id'] for r in needs}<={r['tier_id'] for r in tr}
 va=[r for r in rows if r['writer_id']=='W05' and r['sample_type']=='original'];assert len(va)==14
 replay=read(V/'synthetic_replay_inputs.csv');dev=read(V/'regression_inputs.csv')
 assert all(r['split']=='train' for r in replay)
 assert {r['parent_id'] for r in replay}.isdisjoint({r['parent_id'] for r in dev})
 # Verify allowed input bytes and labels only; never enumerate or open test payloads.
 for r in tr+va:
  assert h.sha(r['path'])==r['sha256'];assert Path(r['label_path']).read_text(encoding='utf-8-sig')==r['raw_label']
 for r in replay+dev:assert h.sha(r['path'])==r['file_sha256']
 e1=c.E/'checkpoints/selected_model/best.pt';assert h.sha(e1)==v2.FROZEN_SHA
 protected={str(p):h.sha(p) for root in [V,h.O] for p in root.rglob('*') if p.is_file()};protected[str(e1)]=h.sha(e1)
 put('protected_inputs.json',protected);write('training_inputs.csv',tr);write('W05_evaluation_inputs.csv',va)
 write('excluded_inputs.csv',removed);write('needs_talim_reader_review.csv',needs);write('recommended_temporary_exclusions.csv',excl)
 put('training_configs/protocol.json',dict(seed=42,training_writers=sorted(v2.TRAIN_WRITERS),W05_selection=False,forbidden_images_opened=False,counts=dict(counts),audit_archive_sha256=h.sha(zpath),gates=json.loads((V/'training_config.json').read_text())['gates'],optional_unconfirmed_variants=False))
 textfile('training_integrity_report.md',f'# V5 training integrity\n\nVerified **105 originals and 2,413 augmented crops**. Removed 3 originals and 56 augmented crops using the supplied audit exclusions. Labels are byte-for-byte unchanged.\n\nTraining writers: W01, W03, W04, W06. W05 has 14 evaluation originals and is excluded from selection. W02/W07 images were never opened. Synthetic replay is TRAIN only and parent-disjoint from regression cohorts. Frozen E1 SHA256: `{v2.FROZEN_SHA}`.\n\nNine NEEDS_TALIM_READER_REVIEW tiers are retained in `needs_talim_reader_review.csv`; the exclusion ablation must remove all descendants of these parents. No semantic corrections applied.\n\nWriter original counts: '+str(dict(collections.Counter(r['writer_id'] for r in tr if r['sample_type']=='original')))+'\n')
 print('INTEGRITY',dict(counts),flush=True)
def evalrows(m,v,rows,contrast=False,restricted=False,cache=None):
 out=[]
 for i,r in enumerate(rows):
  path=cache/(r['id']+'.npz') if cache else None
  if path and path.exists():prob=np.load(path)['probabilities']
  else:
   t=v2.tensors(r)
   if contrast:t,_=v2.normalize(t,r['role'],h.CONTRAST)
   with torch.inference_mode():prob=m(t.unsqueeze(0))[:,0].float().softmax(1).numpy()
   if path:path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,probabilities=prob)
  row=h.result_row(r,h.decode(prob,v,r['role'],restricted),prob,v)
  row['raw_prediction']=h.decode(prob,v,r['role'],False);out.append(row)
  if (i+1)%100==0:print('EVAL',i+1,'/',len(rows),flush=True)
 return out
def cohorts(rr,dev):
 return {name:v2.agg([r for r,d in zip(rr,dev) if pred(d)]) for name,pred in [
 ('synthetic_clean',lambda r:r['cohort'] in {'clean_historical_regression','clean_validation'}),
 ('positive_edge',lambda r:r['cohort']=='positive_development'),
 ('long_positive',lambda r:r['cohort']=='positive_development' and len(r['truth'])>=65)]}
def phase1():
 assert not (O/'phase1_decoder_check.json').exists()
 old=json.loads((V/'final_comparison.json').read_text());path=Path(old['approaches']['D']['checkpoint'])
 assert path.name=='D_normalized_epoch_2.pt';assert h.sha(path)==old['approaches']['D']['sha256']
 m,v,_=h.model(path);va=read(O/'W05_evaluation_inputs.csv');dev=read(V/'regression_inputs.csv');result={};flat=[]
 for mask in [False,True]:
  name='D_role_mask' if mask else 'D_raw_decoder'
  rr=evalrows(m,v,va,True,mask,O/'posteriors/D_W05');write('predictions/'+name+'_W05.csv',rr)
  result[name]=dict(W05_14=v2.summary(rr),W05_13=v2.summary([r for r in rr if r['id']!='all_003801_row_00_count']))
  rr=evalrows(m,v,dev,True,mask,O/'posteriors/D_synthetic');write('predictions/'+name+'_synthetic.csv',rr)
  result[name]['synthetic']=cohorts(rr,dev)
  for cohort,metrics in result[name]['synthetic'].items():flat.append(dict(model=name,cohort=cohort,**metrics))
 put('phase1_decoder_check.json',result);write('synthetic_regression_results.csv',flat)
 textfile('phase1_decoder_check.md','# V2-D decoder check\n\n'+ '\n'.join(f"- {name}: W05 14 CER {r['W05_14']['overall']['cer']:.4%}, no-whitespace {r['W05_14']['overall']['cer_no_whitespace']:.4%}, exact {r['W05_14']['overall']['exact_tiers']}/14; 13-tier CER {r['W05_13']['overall']['cer']:.4%}, exact {r['W05_13']['overall']['exact_tiers']}/13." for name,r in result.items())+'\n\nPer-role metrics and synthetic cohorts are in phase1_decoder_check.json; raw predictions are retained. Role masking does not imply synthetic regression has been repaired.\n')
 print('PHASE1 COMPLETE',flush=True)
def skeleton(mask):
 """Zhang-Suen thinning, on a padded binary image."""
 a=np.pad(mask.astype(np.uint8),1)
 for iteration in range(128):
  changed=False
  for phase in [0,1]:
   p=[a[:-2,1:-1],a[:-2,2:],a[1:-1,2:],a[2:,2:],a[2:,1:-1],a[2:,:-2],a[1:-1,:-2],a[:-2,:-2]]
   n=sum(p);trans=sum((p[i]==0)&(p[(i+1)%8]==1) for i in range(8))
   remove=(a[1:-1,1:-1]>0)&(n>=2)&(n<=6)&(trans==1)
   if phase==0:remove&=(p[0]*p[2]*p[4]==0)&(p[2]*p[4]*p[6]==0)
   else:remove&=(p[0]*p[2]*p[6]==0)&(p[0]*p[4]*p[6]==0)
   if remove.any():a[1:-1,1:-1][remove]=0;changed=True
  if not changed:break
 return a[1:-1,1:-1]
def component_info(a,threshold=128):
 mask=(a<threshold).astype(np.uint8)
 n,labels,stats,cent=cv.connectedComponentsWithStats(mask,8)
 ids=[i for i in range(1,n) if stats[i,cv.CC_STAT_AREA]>=2]
 return mask,labels,stats,ids
def augmented(source,rng):
 """Separate small mark components, redraw body skeleton, reject topology drift at 64px."""
 from scipy.ndimage import gaussian_filter
 raw=np.clip(source.numpy()[0]*255,0,255).astype(np.uint8)
 mask,labels,stats,ids=component_info(raw)
 dots=[i for i in ids if stats[i,cv.CC_STAT_AREA]<=650 and max(stats[i,2:4])<=28 and min(stats[i,2:4])/max(stats[i,2:4])>=.65 and stats[i,cv.CC_STAT_AREA]/max(1,stats[i,2]*stats[i,3])>=.65]
 dotmask=np.isin(labels,dots).astype(np.uint8);body=mask.copy();body[dotmask>0]=0
 sk=skeleton(body);H,W=raw.shape
 for attempt in range(40):
  width=float(rng.uniform(1.5,4));dark=float(rng.uniform(30,200))
  # Distance from skeleton gives continuous stroke widths without eroding dots away.
  dist=cv.distanceTransform((1-sk).astype(np.uint8),cv.DIST_L2,5)
  ink=(dist<=width/2).astype(np.uint8)
  dotstyle=rng.choice(['filled','hollow'])
  for i in dots:
   dm=np.zeros_like(mask);x,y,bw,bh,area=stats[i];center=(int(round(x+(bw-1)/2)),int(round(y+(bh-1)/2)));radius=int(rng.integers(1,4))
   cv.circle(dm,center,radius,1,-1 if dotstyle=='filled' else 1)
   ink=np.maximum(ink,dm)
  # Scale/pitch change preserves the text order and does not introduce glyphs.
  scale=float(rng.uniform(.7,1.1));pitch=float(rng.uniform(.7,1.3));slant=float(rng.uniform(-.12,.12));angle=float(rng.uniform(-8,8))
  outW=max(16,round(W*pitch));rad=np.deg2rad(angle)
  linear=np.array([[np.cos(rad)*scale*pitch,np.sin(rad)*scale*pitch+slant],[-np.sin(rad)*scale,np.cos(rad)*scale]],np.float32)
  mat=np.column_stack([linear,np.array([outW/2,32])-linear@np.array([W/2,32])])
  support=np.column_stack(np.where(ink>0))[:,::-1].astype(np.float32)
  projected=support@mat[:,:2].T+mat[:,2]
  if not len(projected) or np.any(projected[:,0]<1) or np.any(projected[:,0]>outW-2) or np.any(projected[:,1]<1) or np.any(projected[:,1]>62):continue
  warped=cv.warpAffine(ink.astype(np.float32),mat,(outW,64),flags=cv.INTER_LINEAR,borderValue=0)
  yy,xx=np.indices(warped.shape,dtype=np.float32)
  dx=gaussian_filter(rng.normal(size=warped.shape).astype(np.float32),5)*rng.uniform(0,2)
  dy=gaussian_filter(rng.normal(size=warped.shape).astype(np.float32),5)*rng.uniform(0,2)
  warped=cv.remap(warped,xx+dx,yy+dy,cv.INTER_LINEAR,borderMode=cv.BORDER_CONSTANT,borderValue=0)
  image=255-warped*(255-dark)
  if rng.random()<.4:image=cv.GaussianBlur(image,(3,3),float(rng.uniform(.1,.5)))
  lowres=bool(rng.random()<.3)
  if lowres:
   ratio=float(rng.uniform(.7,.95));small=cv.resize(image,None,fx=ratio,fy=ratio,interpolation=cv.INTER_AREA);image=cv.resize(small,(outW,64))
  gap=0
  if rng.random()<.35:
   gap=int(rng.integers(1,4));y=int(rng.integers(5,59));image[y:y+gap]=255
  jpeg=bool(rng.random()<.3)
  image=np.clip(image,0,255).astype(np.uint8)
  if jpeg:
   ok,b=cv.imencode('.jpg',image,[cv.IMWRITE_JPEG_QUALITY,int(rng.integers(65,96))]);image=cv.imdecode(b,0)
  # Darkness-relative threshold is necessary for pale ink; pixel grid is always 64 high.
  am,al,ast,aids=component_info(image,(255+dark)/2)
  if len(aids)!=len(ids):continue
  # All source components must have exactly one distinct transformed counterpart.
  matched=[];safe=True
  for i in ids:
   pts=np.column_stack(np.where(labels==i))[:,::-1].astype(np.float32)
   pts=pts@mat[:,:2].T+mat[:,2]
   x=np.clip(np.rint(pts[:,0]).astype(int),0,outW-1);y=np.clip(np.rint(pts[:,1]).astype(int),0,63)
   hits=al[y,x];hits=hits[hits>0]
   if not len(hits):safe=False;break
   k=int(np.bincount(hits).argmax())
   if k not in aids or k in matched:safe=False;break
   matched.append(k)
  if not safe:continue
  # Preserve body hole counts (dot interiors may intentionally become hollow).
  original_body=(body*255).astype(np.uint8)
  augbody=am.copy()
  for i,k in zip(ids,matched):
   if i in dots:augbody[al==k]=0
  holes=lambda m:sum(1 for e in (cv.findContours(m,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)[1][0] if cv.findContours(m,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)[1] is not None else []) if e[3]>=0)
  if holes(original_body)!=holes((augbody*255).astype(np.uint8)):continue
  return torch.from_numpy(image.astype(np.float32)/255)[None],dict(attempts=attempt+1,target_stroke=width,target_darkness=dark,scale=scale,pitch=pitch,rotation=angle,slant=slant,dot_style=dotstyle,dot_components=len(dots),source_components=len(ids),output_components=len(aids),ruled_gap=gap,low_resolution=lowres,jpeg=jpeg)
 raise RuntimeError('No topology-safe augmentation after 40 attempts')
def appearance(t):
 a=t.numpy()[0];info=v2.inkinfo(a)
 mask=(a<(1+info['blackpoint'])/2).astype(np.uint8);dist=cv.distanceTransform(mask,cv.DIST_L2,5)
 ridge=(mask>0)&(dist>=cv.dilate(dist,np.ones((3,3),np.uint8))-.01)
 core=float(np.median(a[ridge])*255) if ridge.any() else 255.
 return dict(stroke_width=info['stroke'],ink_darkness=core,blackpoint=info['blackpoint']*255)
def augmentation_quality():
 global O
 from PIL import Image,ImageDraw
 replay=read(V/'synthetic_replay_inputs.csv');tr=[r for r in read(O/'training_inputs.csv') if r['sample_type']=='original']
 root_output=O;O=O/'quality_safe3';O.mkdir(exist_ok=True)
 assert not (O/'augmentation_quality.json').exists(),'Preserve completed QA outputs'
 rng=np.random.default_rng(20261110);records=[];failures=[];sheets=[]
 for page in range(10):
  canvas=Image.new('RGB',(1500,20*150),'white');draw=ImageDraw.Draw(canvas)
  for slot in range(20):
   for retry in range(100):
    r=replay[int(rng.integers(len(replay)))];source=v2.tensors(r)
    try:aug,diag=augmented(source,rng);break
    except RuntimeError:failures.append(r['id'])
   else:raise RuntimeError('Unable to obtain 200 safe contact sheet samples')
   idx=len(records);records.append(dict(index=idx,id=r['id'],**diag,**appearance(aug)))
   for j,t in enumerate([source,aug]):
    a=Image.fromarray(np.uint8(np.clip(t.numpy()[0]*255,0,255))).convert('RGB');a.thumbnail((735,105));canvas.paste(a,(j*750,slot*150+35))
   draw.text((5,slot*150+4),f"{r['id']} | source left / augmented right | stroke target {diag['target_stroke']:.2f}, dark {diag['target_darkness']:.0f}",fill='black')
  path=O/f'contact_sheets/augmentation_{page:02}.jpg';path.parent.mkdir(exist_ok=True);canvas.save(path,quality=92);sheets.append(str(path))
 # Refuse a nominal 200-sample deliverable if any examples were rejected completely.
 write('augmentation_sample_measurements.csv',records);put('augmentation_failures.json',failures)
 real=[dict(id=r['id'],**appearance(v2.normalize(v2.tensors(r),r['role'],h.CONTRAST)[0])) for r in tr]
 write('real_handwriting_appearance.csv',real)
 clean=[dict(id=r['id'],**appearance(v2.tensors(r))) for r in replay[:200]];write('clean_synthetic_appearance.csv',clean)
 # Standalone distribution plots, generated with Pillow (matplotlib is unavailable).
 plot=Image.new('RGB',(1200,760),'white');d=ImageDraw.Draw(plot)
 for panel,(key,limit) in enumerate([('stroke_width',20),('ink_darkness',255)]):
  x0=70+panel*600;y0=680;d.text((x0,20),key,fill='black');d.line((x0,60,x0,y0,x0+500,y0),fill='black',width=2)
  for n,data,color in [('augmented',records,'blue'),('handwriting',real,'red'),('synthetic',clean,'green')]:
   hist,edges=np.histogram([r[key] for r in data],bins=40,range=(0,limit));freq=hist/max(len(data),1)
   pts=[(x0+int((i+.5)*500/40),y0-int(y*550)) for i,y in enumerate(freq)];d.line(pts,fill=color,width=3)
   d.text((x0,40+15*['augmented','handwriting','synthetic'].index(n)),n,fill=color)
  d.text((x0,y0+15),'0',fill='black');d.text((x0+470,y0+15),str(limit),fill='black')
 plot.save(O/'appearance_distributions.png')
 put('augmentation_quality.json',dict(accepted=len(records),failures=len(failures),dot_component_guard=True,all_component_guard=True,body_hole_guard=True,full_ink_support_inside_frame=True,optional_unconfirmed_variants=False,real_training_only=True,darkness_measurement='Median grayscale at distance-transform ridges; blackpoint separately reported'))
 textfile('augmentation_quality_report.md',f'# Augmentation quality\n\nAccepted {len(records)} random examples; {len(failures)} source attempts exhausted rejection sampling and were resampled. Contact sheets pair source and augmented crops. All checks operate at model height 64.\n\nSmall compact detached components are processed separately from skeletonized body ink. Every component must map one-to-one after transformation; total component counts and body holes must match. Hollow/filled detached marks are allowed. This conservative automatic guard is not semantic reader verification; connected dots cannot be independently identified without glyph annotation. Unconfirmed X/7/slanted-bar substitutions are disabled and never silently mixed.\n\nRequested stroke widths 1.5–4 and grey ink 30–200 are sampled; actual distributions are measured, not assumed. Rejection sampling changes the accepted parameter distribution. Whole-tier pitch and scale are used; independent glyph/column annotation is unavailable.\n\n![Distributions](appearance_distributions.png)\n\n'+ '\n'.join(f'![Contact sheet {i}](contact_sheets/augmentation_{i:02}.jpg)' for i in range(10)))
 print('AUGMENTATION QUALITY',len(records),'failures',len(failures),flush=True)
 O=root_output
CONFIGS=[dict(name='cnn',groups=['cnn'],lr_cnn=1e-4,lr_rnn=3e-5,lr_classifier=3e-5),dict(name='cnn_bilstm',groups=['cnn','rnn'],lr_cnn=1e-4,lr_rnn=3e-5,lr_classifier=3e-5),dict(name='all_layers',groups=['cnn','rnn','classifier'],lr_cnn=1e-4,lr_rnn=3e-5,lr_classifier=3e-5)]
def macro(rr):
 return float(np.mean([v2.agg([r for r in rr if r['writer_id']==w])['cer'] for w in sorted({r['writer_id'] for r in rr})]))
def train_run(config,tag,heldout=None,max_steps=400,min_steps=400,patience=4,generated_only=False,without_review=False):
 """Every run starts at frozen E1. All descendants of held-out writers are excluded."""
 c.seed(42);rng=np.random.default_rng(42);rand=random.Random(42)
 originals=[r for r in read(O/'training_inputs.csv') if r['sample_type']=='original'];alltrain=read(O/'training_inputs.csv')
 review={r['tier_id'] for r in read(O/'needs_talim_reader_review.csv')}
 train=[r for r in alltrain if r['writer_id']!=heldout and (not without_review or r['tier_id'] not in review)]
 val=[r for r in originals if r['writer_id']==heldout] if heldout else []
 assert all(r['writer_id'] in v2.TRAIN_WRITERS and r['writer_id']!=heldout for r in train)
 assert not {r['tier_id'] for r in train}.intersection({r['tier_id'] for r in val})
 replay=read(V/'synthetic_replay_inputs.csv');m,v,state=v2.load_model();teacher,_,_=v2.load_model();teacher.requires_grad_(False)
 m.requires_grad_(False);groups=[]
 for name in config['groups']:
  module=getattr(m,name);module.requires_grad_(True);groups.append(dict(params=list(module.parameters()),lr=config['lr_'+name],name=name))
 opt=torch.optim.AdamW(groups,weight_decay=1e-4);targets={};tensorcache={};teacher_cache={}
 def tensor(r):
  if r['id'] not in tensorcache:tensorcache[r['id']]=v2.tensors(r)
  return tensorcache[r['id']]
 def ctc(logits,r):
  if r['id'] not in targets:targets[r['id']]=torch.tensor(v.encode(r['truth']),dtype=torch.long)
  t=targets[r['id']]
  assert logits.shape[0]>=len(t)+sum(a==b for a,b in zip(r['truth'],r['truth'][1:])),r['id']
  return torch.nn.functional.ctc_loss(logits.float().log_softmax(2),t,torch.tensor([logits.shape[0]]),torch.tensor([len(t)]),blank=0,reduction='mean',zero_infinity=False)
 bywriter={w:[r for r in train if r['writer_id']==w and r['sample_type']=='original'] for w in sorted({r['writer_id'] for r in train})}
 children=collections.defaultdict(list)
 for r in train:
  if r['sample_type']=='augmented':children[r['tier_id']].append(r)
 runconfig=dict(config,augmentation_version='safe3',heldout=heldout,max_steps=max_steps,min_steps=min_steps,patience=patience,generated_only=generated_only,without_review=without_review,BN='running statistics update on mixed training domains',warmup=100,clip='first 50 observed norms: no clipping; then rolling 95th percentile times 1.2',KL_weight=.5,loss='CTC mean reduction: negative log likelihood divided by target length',selection='heldout writer CER, then no-whitespace CER, then exact tiers; never W05')
 put('training_configs/'+tag+'.json',runconfig)
 losslog=[];evalhistory=[];norms=[];best=None;stale=0;clip=float('inf');started=time.time();rejections=0
 logfile=O/'training_logs'/(tag+'.jsonl');logfile.parent.mkdir(exist_ok=True)
 assert not logfile.exists(),'Do not overwrite prior run logs'
 bestpath=None;(O/'candidate_checkpoints').mkdir(exist_ok=True)
 for step in range(1,max_steps+1):
  m.train();opt.zero_grad(set_to_none=True)
  for g in opt.param_groups:g['lr']=config['lr_'+g['name']]*min(1,step/100)
  sy=rand.choice(replay);syn=tensor(sy)
  for attempt in range(100):
   try:
    aug,adiag=augmented(syn,rng)
    required=len(sy['truth'])+sum(a==b for a,b in zip(sy['truth'],sy['truth'][1:]))
    if aug.shape[-1]//4<required:rejections+=1;continue
    break
   except RuntimeError:rejections+=1;sy=rand.choice(replay);syn=tensor(sy)
  else:raise RuntimeError('Cannot sample topology- and CTC-safe training augmentation')
  batch=[('synthetic_clean',sy,syn,.5 if generated_only else .25),('synthetic_augmented',sy,aug,.5 if generated_only else .25)]
  if not generated_only:
   writer=rand.choice(sorted(bywriter));r=rand.choice(bywriter[writer]);child=rand.choice(children[r['tier_id']]) if children[r['tier_id']] else r
   batch += [('handwritten_original',r,tensor(r),.25),('handwritten_augmented',child,tensor(child),.25)]
  losses={};klvalue=0.
  for domain,r,t,weight in batch:
   logits=m(t[None]);loss=ctc(logits,r);total=loss
   if domain=='synthetic_clean':
    if r['id'] not in teacher_cache:
     with torch.no_grad():teacher_cache[r['id']]=teacher(t[None]).float().softmax(2)
    kl=torch.nn.functional.kl_div(logits.float().log_softmax(2),teacher_cache[r['id']],reduction='none').sum(2).mean();total=loss+.5*kl;klvalue=float(kl.detach())
   assert torch.isfinite(total),tag
   (weight*total).backward();losses[domain]=float(loss.detach())
  norms_by_group={g['name']:sum(float(p.grad.detach().float().pow(2).sum()) for p in g['params'] if p.grad is not None)**.5 for g in opt.param_groups}
  norm=sum(x*x for x in norms_by_group.values())**.5;assert np.isfinite(norm)
  norms.append(norm)
  if step==51 or (step>51 and step%100==1):clip=float(np.quantile(norms[-200:],.95)*1.2)
  clipped=norm>clip
  if clipped:torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],clip)
  opt.step()
  row=dict(step=step,seconds=time.time()-started,**losses,KL=klvalue,gradient_norm=norm,gradient_by_group=norms_by_group,clip_threshold=clip if np.isfinite(clip) else None,clipped=clipped,augmentation_rejections=rejections,exposures=[dict(domain=domain,id=r['id'],writer_id=r.get('writer_id','synthetic'),tier_id=r.get('tier_id',r['id']),weight=weight) for domain,r,t,weight in batch])
  losslog.append(row)
  with logfile.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
  if step%20==0:print('TRAIN',tag,step,'loss',losses,'clip_rate',np.mean([r['clipped'] for r in losslog]),'seconds',round(time.time()-started),flush=True)
  if step%100==0 or step==max_steps:
   m.eval()
   if val:
    rr=evalrows(m,v,val);metrics=v2.agg(rr);key=(metrics['cer'],metrics['cer_no_whitespace'],-metrics['exact_tiers'])
   else:
    devrows=read(V/'regression_inputs.csv')
    controls=[r for cohort in ['clean_historical_regression','positive_development'] for role in ['upper','count','lower'] for r in [x for x in devrows if x['cohort']==cohort and x['role']==role][:20]]
    rr=evalrows(m,v,controls);metrics=v2.agg(rr);key=(metrics['cer'],metrics['cer_no_whitespace'],-metrics['exact_tiers'])
   # Fixed originals/replay probes in eval mode distinguish BN effects from train loss.
   probes=[rand.choice(bywriter[w]) for w in sorted(bywriter)] if not generated_only else []
   with torch.inference_mode():probe_hand=np.mean([h.result_row(r,'',m(tensor(r)[None])[:,0].detach().softmax(1).numpy(),v)['loss'] for r in probes]) if probes else None
   event=dict(step=step,metrics=metrics,handwritten_eval_per_character_CTC=float(probe_hand) if probe_hand is not None else None,clipping_rate=float(np.mean([r['clipped'] for r in losslog])))
   evalhistory.append(event);put('training_logs/'+tag+'_evaluation.json',evalhistory)
   if best is None or key<best:
    best=key;stale=0
    bestpath=O/'candidate_checkpoints'/(tag+'_step_'+str(step)+'.pt');assert not bestpath.exists()
    save={k:state[k] for k in ['char_to_idx','vocabulary_hash']};save.update(model_state_dict=m.state_dict(),V5_config=runconfig,V5_step=step,V5_initial_sha256=v2.FROZEN_SHA,format_version='stage2g_v5_inference_checkpoint',torch_version=str(torch.__version__))
    torch.save(save,bestpath);write('predictions/'+tag+'_heldout.csv',rr)
   else:stale+=1
   if step>=min_steps and stale>=patience:break
 put('training_logs/'+tag+'_completion.json',dict(steps=step,wall_seconds=time.time()-started,clipping_rate=float(np.mean([r['clipped'] for r in losslog])),best_key=best,early_stopped=step<max_steps,checkpoint=str(bestpath),sha256=h.sha(bestpath)))
 return bestpath
def pilots():
 assert (O/'quality_safe3/augmentation_quality.json').exists()
 for config in CONFIGS:
  train_run(config,'pilot_safe3_'+config['name'],heldout='W04',max_steps=400,min_steps=400)
def pilot_assessment():
 """Full regression is a capacity safety check, not W05 tuning."""
 dev=read(V/'regression_inputs.csv');tr=[r for r in read(O/'training_inputs.csv') if r['sample_type']=='original' and r['writer_id']!='W04']
 em,v,_=v2.load_model();base=evalrows(em,v,dev);basereg=h.regression(base,dev);gate=json.loads((V/'training_config.json').read_text())['gates'];out=[]
 for conf in CONFIGS:
  tag='pilot_safe3_'+conf['name'];completion=json.loads((O/'training_logs'/ (tag+'_completion.json')).read_text());m,_,_=v2.load_model(completion['checkpoint']);m.eval()
  rr=evalrows(m,v,dev);reg=h.regression(rr,dev);checks={}
  for cohort,b in basereg.items():
   limit=gate['clean_cer_delta'] if cohort.startswith('clean') else gate['long_cer_delta'] if cohort.startswith('long') else gate['positive_cer_delta']
   checks[cohort+'_CER']=reg[cohort]['cer']<=b['cer']+limit+1e-12
   if cohort.startswith('clean'):checks[cohort+'_exact']=reg[cohort]['exact_accuracy']>=b['exact_accuracy']-gate['clean_exact_drop']-1e-12
  hand=evalrows(m,v,tr);write('predictions/'+tag+'_training_originals.csv',hand);write('predictions/'+tag+'_synthetic.csv',rr)
  loss=v2.agg(hand)['mean_ctc_loss'];out.append(dict(configuration=conf['name'],handwritten_loss_per_character=loss,synthetic_gate_checks=checks,synthetic_gates_passed=all(checks.values()),capacity_success=loss<3 and all(checks.values()),regression=reg,clipping_rate=completion['clipping_rate']))
 put('pilot_assessment.json',out);print('PILOT ASSESSMENT',out,flush=True)
def folds():
 # Exactly three architectural configurations, four writer folds each. No W05 reads.
 for conf in CONFIGS:
  for writer in sorted(v2.TRAIN_WRITERS):
   tag='LOWO_'+conf['name']+'_'+writer
   if not (O/'training_logs'/(tag+'_completion.json')).exists():train_run(conf,tag,heldout=writer,max_steps=10000,min_steps=2000,patience=5)
def fold_selection():
 allrows=[];summaries=[]
 for conf in CONFIGS:
  results=[]
  for writer in sorted(v2.TRAIN_WRITERS):
   tag='LOWO_'+conf['name']+'_'+writer;comp=json.loads((O/'training_logs'/(tag+'_completion.json')).read_text());rr=read(O/'predictions'/(tag+'_heldout.csv'));metrics=v2.agg(rr)
   allrows.append(dict(configuration=conf['name'],heldout_writer=writer,**metrics,checkpoint=comp['checkpoint'],steps=comp['steps']));results.append(metrics)
  summaries.append(dict(configuration=conf['name'],macro_writer_CER=float(np.mean([r['cer'] for r in results])),macro_writer_CER_no_whitespace=float(np.mean([r['cer_no_whitespace'] for r in results])),macro_writer_exact_accuracy=float(np.mean([r['exact_accuracy'] for r in results]))))
 selected=min(summaries,key=lambda r:(r['macro_writer_CER'],r['macro_writer_CER_no_whitespace'],-r['macro_writer_exact_accuracy']))
 config=next(r for r in CONFIGS if r['name']==selected['configuration'])
 # Fix final duration before W05: median selected fold's best-step count.
 steps=[]
 for r in allrows:
  if r['configuration']==config['name']:
   cp=torch.load(r['checkpoint'],map_location='cpu',weights_only=True);steps.append(cp['V5_step'])
 selected['final_training_steps']=max(2000,int(np.median(steps)))
 put('selection_before_W05.json',dict(selected=selected,configurations=summaries,criterion='four-fold writer-macro CER; no W05 data',timestamp=time.time()))
 write('writer_independent_results.csv',allrows)
 for writer in sorted(v2.TRAIN_WRITERS):
  tag='review_ablation_'+config['name']+'_'+writer
  if not (O/'training_logs'/(tag+'_completion.json')).exists():train_run(config,tag,heldout=writer,max_steps=10000,min_steps=2000,patience=5,without_review=True)
  rr=read(O/'predictions'/(tag+'_heldout.csv'));completion=json.loads((O/'training_logs'/(tag+'_completion.json')).read_text());allrows.append(dict(configuration=config['name']+'_without_9_review_parents',heldout_writer=writer,**v2.agg(rr),checkpoint=completion['checkpoint'],steps=completion['steps']))
 write('writer_independent_results.csv',allrows)
 # No handwritten validation is used in final all-writer fitting: duration is already fixed.
 tag='final_'+config['name']
 if not (O/'training_logs'/(tag+'_completion.json')).exists():train_run(config,tag,max_steps=selected['final_training_steps'],min_steps=selected['final_training_steps'],patience=100000)
def generated_only():
 # Fixed protocol uses all layers and 2,000 updates, synthetic-only weight updates.
 tag='generated_only_safe3_all_layers'
 if not (O/'training_logs'/(tag+'_completion.json')).exists():train_run(CONFIGS[2],tag,max_steps=2000,min_steps=2000,patience=100000,generated_only=True)
 completion=json.loads((O/'training_logs'/(tag+'_completion.json')).read_text());m,v,_=v2.load_model(completion['checkpoint']);m.eval()
 tr=[r for r in read(O/'training_inputs.csv') if r['sample_type']=='original'];va=read(O/'W05_evaluation_inputs.csv')
 result={}
 for name,rows in [('unseen_training_writer_originals',tr),('W05_14',va)]:
  rr=evalrows(m,v,rows);write('predictions/generated_only_'+name+'.csv',rr);result[name]=v2.summary(rr)
 put('generated_only_results.json',result)
def final_evaluation():
 assert not (O/'final_evaluation.json').exists(),'W05 final candidate may be evaluated only once'
 selection=json.loads((O/'selection_before_W05.json').read_text());tag='final_'+selection['selected']['configuration'];completion=json.loads((O/'training_logs'/(tag+'_completion.json')).read_text())
 m,v,_=v2.load_model(completion['checkpoint']);m.eval();va=read(O/'W05_evaluation_inputs.csv');dev=read(V/'regression_inputs.csv')
 old=json.loads((V/'final_comparison.json').read_text());base=old['approaches']['A'];gates=json.loads((V/'training_config.json').read_text())['gates'];result={};synflat=read(O/'synthetic_regression_results.csv');errors=[]
 def features(ch):
  n=next((n for n,g in h.rules.COUNT_ENCODING.items() if g==ch),None)
  if n is None:return None
  return dict(oval=n>=10,dots=(n%100)//10 if n>=10 else 0,cross=n//100 if n>=10 else 0)
 for mask in [False,True]:
  name='final_role_mask' if mask else 'final_raw_decoder'
  rr=evalrows(m,v,va,restricted=mask,cache=O/'posteriors/final_W05');write('predictions/'+name+'_W05.csv',rr)
  val=v2.summary(rr);sub=v2.summary([r for r in rr if r['id']!='all_003801_row_00_count'])
  near={str(n):[r['id'] for r in rr if r['edit_distance']==n] for n in [1,2]}
  regrows=evalrows(m,v,dev,restricted=mask,cache=O/'posteriors/final_synthetic');write('predictions/'+name+'_synthetic.csv',regrows);reg=h.regression(regrows,dev)
  checks=v2.gates(val,reg,base['validation'],base['regression'],gates)
  result[name]=dict(W05_14=val,W05_13=sub,near_exact=near,regression=reg,unchanged_gates=checks)
  for cohort,metrics in cohorts(regrows,dev).items():synflat.append(dict(model=name,cohort=cohort,**metrics))
  for r in rr:
   if r['role']!='count':continue
   for op,i,j,a,b in h.align(r['truth'],r['prediction']):
    if op=='match':continue
    fa,fb=features(a),features(b);oval=bool(fa and fb and fa['oval']!=fb['oval']);cross=bool(fa and fb and fa['cross']!=fb['cross']);dots=bool(fa and fb and fa['dots']!=fb['dots'])
    typ='missed_glyph' if op=='deletion' else 'extra_glyph' if op=='insertion' else 'oval_vs_plain_shape' if oval else 'wrong_cross_type' if cross else 'wrong_dot_count' if dots else 'plain_shape_confusion'
    errors.append(dict(model=name,id=r['id'],operation=op,truth_position=i,prediction_position=j,reference=a,predicted=b,primary_type=typ,oval_mismatch=oval,dot_count_mismatch=dots,cross_type_mismatch=cross))
 write('error_breakdown.csv',errors);write('synthetic_regression_results.csv',synflat)
 put('final_evaluation.json',dict(checkpoint=completion['checkpoint'],sha256=h.sha(completion['checkpoint']),selection_sha256=h.sha(O/'selection_before_W05.json'),results=result,operational_default='frozen_E1' if not result['final_role_mask']['unchanged_gates']['passed'] else 'candidate_eligible_for_review',deployed=False))
 comparison=[]
 frozen=read(V/'predictions/A_frozen_original.csv');role=read(h.O/'predictions/F1_role_mask_W05.csv')
 diagnostic=[('frozen_E1',frozen),('V3_role_aware_baseline',role),('V2_D',read(O/'predictions/D_raw_decoder_W05.csv')),('V2_D_role_mask',read(O/'predictions/D_role_mask_W05.csv')),('generated_only',read(O/'predictions/generated_only_W05_14.csv')),('final_raw',read(O/'predictions/final_raw_decoder_W05.csv')),('final_role_mask',read(O/'predictions/final_role_mask_W05.csv'))]
 for name,rr in diagnostic:comparison.append(dict(model=name,**v2.agg(rr)))
 write('final_comparison.csv',comparison)
 protected=json.loads((O/'protected_inputs.json').read_text());assert all(h.sha(p)==s for p,s in protected.items())
 put('final_integrity.json',dict(previous_results_and_checkpoints_unchanged=True,labels_unchanged=True,no_Git_operations=True,no_W02_W07_images_opened=True,selection_fixed_before_final_W05=True))
 lines=['# V5 final comparison','', '| Model | W05 CER | No-whitespace CER | Exact / 14 |','|---|---:|---:|---:|']
 lines += [f"| {r['model']} | {r['cer']:.4%} | {r['cer_no_whitespace']:.4%} | {r['exact_tiers']} |" for r in comparison]
 passed=result['final_role_mask']['unchanged_gates']['passed'];candidate=result['final_role_mask']['W05_14']['overall'];generated=json.loads((O/'generated_only_results.json').read_text())
 lines+=['',f"Promotion gates passed: **{passed}**. Operational files were not changed; frozen E1 remains the default when any gate fails.",'',f"Generated-only W05 CER: {generated['W05_14']['overall']['cer']:.4%}; training-writer originals CER: {generated['unseen_training_writer_originals']['overall']['cer']:.4%}. Both were unseen during weight updates.",'','Writer-independent configuration selection and the nine-parent review ablation are in writer_independent_results.csv. W01 is not allowed to dominate the writer-macro criterion. V2-D was fitted to all four training writers; its training-original diagnostics are not unbiased LOWO estimates.','', 'Final per-role exact counts: '+str({k:x['exact_tiers'] for k,x in result['final_role_mask']['W05_14']['per_role'].items()}), '', 'One/two-edit tiers: '+str(result['final_role_mask']['near_exact']), '', 'Count errors: '+str(dict(collections.Counter(r['primary_type'] for r in errors if r['model']=='final_role_mask'))),'','Count error feature categories use the existing canonical numeric mapping (plain 1–9, oval groups, tens dots and hundreds cross group); they are label-based categories, not independent image-level reader diagnoses.','', 'Augmentation limitations: automatic compact-component dot identification, conservative rejection, whole-tier affine scale/pitch rather than independently annotated glyph transformations, and disabled unconfirmed writer variants. These limit causal claims about exact stroke mechanisms.','', ('Appearance adaptation improved W05 CER relative to frozen E1.' if candidate['cer']<v2.agg(frozen)['cer'] else 'Appearance adaptation did not improve W05 CER relative to frozen E1.')+' Generalization is supported only to the extent shown by the four held-out training writers and one final W05 writer; no population-wide claim is established. Exact-tier blockers are listed in error_breakdown.csv.']
 textfile('final_comparison_report.md','\n'.join(lines)+'\n')
if __name__=='__main__':
 c.seed(42)
 for arg in sys.argv[1:]:
  {'integrity':integrity,'phase1':phase1,'augmentation':augmentation_quality,'pilots':pilots,'pilot_assessment':pilot_assessment,'folds':folds,'selection':fold_selection,'generated_only':generated_only,'final':final_evaluation}[arg]()
