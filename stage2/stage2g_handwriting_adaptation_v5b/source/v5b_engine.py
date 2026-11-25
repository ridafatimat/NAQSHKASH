"""NAQSHKASH V5b: safe4 augmentation and resumable CPU experiments."""
import sys,os,json,csv,time,random,hashlib,collections,math,importlib.util,argparse
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path('C:/NAQSHKASH/stage2/stage2g_handwriting_adaptation_v5b')
V5=ROOT.parent/'stage2g_handwriting_adaptation_v5'
spec=importlib.util.spec_from_file_location('legacy_readonly',V5/'src/stage2g_v5.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
c=legacy.c;h=legacy.h;v2=legacy.v2;torch=legacy.torch;np=legacy.np;cv=legacy.cv
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 for r in rows:
  if 'loss' in r:r['loss']=float(r['loss'])
 return rows
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(str(tmp),str(p))
def write(p,rows):
 if not rows:return
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def config():return json.loads((ROOT/'local_run/config.json').read_text(encoding='utf-8-sig'))
def inputs():
 rows=read(V5/'training_inputs.csv');counts=collections.Counter(r['sample_type'] for r in rows)
 assert counts=={'original':105,'augmented':2413},counts
 assert {r['writer_id'] for r in rows}=={'W01','W03','W04','W06'}
 assert not any(r['tier_id'] in {'all_004383_row_01_count','all_004383_row_02_count','all_004254_row_02_count'} or r['id']=='all_004383_row_03_count__phase3__partial_crop' for r in rows)
 replay=read(legacy.V/'synthetic_replay_inputs.csv');dev=read(legacy.V/'regression_inputs.csv')
 assert all(r['split']=='train' for r in replay),'Synthetic replay must be TRAIN only'
 assert {r['parent_id'] for r in replay}.isdisjoint({r['parent_id'] for r in dev}),'Replay/regression parent leakage'
 return rows,replay,dev
def holes(mask):
 _,tree=cv.findContours((mask>0).astype(np.uint8),cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
 return sum(int(e[3]>=0) for e in tree[0]) if tree is not None else 0
def components(image,threshold=None):
 if threshold is None:
  a=image.astype(np.float32)/255;fg=a[a<.98];black=float(np.percentile(fg,5)) if len(fg) else 0;threshold=(1+black)*127.5
 return legacy.component_info(image,threshold)
def compact_ids(stats,ids,real=False):
 return [i for i in ids if 6<=stats[i,4]<=650 and 3<=min(stats[i,2:4]) and max(stats[i,2:4])<=28 and min(stats[i,2:4])/max(stats[i,2:4])>=.65 and stats[i,4]/max(1,stats[i,2]*stats[i,3])>=.65]
def appearance(t):
 out=legacy.appearance(t);a=np.uint8(np.clip(t.numpy()[0]*255,0,255));mask,labels,stats,ids=components(a);dots=compact_ids(stats,ids,True)
 out['dot_diameters']=[float(math.sqrt(stats[i,2]*stats[i,3])) for i in dots]
 return out
class ImageCache:
 def __init__(self,limit):self.limit=limit;self.data=collections.OrderedDict()
 def get(self,r):
  key=r['id']
  if key in self.data:self.data.move_to_end(key);return self.data[key]
  if 'writer_id' in r:assert r['writer_id'] in {'W01','W03','W04','W06','W05'}
  value=v2.tensors(r);self.data[key]=value
  if len(self.data)>self.limit:self.data.popitem(last=False)
  return value
class Safe4:
 def __init__(self,cfg,darkness_reference):
  self.cfg=cfg;self.darkness=np.array(darkness_reference,dtype=float);self.prepared=collections.OrderedDict();self.counters=collections.Counter()
 def prepare(self,t,key):
  if key in self.prepared:self.prepared.move_to_end(key);return self.prepared[key]
  raw=np.uint8(np.clip(t.numpy()[0]*255,0,255));mask,labels,stats,ids=components(raw,128);dots=compact_ids(stats,ids)
  body=mask.copy();body[np.isin(labels,dots)]=0;sk=legacy.skeleton(body)
  data=(raw,mask,labels,stats,ids,dots,body,cv.distanceTransform((1-sk).astype(np.uint8),cv.DIST_L2,5))
  self.prepared[key]=data
  if len(self.prepared)>self.cfg['prepared_cache_size']:self.prepared.popitem(last=False)
  return data
 def sample(self,t,rng,key):
  from scipy.ndimage import gaussian_filter
  raw,mask,labels,stats,ids,dots,body,dist=self.prepare(t,key);H,W=raw.shape
  for attempt in range(self.cfg['max_attempts']):
   self.counters['proposals']+=1
   width=float(rng.uniform(*self.cfg['stroke_width']))
   dark=float(np.clip(rng.choice(self.darkness)+self.cfg['darkness_offset'],*self.cfg['darkness_range']))
   ink=(dist<=width/2).astype(np.float32);style=str(rng.choice(['filled','hollow']));yy0,xx0=np.indices(raw.shape,dtype=np.float32)
   for i in dots:
    x,y,bw,bh,area=stats[i];cx=x+(bw-1)/2;cy=y+(bh-1)/2;diam=float(rng.uniform(*self.cfg['source_dot_diameter']));radius=diam/2
    distance=np.sqrt((xx0-cx)**2+(yy0-cy)**2);dm=np.clip(radius+.5-distance,0,1)
    if style=='hollow' and radius-width>.5:dm*=np.clip(distance-(radius-width)+.5,0,1)
    ink=np.maximum(ink,dm)
   scale=float(rng.uniform(*self.cfg['scale']));pitch=float(rng.uniform(*self.cfg['pitch']));angle=float(rng.uniform(*self.cfg['rotation']));slant=float(rng.uniform(*self.cfg['slant']))
   outW=max(16,round(W*pitch));rad=np.deg2rad(angle);linear=np.array([[np.cos(rad)*scale*pitch,np.sin(rad)*scale*pitch+slant],[-np.sin(rad)*scale,np.cos(rad)*scale]],np.float32)
   mat=np.column_stack([linear,np.array([outW/2,32])-linear@np.array([W/2,32])]);support=np.column_stack(np.where(ink>.01))[:,::-1].astype(np.float32);projected=support@mat[:,:2].T+mat[:,2]
   if not len(projected) or np.any(projected[:,0]<1) or np.any(projected[:,0]>outW-2) or np.any(projected[:,1]<1) or np.any(projected[:,1]>62):self.counters['boundary_rejections']+=1;continue
   warped=cv.warpAffine(ink,mat,(outW,64),flags=cv.INTER_LINEAR,borderValue=0);yy,xx=np.indices(warped.shape,dtype=np.float32)
   dx=gaussian_filter(rng.normal(size=warped.shape).astype(np.float32),self.cfg['elastic_sigma'])*rng.uniform(0,self.cfg['elastic_amplitude'])
   dy=gaussian_filter(rng.normal(size=warped.shape).astype(np.float32),self.cfg['elastic_sigma'])*rng.uniform(0,self.cfg['elastic_amplitude'])
   warped=cv.remap(warped,xx+dx,yy+dy,cv.INTER_LINEAR,borderMode=cv.BORDER_CONSTANT,borderValue=0);image=255-warped*(255-dark)
   blur=bool(rng.random()<self.cfg['blur_probability'])
   if blur:image=cv.GaussianBlur(image,(3,3),float(rng.uniform(*self.cfg['blur_sigma'])))
   lowres=bool(rng.random()<self.cfg['low_resolution_probability'])
   if lowres:
    ratio=float(rng.uniform(*self.cfg['resolution_ratio']));small=cv.resize(image,None,fx=ratio,fy=ratio,interpolation=cv.INTER_AREA);image=cv.resize(small,(outW,64))
   gap=0
   if rng.random()<self.cfg['ruled_gap_probability']:
    gap=int(rng.integers(1,4));y=int(rng.integers(5,59));image[y:y+gap]=255
   image=np.uint8(np.clip(image,0,255));jpeg=bool(rng.random()<self.cfg['jpeg_probability'])
   if jpeg:
    ok,b=cv.imencode('.jpg',image,[cv.IMWRITE_JPEG_QUALITY,int(rng.integers(*self.cfg['jpeg_quality']))]);image=cv.imdecode(b,0)
   am,al,ast,aids=components(image,(255+dark)/2)
   self.counters['visibility_checked']+=1
   if any(int(image[al==i].min())>self.cfg['component_darkest_max'] for i in aids):self.counters['visibility_rejections']+=1;continue
   if len(aids)!=len(ids):self.counters['component_count_rejections']+=1;continue
   matched=[];safe=True
   for i in ids:
    pts=np.column_stack(np.where(labels==i))[:,::-1].astype(np.float32);pts=pts@mat[:,:2].T+mat[:,2];x=np.clip(np.rint(pts[:,0]).astype(int),0,outW-1);y=np.clip(np.rint(pts[:,1]).astype(int),0,63);hits=al[y,x];hits=hits[hits>0]
    if not len(hits):safe=False;break
    k=int(np.bincount(hits).argmax())
    if k not in aids or k in matched:safe=False;break
    matched.append(k)
   if not safe:self.counters['correspondence_rejections']+=1;continue
   augbody=am.copy();diameters=[]
   for i,k in zip(ids,matched):
    if i in dots:augbody[al==k]=0;diameters.append(float(math.sqrt(ast[k,2]*ast[k,3])))
   if any(not self.cfg['accepted_dot_diameter'][0]<=d<=self.cfg['accepted_dot_diameter'][1] for d in diameters):self.counters['dot_diameter_rejections']+=1;continue
   if holes(body)!=holes(augbody):self.counters['hole_rejections']+=1;continue
   self.counters['accepted']+=1
   return torch.from_numpy(image.astype(np.float32)/255)[None],dict(attempts=attempt+1,target_stroke=width,target_darkness=dark,scale=scale,pitch=pitch,rotation=angle,slant=slant,dot_style=style,dot_diameters=diameters,source_components=len(ids),output_components=len(aids),component_darkest_max=max([int(image[al==i].min()) for i in aids] or [255]),ruled_gap=gap,blur=blur,low_resolution=lowres,jpeg=jpeg)
  self.counters['exhausted_sources']+=1;raise RuntimeError('safe4 rejection sampling exhausted for '+key)
def all_metrics(rows):
 result=v2.summary(rows);result['per_writer']={w:v2.agg([r for r in rows if r['writer_id']==w]) for w in sorted({r['writer_id'] for r in rows})}
 result['macro_writer']={k:float(np.mean([r[k] for r in result['per_writer'].values()])) for k in ['cer','cer_no_whitespace','exact_accuracy','mean_ctc_loss']}
 return result
def evaluate(m,v,rows,outdir,stem,save_posteriors=False):
 outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True);raw=[];masked=[]
 m.eval()
 for r in rows:
  if 'writer_id' in r:assert r['writer_id'] in {'W01','W03','W04','W06','W05'}
  posterior_path=outdir/(stem+'_posteriors')/(r['id']+'.npz') if save_posteriors else None
  if posterior_path and posterior_path.exists():prob=np.load(posterior_path)['probabilities']
  else:
   with torch.inference_mode():prob=m(v2.tensors(r)[None])[:,0].float().softmax(1).numpy()
  for restrict,dest in [(False,raw),(True,masked)]:
   row=h.result_row(r,h.decode(prob,v,r['role'],restrict),prob,v);row['raw_prediction']=h.decode(prob,v,r['role'],False);dest.append(row)
  if posterior_path and not posterior_path.exists():
   posterior_path.parent.mkdir(exist_ok=True);np.savez_compressed(posterior_path,probabilities=prob)
 write(outdir/(stem+'_raw.csv'),raw);write(outdir/(stem+'_role_mask.csv'),masked)
 metrics={'raw':all_metrics(raw),'role_mask':all_metrics(masked)};dump(outdir/(stem+'_metrics.json'),metrics)
 return metrics,raw,masked
def count_errors(rows,model):
 out=[];mapping={g:n for n,g in h.rules.COUNT_ENCODING.items() if g}
 def feat(ch):
  n=mapping.get(ch)
  return None if n is None else dict(oval=n>=10,dots=(n%100)//10 if n>=10 else 0,cross=n//100 if n>=10 else 0)
 for r in rows:
  if r['role']!='count':continue
  for op,i,j,a,b in h.align(r['truth'],r['prediction']):
   if op=='match':continue
   x,y=feat(a),feat(b)
   typ='missed_glyph' if op=='deletion' else 'extra_glyph' if op=='insertion' else 'out_of_role' if x is None or y is None else 'oval_vs_plain_shape' if x['oval']!=y['oval'] else 'wrong_cross_type' if x['cross']!=y['cross'] else 'wrong_dot_count' if x['dots']!=y['dots'] else 'plain_shape_confused'
   out.append(dict(model=model,id=r['id'],writer_id=r['writer_id'],operation=op,reference=a,predicted=b,type=typ,reference_position=i,prediction_position=j))
 return out
def synthetic_metrics(rows,dev):return h.regression(rows,dev)
def synthetic_gates(reg,base,cfg,no_whitespace=False,clean_only=False):
 checks={};key='cer_no_whitespace' if no_whitespace else 'cer'
 for cohort,b in base.items():
  if clean_only and not cohort.startswith('clean'):continue
  limit=cfg['clean_cer_delta'] if cohort.startswith('clean') else cfg['long_cer_delta'] if cohort.startswith('long') else cfg['positive_cer_delta']
  checks[cohort+'_'+key]=reg[cohort][key]<=b[key]+limit+1e-12
  if cohort.startswith('clean'):checks[cohort+'_exact']=reg[cohort]['exact_accuracy']>=b['exact_accuracy']-cfg['clean_exact_drop']-1e-12
 return dict(passed=all(checks.values()),checks=checks)
def choose_initial(cfg,generated=False,smoke=False):
 frozen=c.E/'checkpoints/selected_model/best.pt';assert sha(frozen)==cfg['frozen_e1_sha256']
 if generated or smoke or cfg['initialization']=='frozen_e1':return frozen
 decision_path=(ROOT/'local_run'/cfg['generated_decision_path']).resolve()
 if not decision_path.exists():
  if cfg['initialization']=='step2':raise RuntimeError('Step 2 decision is unavailable')
  return frozen
 decision=json.loads(decision_path.read_text(encoding='utf-8'))
 if cfg['initialization']=='step2' and not decision['passed']:raise RuntimeError('Step 2 failed; explicit Step 2 initialization is not eligible')
 if cfg['initialization'] in ['auto_step2','step2'] and decision['passed']:
  assert sha(decision['checkpoint'])==decision['checkpoint_sha256'];return Path(decision['checkpoint'])
 if cfg['initialization']!='auto_step2':raise ValueError('Unknown initialization switch')
 return frozen
def train_run(folder,cfg,mode,groups,heldout=None,max_updates=None,smoke=False,resume=False):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 completion_path=folder/'completion.json'
 if completion_path.exists():
  previous=json.loads((folder/'run_config.json').read_text())
  assert previous['config']==cfg and previous['groups']==groups and previous['heldout_writer']==heldout and previous['mode']==mode and previous['smoke']==smoke,'Completed run settings changed; start a new run'
  return json.loads(completion_path.read_text())
 alltrain,replay,dev=inputs();originals=[r for r in alltrain if r['sample_type']=='original'];generated=mode=='generated';final=mode=='final'
 params=cfg['generated'] if generated else cfg['folds'];max_updates=max_updates or params['max_updates'];minimum=0 if generated or smoke else params['min_updates'];evaluate_every=20 if smoke else params['evaluate_every'];patience=params['patience'];batch_spec=params['batch']
 val=originals if generated else [r for r in originals if r['writer_id']==heldout] if heldout else []
 tr=[r for r in alltrain if r['writer_id']!=heldout];assert all(r['writer_id'] in {'W01','W03','W04','W06'} and r['writer_id']!=heldout for r in tr)
 assert not heldout or {r['tier_id'] for r in tr}.isdisjoint({r['tier_id'] for r in val})
 c.seed(cfg['seed']);torch.set_num_threads(cfg['threads']);rng=np.random.default_rng(cfg['seed']);rand=random.Random(cfg['seed'])
 initial=choose_initial(cfg,generated,smoke);m,v,metadata=v2.load_model(initial);teacher,_,_=v2.load_model();teacher.requires_grad_(False);teacher.eval();m.requires_grad_(False)
 optimizer_groups=[]
 for name in groups:
  module=getattr(m,name);module.requires_grad_(True);optimizer_groups.append(dict(params=list(module.parameters()),lr=cfg['learning_rates'][name],name=name))
 opt=torch.optim.AdamW(optimizer_groups,weight_decay=cfg['weight_decay'])
 accepted=cfg['augmentation'];darkness=json.loads((ROOT/'quality_safe4/darkness_reference.json').read_text());sampler=Safe4(accepted,darkness);cache=ImageCache(cfg['image_cache_size']);teacher_cache=collections.OrderedDict();targets={}
 bywriter={w:[r for r in tr if r['writer_id']==w and r['sample_type']=='original'] for w in sorted({r['writer_id'] for r in tr})};augwriter={w:[r for r in tr if r['writer_id']==w and r['sample_type']=='augmented'] for w in bywriter}
 runconfig=dict(config=cfg,mode=mode,groups=groups,heldout_writer=heldout,smoke=smoke,max_updates=max_updates,effective_batch=batch_spec,initial_checkpoint=str(initial),initial_sha256=sha(initial),E1_teacher_sha256=cfg['frozen_e1_sha256'],KL_formula='mean(CTC/target-length across 16) + 0.5 * mean(KL across 4 clean tiers)',BN='updated for every accumulated microbatch',training_originals=0 if generated else sum(r['sample_type']=='original' for r in tr),training_augmented=0 if generated else sum(r['sample_type']=='augmented' for r in tr),heldout_descendants_excluded=bool(heldout))
 latest_path=folder/'last_checkpoint.json';state=dict(step=0,best_key=None,best_primary=None,best_checkpoint=None,best_step=None,stale=0,norms=[],clip_threshold=None,clipped_steps=0,elapsed_seconds=0.,evaluation_history=[])
 if latest_path.exists():
  if not resume:raise RuntimeError('Existing run requires --resume: '+str(folder))
  last=json.loads(latest_path.read_text());assert sha(last['checkpoint'])==last['sha256'];saved=torch.load(last['checkpoint'],map_location='cpu',weights_only=True)
  assert saved['run_config']==runconfig,'Configuration changed; use a new run folder'
  m.load_state_dict(saved['model_state_dict']);opt.load_state_dict(saved['optimizer_state_dict']);state=saved['run_state'];rng.bit_generator.state=saved['numpy_rng'];rand.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng']);sampler.counters.update(saved['augmentation_counters'])
 else:
  assert not any(folder.glob('attempt_*')),'No resumable checkpoint exists; preserve this folder and start a fresh run'
  dump(folder/'run_config.json',runconfig)
 attempt=len(list(folder.glob('attempt_*')))+1;segment=folder/f'attempt_{attempt:03}';segment.mkdir();logpath=segment/'training.jsonl';started=time.time();base_elapsed=state['elapsed_seconds']
 def log(record):
  with logpath.open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
 log(dict(event='start' if state['step']==0 else 'resume',from_step=state['step'],attempt=attempt,config_sha256=sha(ROOT/'local_run/config.json')))
 def ctc(logits,r):
  if r['id'] not in targets:targets[r['id']]=torch.tensor(v.encode(r['truth']),dtype=torch.long)
  target=targets[r['id']];required=len(target)+sum(a==b for a,b in zip(r['truth'],r['truth'][1:]));assert logits.shape[0]>=required,r['id']
  return torch.nn.functional.ctc_loss(logits.float().log_softmax(2),target,torch.tensor([logits.shape[0]]),torch.tensor([len(target)]),blank=0,reduction='mean',zero_infinity=False)
 step=state['step']
 resume_already_stopped=bool(val and step>=minimum and state['stale']>=patience)
 try:
  for step in ([] if resume_already_stopped else range(state['step']+1,max_updates+1)):
   m.train();opt.zero_grad(set_to_none=True)
   for g in opt.param_groups:g['lr']=cfg['learning_rates'][g['name']]*min(1,step/cfg['warmup_steps'])
   batch=[]
   for domain,n in batch_spec.items():
    for k in range(n):
     if domain.startswith('synthetic'):
      for retry in range(200):
       r=rand.choice(replay);t=cache.get(r)
       if domain=='synthetic_augmented':
        try:t,diag=sampler.sample(t,rng,r['id'])
        except RuntimeError:continue
        required=len(r['truth'])+sum(a==b for a,b in zip(r['truth'],r['truth'][1:]))
        if t.shape[-1]//4<required:sampler.counters['CTC_length_rejections']+=1;continue
       break
      else:raise RuntimeError('Unable to sample safe4 tier')
     else:
      assert not generated
      writer=rand.choice(sorted(bywriter));pool=bywriter[writer] if domain=='handwritten_original' else augwriter[writer];assert pool,writer;r=rand.choice(pool);t=cache.get(r)
      assert r['writer_id']!=heldout and r['split']=='train'
     batch.append((domain,r,t))
   assert len(batch)==cfg['batch_size'];rand.shuffle(batch);losses=collections.defaultdict(list);klvalues=[]
   for domain,r,t in batch:
    logits=m(t[None]);loss=ctc(logits,r);total=loss/cfg['batch_size']
    if domain=='synthetic_clean':
     if r['id'] not in teacher_cache:
      with torch.no_grad():teacher_cache[r['id']]=teacher(t[None]).float().softmax(2).detach()
      if len(teacher_cache)>cfg['teacher_cache_size']:teacher_cache.popitem(last=False)
     else:teacher_cache.move_to_end(r['id'])
     kl=torch.nn.functional.kl_div(logits.float().log_softmax(2),teacher_cache[r['id']],reduction='none').sum(2).mean();total=total+cfg['KL_weight']*kl/batch_spec['synthetic_clean'];klvalues.append(float(kl.detach()))
    assert torch.isfinite(total),'Nonfinite training loss';total.backward();losses[domain].append(float(loss.detach()))
   groupnorm={g['name']:sum(float(p.grad.detach().float().pow(2).sum()) for p in g['params'] if p.grad is not None)**.5 for g in opt.param_groups};norm=sum(x*x for x in groupnorm.values())**.5;assert np.isfinite(norm)
   state['norms'].append(norm);state['norms']=state['norms'][-cfg['clip']['window']:]
   if step==cfg['clip']['calibration_steps']+1 or (step>cfg['clip']['calibration_steps'] and step%cfg['clip']['refresh_every']==1):state['clip_threshold']=float(np.quantile(state['norms'],cfg['clip']['quantile'])*cfg['clip']['multiplier'])
   clipped=state['clip_threshold'] is not None and norm>state['clip_threshold']
   if clipped:torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],state['clip_threshold']);state['clipped_steps']+=1
   opt.step();state['step']=step;state['elapsed_seconds']=base_elapsed+time.time()-started
   log(dict(event='update',step=step,elapsed_seconds=state['elapsed_seconds'],domain_CTC={k:float(np.mean(a)) for k,a in losses.items()},mean_KL=float(np.mean(klvalues)),gradient_by_group=groupnorm,gradient_norm=norm,clip_threshold=state['clip_threshold'],clipped=clipped,clipping_rate=state['clipped_steps']/step,exposures=[dict(domain=d,id=r['id'],writer_id=r.get('writer_id','synthetic'),tier_id=r.get('tier_id',r['id'])) for d,r,t in batch],augmentation_counters=dict(sampler.counters)))
   if step%cfg['eta_every']==0:
    speed=state['elapsed_seconds']/step;print('TRAIN',folder.name,step,'/',max_updates,'seconds/update',round(speed,2),'ETA ceiling hours',round(speed*(max_updates-step)/3600,2),'clip',round(state['clipped_steps']/step,4),flush=True)
   if step%evaluate_every==0 or step==max_updates:
    cp=folder/'checkpoints'/f'attempt_{attempt:03}_step_{step:05}.pt';cp.parent.mkdir(exist_ok=True);assert not cp.exists()
    if val:
     metrics,raw,masked=evaluate(m,v,val,segment/'evaluations',f'step_{step:05}')
     chosen=metrics[params['selection_decoder']];primary=chosen['macro_writer']['cer'] if generated else chosen['overall']['cer'];key=[primary,chosen['overall']['cer_no_whitespace'],-chosen['overall']['exact_tiers']]
     improved=state['best_primary'] is None or primary<state['best_primary']-1e-12
     state['stale']=0 if improved else state['stale']+1
     if improved:state['best_primary']=primary
     if state['best_key'] is None or tuple(key)<tuple(state['best_key']):state['best_key']=key;state['best_checkpoint']=str(cp);state['best_step']=step
     event=dict(step=step,metrics=metrics,checkpoint=str(cp),stale=state['stale']);state['evaluation_history'].append(event)
     write(segment/'evaluations'/f'step_{step:05}_count_errors.csv',count_errors(raw,'raw')+count_errors(masked,'role_mask'));log(dict(event='evaluation',**event));print('EVAL',folder.name,step,'macro CER',chosen['macro_writer']['cer'],'overall CER',chosen['overall']['cer'],'stale',state['stale'],flush=True)
    else:state['best_checkpoint']=str(cp);state['best_step']=step
    save=dict(format_version='stage2g_v5b_resume_v1',model_state_dict=m.state_dict(),optimizer_state_dict=opt.state_dict(),char_to_idx=metadata['char_to_idx'],vocabulary_hash=metadata['vocabulary_hash'],run_config=runconfig,run_state=state,numpy_rng=rng.bit_generator.state,python_rng=rand.getstate(),torch_rng=torch.get_rng_state(),augmentation_counters=dict(sampler.counters),E1_sha256=cfg['frozen_e1_sha256'])
    tmp=cp.with_suffix('.tmp');torch.save(save,tmp);os.replace(str(tmp),str(cp));dump(latest_path,dict(checkpoint=str(cp),sha256=sha(cp),step=step,attempt=attempt));dump(folder/'progress.json',state)
    if val and step>=minimum and state['stale']>=patience:break
  completion=dict(steps=step,best_checkpoint=state['best_checkpoint'],best_step=state['best_step'],best_key=state['best_key'],elapsed_seconds=state['elapsed_seconds'],seconds_per_update=state['elapsed_seconds']/step,clipping_rate=state['clipped_steps']/step,stopped_early=step<max_updates,hit_ceiling=step==max_updates,heldout_writer=heldout,mode=mode,smoke=smoke,evaluation_history=state['evaluation_history'],checkpoint_sha256=sha(state['best_checkpoint']))
  dump(completion_path,completion);return completion
 except BaseException as ex:
  dump(segment/'interrupted.json',dict(error=repr(ex),last_committed_step=state['step'],last_resumable_checkpoint=json.loads(latest_path.read_text()) if latest_path.exists() else None));raise
