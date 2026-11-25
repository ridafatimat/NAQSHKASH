"""Stage 2G V2: allowlisted import, train-only calibration, paired CNN-only pilots."""
import sys,json,csv,hashlib,io,zipfile,collections,random,time,math,platform,argparse
from pathlib import Path
sys.dont_write_bytecode=True
P=Path('C:/NAQSHKASH');O=P/'stage2/stage2g_writer_independent_finetune_v2'
SPLIT='naqshkash_stage2g_writer_split_v2_final';AUG='naqshkash_stage2g_augmented_training_v1_1';OLD='naqshkash_stage2g_handwritten_release_v1'
sys.path[:0]=[str(P),str(P/'stage2/stage2e_finetuning/src')]
import context as c
from checkpoint_utils import forward_training
np=c.np;cv=c.cv2;torch=c.torch
E=c.E;TRAIN_WRITERS={'W01','W03','W04','W06'}
FROZEN_SHA='27f2da50795e2fca8f327afefdce10645f3e60e5320060a8cee93994fabdcdf1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bh(b):return hashlib.sha256(b).hexdigest()
def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def zread(z,n):return list(csv.DictReader(io.StringIO(z.read(n).decode('utf-8-sig'),newline='')))
def dump(n,x):
 p=O/n;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2),encoding='utf-8');tmp.replace(p)
def csvout(n,rr):
 if not rr:return
 p=O/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def ns(s):return ''.join(s.split())
def agg(rr):
 counts={k:sum(int(r[k]) for r in rr) for k in ['substitutions','deletions','insertions']};no={k:sum(int(r['no_space_'+k]) for r in rr) for k in counts};chars=sum(len(r['truth']) for r in rr);nc=sum(len(ns(r['truth'])) for r in rr)
 return dict(tiers=len(rr),characters=chars,nonspace_characters=nc,errors=sum(counts.values()),cer=sum(counts.values())/chars,cer_no_whitespace=sum(no.values())/nc,exact_tiers=sum(r['truth']==r['prediction'] for r in rr),exact_accuracy=sum(r['truth']==r['prediction'] for r in rr)/len(rr),**counts,**{'no_space_'+k:v for k,v in no.items()},mean_ctc_loss=float(np.mean([r['loss'] for r in rr])))
def summary(rr):
 return dict(status='PROVISIONAL',overall=agg(rr),per_role={role:agg([r for r in rr if r['role']==role]) for role in sorted({r['role'] for r in rr})},per_source={s:agg([r for r in rr if r['source_id']==s]) for s in sorted({r['source_id'] for r in rr})},macro_source_cer=float(np.mean([agg([r for r in rr if r['source_id']==s])['cer'] for s in {r['source_id'] for r in rr}])))
def load_model(path=None):
 cp=Path(path) if path else E/'checkpoints/selected_model/best.pt'
 if path is None:assert sha(cp)==FROZEN_SHA
 m,v,s=c.model_load(cp);m.eval();return m,v,s
def audit():
 assert not (O/'dataset_audit.json').exists(),'Preserve completed audit'
 O.mkdir(parents=True,exist_ok=True)
 paths={name:Path.home()/'Downloads'/(name+'.zip') for name in [SPLIT,AUG,OLD]}
 for name,p in paths.items():assert p.is_file(),f'Dataset not found: {p}'
 archives={name:zipfile.ZipFile(p) for name,p in paths.items()};hashes={name:sha(p) for name,p in paths.items()}
 assert hashes[AUG]=='b0550bdc3247f054637565edfbd11135d87572ad85ec03f6122682042407e420'
 assert hashes[OLD]=='7c3c901c4963f9aa4906d14f0868b1fadcdf131f6c49ec4803bfe3e3035f075b'
 z=archives[SPLIT];prefix=SPLIT+'/'
 for r in zread(z,prefix+'checksums.csv'):assert bh(z.read(prefix+r['path']))==r['sha256'],r['path']
 # Metadata only, including forbidden sample identities for disjointness. Never open forbidden pixels/labels.
 meta={f:zread(z,prefix+f) for f in ['train_manifest_v2.csv','validation_manifest_v2.csv','validation_primary_originals_v2.csv','validation_augmented_secondary_v2.csv','test_manifest_v2.csv','test_pending_annotation_manifest_v2.csv','quarantine_manifest_v2.csv','excluded_tiers_v2.csv','source_to_writer_mapping_final_v2.csv','writer_split_assignment_v2.csv']}
 for f in ['codex_handoff_frozen_e1_validation.md','writer_split_report_v2.md']:(O/f).write_bytes(z.read(prefix+f))
 tr=meta['train_manifest_v2.csv'];va=meta['validation_manifest_v2.csv'];primary=meta['validation_primary_originals_v2.csv'];secondary=meta['validation_augmented_secondary_v2.csv'];forbidden=meta['test_manifest_v2.csv']+meta['test_pending_annotation_manifest_v2.csv']+meta['quarantine_manifest_v2.csv']
 assert len(tr)==2577 and collections.Counter(r['sample_type'] for r in tr)=={'original':108,'augmented':2469}
 assert len(primary)==14 and len(secondary)==312 and len(va)==326
 assert {r['writer_id'] for r in tr}==TRAIN_WRITERS and {r['writer_id'] for r in va}=={'W05'}
 assert {r['sample_id'] for r in va}=={r['sample_id'] for r in primary+secondary}
 assert {r['source_id'] for r in tr}.isdisjoint({r['source_id'] for r in va})
 assert {r['sample_id'] for r in tr+va}.isdisjoint({r['sample_id'] for r in forbidden})
 assert {r['tier_id'] for r in tr+va}.isdisjoint({r['tier_id'] for r in forbidden+meta['excluded_tiers_v2.csv']})
 assert all(r['source_id']!='all_006767' and not (r['source_id']=='all_003191' and r['row_index']=='1') for r in tr+va)
 assignments={r['writer_id']:r['new_split'] for r in meta['writer_split_assignment_v2.csv']}
 assert all(assignments[r['writer_id']]=='train' and r['new_split']=='train' for r in tr)
 assert all(assignments[r['writer_id']]=='validation' and r['new_split']=='validation' for r in va)
 assert {r['writer_id'] for r in meta['test_manifest_v2.csv']}=={'W02','W07'}
 source_writer={r['source_id']:r['writer_id'] for r in meta['source_to_writer_mapping_final_v2.csv']}
 assert all(source_writer[r['source_id']]==r['writer_id'] for r in tr+va)
 expected={'W01':55,'W03':17,'W04':12,'W06':24};assert collections.Counter(r['writer_id'] for r in tr if r['sample_type']=='original')==expected
 # Extract only allowed payloads, retaining package-relative paths without restructuring datasets.
 checksums={name:{r['path']:r for r in zread(archives[name],name+'/checksums.csv')} for name in [AUG,OLD]}
 augchecks={r['crop_id']:r for r in zread(archives[AUG],AUG+'/audit/all_augmented_crop_checks.csv')}
 _,v,_=load_model();records=[];image_splits=collections.defaultdict(set);input_locks={}
 for split,rows in [('train',tr),('validation',va)]:
  for r in rows:
   name=r['package'][:-4];assert name in [AUG,OLD];assert r['ocr_text']==r['raw_label'].lstrip(' ');assert v.decode(v.encode(r['ocr_text']))==r['ocr_text']
   for key in ['crop_path','label_path']:
    rel=r[key];assert not Path(rel).is_absolute() and '..' not in Path(rel).parts and ':' not in rel and '\\' not in rel
    data=archives[name].read(name+'/'+rel);ch=checksums[name][rel];assert bh(data)==ch['sha256'] and len(data)==int(ch['bytes'])
    dest=O/'packages'/name/rel;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists(),dest;dest.write_bytes(data);input_locks[str(dest)]=bh(data)
    if key=='crop_path':assert bh(data)==r['crop_sha256'];im=cv.imdecode(np.frombuffer(data,np.uint8),0)
    else:assert data.decode('utf-8-sig')==r['raw_label']
   assert im is not None and im.min()<im.max();h,w=im.shape;rw=round(w*64/h);required=len(r['ocr_text'])+sum(a==b for a,b in zip(r['ocr_text'],r['ocr_text'][1:]));assert rw//4>=required
   if r['sample_type']=='augmented':
    ar=augchecks[r['sample_id']];assert ar['status']=='ACCEPTED' and ar['raw_label']==r['raw_label'] and ar['tier_id']==r['tier_id'] and int(ar['components_failed'])==0
   records.append(dict(id=r['sample_id'],tier_id=r['tier_id'],source_id=r['source_id'],writer_id=r['writer_id'],split=split,sample_type=r['sample_type'],variant_id=r['variant_id'],role=r['tier'],row_index=r['row_index'],truth=r['ocr_text'],raw_label=r['raw_label'],path=str(O/'packages'/name/r['crop_path']),label_path=str(O/'packages'/name/r['label_path']),sha256=r['crop_sha256'],near_duplicate=r['near_duplicate_of_original'],resized_width=rw,ctc_required=required))
   image_splits[r['crop_sha256']].add(split)
 assert not any(len(s)>1 for s in image_splits.values())
 originals={r['tier_id']:r for r in records if r['sample_type']=='original'}
 assert all(r['truth']==originals[r['tier_id']]['truth'] and r['writer_id']==originals[r['tier_id']]['writer_id'] for r in records)
 csvout('audited_handwritten_inputs.csv',records);dump('import_input_locks.json',input_locks)
 replay=c.read(E/'manifests/synthetic_replay_train.csv');dev=[r for r in c.read(E/'manifests/evaluation.csv') if r['cohort'] in ['clean_historical_regression','clean_validation','positive_development']]
 assert {r['parent_id'] for r in replay}.isdisjoint({r['parent_id'] for r in dev})
 for r in replay+dev:assert sha(r['path'])==r['file_sha256']
 assert all(r['split']=='train' for r in replay)
 csvout('synthetic_replay_inputs.csv',replay);csvout('regression_inputs.csv',dev)
 protected={}
 for folder in [P/'stage2/stage2g_final_handwritten_e1_20261109',P/'stage2/stage2g_handwriting_domain_gap_diagnostics']:
  for f in folder.rglob('*'):
   if f.is_file():protected[str(f)]=sha(f)
 protected[str(E/'checkpoints/selected_model/best.pt')]=sha(E/'checkpoints/selected_model/best.pt')
 dump('protected_previous_files.json',protected)
 results=dict(status='PROVISIONAL',passed=True,archive_paths={n:str(p) for n,p in paths.items()},archive_sha256=hashes,train_originals=108,train_augmented=2469,validation_originals=14,validation_augmented=312,train_writer_original_counts=expected,train_sources=len({r['source_id'] for r in tr}),validation_sources=sorted({r['source_id'] for r in va}),forbidden_image_or_label_members_opened=0,quarantined_tiers=len({r['tier_id'] for r in meta['quarantine_manifest_v2.csv']}),no_cross_split_image_hash_duplicates=True,prior_split_not_reused=True,writer_provenance='PROVISIONAL, assignment/replacement-capture evidence incomplete',talim_reader_verified=False,normalization_not_yet_selected=True)
 dump('dataset_audit.json',results);print('AUDIT PASS',results,flush=True)
def tensors(row):
 im=cv.imread(row['path'],0);assert im is not None
 return c.resize_preserve_aspect_ratio(c.tensor(im))
def inkinfo(a):
 vals=a[a<.98];black=float(np.percentile(vals,5)) if len(vals) else 0;threshold=(1+black)/2;mask=a<threshold
 d=cv.distanceTransform(mask.astype(np.uint8),cv.DIST_L2,5);ridge=mask&(d>=cv.dilate(d,np.ones((3,3),np.uint8))-.01)
 y,x=np.where(a<.98)
 return dict(blackpoint=black,stroke=float(np.median(2*d[ridge])) if ridge.any() else 0,ink_span_fraction=(int(y.max())-int(y.min())+1)/64 if len(y) else 0,foreground_median=float(np.median(vals)) if len(vals) else 1,width=a.shape[1])
def topology_safe(before,after,black_before,black_after):
 # Multi-threshold component/holes safeguard is operational, never semantic certification.
 for q in [.25,.5,.75]:
  b=before<black_before+(1-black_before)*q;a=after<black_after+(1-black_after)*q
  nb,lb,st,_=cv.connectedComponentsWithStats(b.astype(np.uint8),8);ids=np.where(st[:,4]>=2)[0];ids=ids[ids>0]
  na,la=cv.connectedComponents(a.astype(np.uint8),8)
  owners=collections.defaultdict(set)
  for k in ids:
   vals=np.unique(la[lb==k]);vals=vals[vals>0]
   if len(vals)!=1:return False
   owners[int(vals[0])].add(int(k))
  if any(len(v)>1 for v in owners.values()):return False
  holes=lambda m:sum(h[3]>=0 for h in (cv.findContours(m.astype(np.uint8),cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)[1][0] if cv.findContours(m.astype(np.uint8),cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)[1] is not None else []))
  if holes(a)!=holes(b):return False
 return True
def normalize(t,role,conf):
 if conf['name']=='identity':return t,dict(applied_strength=0.,guard_fallback=False,blackpoint=0.)
 a=t[0].numpy().copy();info=inkinfo(a);black=info['blackpoint'];base=a.copy()
 if conf.get('contrast'):
  a=np.clip((a-black)/max(1-black,.01),0,1);bafter=0.
 else:bafter=black
 contrastonly=a.copy();strength=conf.get('strength',0.)
 if conf.get('adaptive'):strength=min(strength,max(0.,(conf['stroke_targets'][role]-info['stroke'])/2))
 if strength>0:
  dil=cv.erode(a,cv.getStructuringElement(cv.MORPH_CROSS,(3,3)),iterations=1);accepted=0.
  for value in [strength,strength/2]:
   candidate=(1-value)*a+value*dil
   if topology_safe(base,candidate,black,bafter):a=candidate;accepted=value;break
  fallback=accepted==0;strength=accepted
 else:fallback=False
 assert np.all(a<=base+1e-6),'Normalization may not whiten any source ink'
 return torch.from_numpy(a.astype(np.float32)).unsqueeze(0),dict(applied_strength=strength,guard_fallback=fallback,blackpoint=black,stroke_before=info['stroke'],stroke_after=inkinfo(a)['stroke'],ink_span_before=info['ink_span_fraction'],ink_span_after=inkinfo(a)['ink_span_fraction'])
def evaluate(m,v,rows,conf,name=None,progress=False):
 m.eval();encoder=c.CTCLabelEncoder(v);criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False);pred=[]
 for i,r in enumerate(rows):
  image,diag=normalize(tensors(r),r['role'],conf);b=c.talim_collate_fn([dict(image=image,label_text=r['truth'],sample_id=r['id'],tier_name=r['role'])]);encoded=encoder.encode_batch(b['label_texts'])
  with torch.inference_mode():
   logits=m(b['images']);lengths,_=c.runtime_ctc_lengths(m,b,encoded,logits.shape[0]);loss,_=c.fp32_ctc_loss(logits,encoded['targets'],lengths,encoded['target_lengths'],criterion);assert torch.isfinite(loss);prediction=c.greedy_ctc_decode(logits.argmax(2)[:int(lengths[0]),0].tolist(),v)
  counts=c.dc.old.edit_counts(r['truth'],prediction);no=c.dc.old.edit_counts(ns(r['truth']),ns(prediction))
  pred.append(dict(id=r['id'],source_id=r.get('source_id',r.get('parent_id','')),writer_id=r.get('writer_id','synthetic'),role=r['role'],variant_id=r.get('variant_id',r.get('augmentation','clean')),truth=r['truth'],prediction=prediction,**counts,**{'no_space_'+k:x for k,x in no.items()},edit_distance=sum(counts.values()),cer=sum(counts.values())/len(r['truth']),cer_no_whitespace=sum(no.values())/len(ns(r['truth'])),exact=prediction==r['truth'],loss=float(loss),time_steps=int(lengths[0]),normalization_applied_strength=diag['applied_strength'],normalization_fallback=diag['guard_fallback']))
  if progress and (i+1)%200==0:print('EVAL',name,i+1,'/',len(rows),flush=True)
 result=summary(pred)
 if name:dump('metrics/'+name+'.json',result);csvout('predictions/'+name+'.csv',pred)
 return result,pred
def calibrate():
 assert not (O/'selected_normalization.json').exists(),'Normalization already frozen'
 rows=read(O/'audited_handwritten_inputs.csv');tr=[r for r in rows if r['split']=='train' and r['sample_type']=='original'];assert len(tr)==108
 dev=read(O/'regression_inputs.csv');controls=[]
 for cohort in ['clean_historical_regression','positive_development']:
  for role in ['upper','count','lower']:controls+=sorted([r for r in dev if r['cohort']==cohort and r['role']==role],key=lambda x:x['id'])[:20]
 # Synthetic development controls alone define role-specific stroke targets.
 targets={role:float(np.median([inkinfo(tensors(r)[0].numpy())['stroke'] for r in controls if r['role']==role])) for role in ['upper','count','lower']}
 configs=[dict(name='identity'),dict(name='contrast',contrast=True),dict(name='guarded_fixed_thickness',strength=.5),dict(name='guarded_adaptive_stroke',strength=.5,adaptive=True,stroke_targets=targets),dict(name='contrast_guarded_adaptive',contrast=True,strength=.5,adaptive=True,stroke_targets=targets)]
 m,v,_=load_model();m.requires_grad_(False);base=None;records=[];details=[]
 for conf in configs:
  train,pred=evaluate(m,v,tr,conf,'calibration_train_'+conf['name']);synthetic,sp=evaluate(m,v,controls,conf,'calibration_synthetic_'+conf['name'])
  if base is None:base=sp
  safe=all(sum(x[k] for x in sp if x['id']==r['id'] for k in ['substitutions','deletions','insertions'])<=sum(r[k] for k in ['substitutions','deletions','insertions']) for r in base)
  # Prevent normalizer being selected by an apparent gain that damages synthetic controls.
  writer_cer={w:agg([r for r in pred if r['writer_id']==w])['cer'] for w in sorted(TRAIN_WRITERS)}
  records.append(dict(name=conf['name'],train_cer=train['overall']['cer'],train_cer_no_whitespace=train['overall']['cer_no_whitespace'],writer_macro_cer=float(np.mean(list(writer_cer.values()))),synthetic_cer=synthetic['overall']['cer'],safe_on_each_synthetic_control=safe,guard_fallbacks=sum(r['normalization_fallback'] for r in pred),mean_dilation_strength=float(np.mean([r['normalization_applied_strength'] for r in pred]))))
  for r in tr+controls:
   before=tensors(r);after,diag=normalize(before,r['role'],conf);inf=inkinfo(before[0].numpy());aft=inkinfo(after[0].numpy());details.append(dict(name=conf['name'],id=r['id'],domain='handwritten_train' if r in tr else 'synthetic_development',writer_id=r.get('writer_id','synthetic'),role=r['role'],stroke_before=inf['stroke'],stroke_after=aft['stroke'],ink_span_before=inf['ink_span_fraction'],ink_span_after=aft['ink_span_fraction'],foreground_before=inf['foreground_median'],foreground_after=aft['foreground_median'],applied_strength=diag['applied_strength'],fallback=diag['guard_fallback']))
  print('CALIBRATION',records[-1],flush=True)
 good=[r for r in records if r['safe_on_each_synthetic_control']];chosen=min(good,key=lambda r:(r['writer_macro_cer'],r['train_cer_no_whitespace']));selected=next(conf for conf in configs if conf['name']==chosen['name'])
 dump('selected_normalization.json',dict(configuration=selected,selection=chosen,calibrated_on='108 training-writer originals plus 120 existing synthetic development controls only; no W05 pixels/inference used for calibration',selection_rule='No individual synthetic development-control edit increase; lowest training macro-writer CER, tie no-whitespace CER',all_candidates=configs,semantic_safety='Multi-threshold topology/holes guards and nonwhitening support checks are operational safeguards, not proof of Talim mark correctness. Representative train-only visual review required.',created_before_W05_evaluation=True))
 csvout('normalization_calibration_results.csv',records);csvout('normalization_appearance_geometry.csv',details)
 # Representative training-only review: one upper/count/lower per writer, plus strongest changes.
 from PIL import Image,ImageDraw
 chosenrows=[]
 for w in sorted(TRAIN_WRITERS):
  for role in ['upper','count','lower']:chosenrows.append(next(r for r in tr if r['writer_id']==w and r['role']==role))
 canvas=Image.new('RGB',(1800,len(chosenrows)*180),'#eeeeee');draw=ImageDraw.Draw(canvas)
 for i,r in enumerate(chosenrows):
  before=tensors(r);after,_=normalize(before,r['role'],selected)
  for j,(label,t) in enumerate([('original',before),('selected',after)]):
   im=Image.fromarray(np.round(t[0].numpy()*255).astype(np.uint8)).convert('RGB');im.thumbnail((1780,64));draw.text((10,i*180+j*85+2),r['writer_id']+' '+r['id']+' '+label,fill='black');canvas.paste(im,(10,i*180+j*85+20))
 canvas.save(O/'normalization_train_review.png')
 print('NORMALIZATION FROZEN',selected,flush=True)
def regression(m,v,rows,conf,name):
 _,pred=evaluate(m,v,rows,conf,'regression_'+name,progress=True)
 out={cohort:agg([r for r,x in zip(pred,rows) if x['cohort']==cohort]) for cohort in sorted({r['cohort'] for r in rows})}
 long=[r for r,x in zip(pred,rows) if x['cohort']=='positive_development' and len(x['truth'])>=65];assert len(long)==9
 out['long_positive_development']=agg(long);dump('metrics/regression_'+name+'_cohorts.json',out);return out
def gates(val,reg,baseline,basereg,config):
 checks={'minimum_three_fewer_handwritten_edits':val['overall']['errors']<=baseline['overall']['errors']-config['minimum_error_reduction'],
 'no_whitespace_CER_not_worse':val['overall']['cer_no_whitespace']<=baseline['overall']['cer_no_whitespace'],
 'exact_accuracy_not_lower':val['overall']['exact_accuracy']>=baseline['overall']['exact_accuracy'],
 'both_validation_sources_not_worse':all(val['per_source'][s]['cer']<=baseline['per_source'][s]['cer'] for s in baseline['per_source'])}
 for cohort,base in basereg.items():
  limit=config['clean_cer_delta'] if cohort.startswith('clean') else config['long_cer_delta'] if cohort.startswith('long') else config['positive_cer_delta']
  checks[cohort+'_cer']=reg[cohort]['cer']-base['cer']<=limit+1e-12
  if cohort.startswith('clean'):checks[cohort+'_exact']=reg[cohort]['exact_accuracy']>=base['exact_accuracy']-config['clean_exact_drop']-1e-12
 return dict(passed=all(checks.values()),checks=checks)
def schedule_inputs(train,replay,config):
 originals=[r for r in train if r['sample_type']=='original'];assert len(originals)==108
 safevariants={'brightness','contrast','exposure','gaussian_blur','jpeg_compression','local_shadow'}
 pools=collections.defaultdict(list);reason=collections.Counter()
 for r in train:
  if r['sample_type']!='augmented':continue
  if r['near_duplicate'].lower()=='true':reason['near_duplicate']+=1;continue
  if r['variant_id'] not in safevariants:reason['outside_bounded_appearance_variant_list']+=1;continue
  if r['sha256']==next(x['sha256'] for x in originals if x['tier_id']==r['tier_id']):reason['exact_original_duplicate']+=1;continue
  pools[r['tier_id']].append(r)
 rng=random.Random(config['seed']);replay_order=list(range(len(replay)));rng.shuffle(replay_order);used_aug=collections.defaultdict(set);schedule=[];ri=0
 for epoch in range(config['max_epochs']):
  by={w:[r for r in originals if r['writer_id']==w] for w in sorted(TRAIN_WRITERS)}
  for g in by.values():rng.shuffle(g)
  ordered=[]
  while any(by.values()):
   for w in sorted(by):
    if by[w]:ordered.append(by[w].pop())
  for orig in ordered:
   candidates=[r for r in pools[orig['tier_id']] if r['sha256'] not in used_aug[orig['tier_id']]]
   chosen=rng.choice(candidates) if candidates else None
   if chosen:used_aug[orig['tier_id']].add(chosen['sha256'])
   hw=[orig]+([chosen] if chosen else []);wfactor=108/(4*sum(r['writer_id']==orig['writer_id'] for r in originals))
   batch=[dict(domain='handwritten_'+r['sample_type'],row=r,loss_weight=.5*wfactor/len(hw)) for r in hw]
   for j in range(2):batch.append(dict(domain='synthetic_replay',row=replay[replay_order[ri%len(replay_order)]],loss_weight=.25));ri+=1
   schedule.append(dict(epoch=epoch+1,batch=batch))
 assert len(schedule)==216
 dump('sampling_schedule.json',schedule);dump('augmentation_sampling_policy.json',dict(available_augmented=2469,eligible_variants=sorted(safevariants),eligible_samples=sum(len(g) for g in pools.values()),eligible_original_parents=len(pools),not_sampled_reasons=dict(reason),policy='One original per update, at most one augmentation from its exact tier parent, two replay tiers. Unique augmentation hash per parent across both epochs. Missing augmentation is not replaced by another parent; original receives the full handwritten half-loss. Writer inverse-frequency weights make original-parent effective contribution equal per writer across a full pass. Both branches use the identical saved schedule.'))
 return schedule
def student_mode(m):
 m.eval();m.cnn.train()
 for module in m.cnn.modules():
  if isinstance(module,torch.nn.BatchNorm2d):module.eval()
def subhash(m,prefix):
 h=hashlib.sha256()
 for k,x in m.state_dict().items():
  if k.startswith(prefix):h.update(k.encode());h.update(x.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def train_branch(branch,conf,schedule,primary,secondary,dev,baseval,basereg,config,teacher,v,initial):
 c.seed(config['seed']);m,_,_=load_model();m.requires_grad_(False);m.cnn.requires_grad_(True)
 optimizer=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=config['learning_rate'],weight_decay=config['weight_decay']);encoder=c.CTCLabelEncoder(v);criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False)
 rnn_hash=subhash(m,'rnn.');classifier_hash=subhash(m,'classifier.');teacher_cache={};history=[];usage=[];candidates=[];bestcer=baseval['overall']['cer'];stale=0;epoch_ctc=[];start=time.time()
 for step,entry in enumerate(schedule,1):
  student_mode(m);optimizer.zero_grad(set_to_none=True);losses=[];klvalues=[];t0=time.time()
  for item in entry['batch']:
   r=item['row'];domain=item['domain'];synthetic=domain=='synthetic_replay'
   if not synthetic:assert r['writer_id'] in TRAIN_WRITERS and r['split']=='train'
   else:assert r['split']=='train'
   raw=tensors(r);image,diag=normalize(raw,r['role'],conf);batch=c.talim_collate_fn([dict(image=image,label_text=r['truth'],sample_id=r['id'],tier_name=r['role'])]);encoded=encoder.encode_batch(batch['label_texts']);logits=forward_training(m,batch['images'],2048);lengths,_=c.runtime_ctc_lengths(m,batch,encoded,logits.shape[0]);ctc,_=c.fp32_ctc_loss(logits,encoded['targets'],lengths,encoded['target_lengths'],criterion);assert torch.isfinite(ctc)
   total=ctc
   kl=torch.tensor(0.)
   if synthetic:
    if r['id'] not in teacher_cache:
     with torch.no_grad():teacher_cache[r['id']]=teacher(raw.unsqueeze(0)).float().softmax(2).detach()
    tp=teacher_cache[r['id']];assert tp.shape==logits.shape
    kl=torch.nn.functional.kl_div(logits.float().log_softmax(2),tp,reduction='none').sum(2).mean();total=ctc+config['replay_distillation_weight']*kl
   (item['loss_weight']*total).backward();losses.append((domain,float(ctc.detach())))
   if synthetic:klvalues.append(float(kl.detach()))
   usage.append(dict(branch=branch,step=step,epoch=entry['epoch'],domain=domain,id=r['id'],tier_id=r.get('tier_id',r['id']),source_id=r.get('source_id',r.get('parent_id','')),writer_id=r.get('writer_id','synthetic'),variant_id=r.get('variant_id',r.get('augmentation','clean')),loss_weight=item['loss_weight'],normalization=conf['name'],normalization_strength=diag['applied_strength']))
  grad=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],config['gradient_clip']);assert torch.isfinite(grad);optimizer.step()
  history.append(dict(step=step,epoch=entry['epoch'],mean_ctc_loss=float(np.mean([x[1] for x in losses])),handwritten_ctc_loss=float(np.mean([x[1] for x in losses if x[0]!='synthetic_replay'])),replay_ctc_loss=float(np.mean([x[1] for x in losses if x[0]=='synthetic_replay'])),mean_replay_KL=float(np.mean(klvalues)),gradient_norm=float(grad),seconds=time.time()-t0))
  if step%8==0:print('TRAIN',branch,step,'/',len(schedule),'ctc',history[-1]['mean_ctc_loss'],flush=True)
  if step%4==0:dump('training/'+branch+'_history.json',history);csvout('training/'+branch+'_exposures.csv',usage)
  if step%108==0:
   assert subhash(m,'rnn.')==rnn_hash and subhash(m,'classifier.')==classifier_hash
   epoch=entry['epoch'];name=branch+'_epoch_'+str(epoch);val,_=evaluate(m,v,primary,conf,name);reg=regression(m,v,dev,conf,name);gate=gates(val,reg,baseval,basereg,config['gates'])
   checkpoint=O/'checkpoints'/(name+'.pt');checkpoint.parent.mkdir(exist_ok=True);assert not checkpoint.exists()
   state=dict(initial);state.update(model_state_dict=m.state_dict(),optimizer_state_dict=optimizer.state_dict(),stage2g_v2_config=config,stage2g_v2_branch=branch,stage2g_v2_normalization=conf,stage2g_v2_step=step,stage2g_v2_epoch=epoch,stage2g_v2_initial_checkpoint_sha256=FROZEN_SHA,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),sampler_schedule_sha256=sha(O/'sampling_schedule.json'))
   torch.save(state,checkpoint)
   result=dict(branch=branch,epoch=epoch,checkpoint=str(checkpoint),sha256=sha(checkpoint),validation=val,regression=reg,gate_vs_matching_frozen_pipeline=gate,rnn_and_classifier_unchanged=True)
   candidates.append(result);dump('training/'+branch+'_candidates.json',candidates);dump('training/'+branch+'_history.json',history);csvout('training/'+branch+'_exposures.csv',usage)
   print('EPOCH',branch,epoch,'CER',val['overall']['cer'],'gate',gate,flush=True)
   if val['overall']['cer']<bestcer:bestcer=val['overall']['cer'];stale=0
   else:stale+=1
   if stale>=config['early_stopping_patience']:break
 dump('training/'+branch+'_completion.json',dict(steps=len(history),epochs=candidates[-1]['epoch'],wall_seconds=time.time()-start,original_exposures=sum(r['domain']=='handwritten_original' for r in usage),augmented_exposures=sum(r['domain']=='handwritten_augmented' for r in usage),replay_exposures=sum(r['domain']=='synthetic_replay' for r in usage),unique_original_tiers=len({r['id'] for r in usage if r['domain']=='handwritten_original'}),unique_augmented_crops=len({r['id'] for r in usage if r['domain']=='handwritten_augmented'}),normalization=conf,rnn_and_classifier_unchanged=True))
 return candidates
def run(reuse_baselines=False):
 assert not (O/'final_report.md').exists(),'Preserve completed experiment'
 assert (O/'normalization_visual_review.json').exists(),'Finish training-only visual review before W05 inference'
 assert json.loads((O/'normalization_visual_review.json').read_text())['approved_for_bounded_experiment']
 rows=read(O/'audited_handwritten_inputs.csv');train=[r for r in rows if r['split']=='train'];primary=[r for r in rows if r['split']=='validation' and r['sample_type']=='original'];secondary=[r for r in rows if r['split']=='validation' and r['sample_type']=='augmented'];dev=read(O/'regression_inputs.csv');replay=read(O/'synthetic_replay_inputs.csv')
 norm=json.loads((O/'selected_normalization.json').read_text())['configuration'];identity={'name':'identity'}
 config=dict(status='PROVISIONAL',seed=42,learning_rate=1e-5,weight_decay=1e-4,gradient_clip=1.,max_epochs=2,early_stopping_patience=2,optimizer='AdamW',trainable='CNN only; BiLSTM and classifier frozen throughout; BN running statistics frozen',replay_distillation_weight=.1,distillation='Mean over true time steps KL(frozen E1 raw replay || student replay with branch actual preprocessing), T=1',handwritten_representation={'C_raw':identity,'D_normalized':norm},normalization_scope='Each approach applies its actual preprocessing to BOTH handwritten and synthetic inference/replay. No handwriting-oracle routing.',microbatch=1,effective_batch='one original, at most one same-parent augmentation, two synthetic replay; equal domain loss and inverse-original-count writer weights',gates=dict(minimum_error_reduction=3,clean_cer_delta=.001,clean_exact_drop=.01,positive_cer_delta=.005,long_cer_delta=.005),selection='All gates versus matching frozen pipeline and absolute synthetic gates versus A; minimum 3 fewer edits, no-whitespace/exact/source nonworsening. Rank eligible by CER, no-whitespace CER, negative exact. Frozen E1 fallback. Secondary augmented validation never selects.',uncertainty='One validation writer/two sources. Sourcewise/leave-one-source-out sensitivity only; no IID-tier CI or writer-population CI.',frozen_e1_sha256=FROZEN_SHA)
 if (O/'training_config.json').exists():
  assert reuse_baselines,'Run already started; use explicit continuation'
  prior=json.loads((O/'training_config.json').read_text());assert prior==config,'No hyperparameters/gates may change during continuation'
 else:dump('training_config.json',config)
 schedule=schedule_inputs(train,replay,config)
 # All configuration and schedules frozen BEFORE first W05 inference.
 c.seed(42);teacher,v,initial=load_model();teacher.requires_grad_(False)
 if reuse_baselines:
  base=json.loads((O/'baseline_results.json').read_text());A=base['A'];B=base['B'];regA=base['regression_A'];regB=base['regression_B']
  print('REUSED COMPLETED A/B ORIGINAL VALIDATION AND REGRESSION BASELINES',flush=True)
 else:
  A,_=evaluate(teacher,v,primary,identity,'A_frozen_original');B,_=evaluate(teacher,v,primary,norm,'B_frozen_normalized')
  regA=regression(teacher,v,dev,identity,'A_frozen_original');regB=regression(teacher,v,dev,norm,'B_frozen_normalized')
  dump('baseline_results.json',dict(A=A,B=B,regression_A=regA,regression_B=regB))
 candidates=[]
 for branch,conf,base,regbase in [('C_raw',identity,A,regA),('D_normalized',norm,B,regB)]:
  candidates+=train_branch(branch,conf,schedule,primary,secondary,dev,base,regbase,config,teacher,v,initial)
 for x in candidates:
  x['gate_vs_A']=gates(x['validation'],x['regression'],A,regA,config['gates']);x['eligible']=x['gate_vs_A']['passed'] and x['gate_vs_matching_frozen_pipeline']['passed']
 dump('all_candidates.json',candidates)
 chosen={}
 for branch in ['C_raw','D_normalized']:
  xx=[r for r in candidates if r['branch']==branch];eligible=[r for r in xx if r['eligible']];best=min(eligible or xx,key=lambda r:(r['validation']['overall']['cer'],r['validation']['overall']['cer_no_whitespace'],-r['validation']['overall']['exact_accuracy']));chosen[branch]=best
 b_gate=gates(B,regB,A,regA,config['gates']);approaches={'A':dict(validation=A,regression=regA,eligible=True,checkpoint=str(E/'checkpoints/selected_model/best.pt'),sha256=FROZEN_SHA),'B':dict(validation=B,regression=regB,eligible=b_gate['passed'],checkpoint=str(E/'checkpoints/selected_model/best.pt'),sha256=FROZEN_SHA,gate=b_gate)}
 for code,branch in [('C','C_raw'),('D','D_normalized')]:approaches[code]=dict(chosen[branch],validation=chosen[branch]['validation'],regression=chosen[branch]['regression'])
 rawbest=min(approaches,key=lambda k:(approaches[k]['validation']['overall']['cer'],approaches[k]['validation']['overall']['cer_no_whitespace'],k));safebest=min([k for k in approaches if approaches[k]['eligible']],key=lambda k:(approaches[k]['validation']['overall']['cer'],approaches[k]['validation']['overall']['cer_no_whitespace'],k))
 result=dict(status='PROVISIONAL',approaches=approaches,best_observed_approach=rawbest,selected_under_gates=safebest,frozen_E1_retained=safebest in ['A','B'],best_passing_candidate=next((dict(path=approaches[k]['checkpoint'],sha256=approaches[k]['sha256']) for k in [safebest] if k in ['C','D']),None),normalization=norm,Stage2F_rejection_policy='NOT EVALUATED: existing pixel/evaluator APIs hard-lock frozen E1 and original preprocessing; no supported candidate/preprocessor injection. Reuse would require an adapter/lineage change, outside this conditional check. No rejection policy retrained or changed. OCR gates do not certify rejection behavior.')
 dump('final_comparison.json',result);final_report(result,config)
 protected=json.loads((O/'protected_previous_files.json').read_text());locks=json.loads((O/'import_input_locks.json').read_text());assert all(sha(p)==h for p,h in protected.items());assert all(sha(p)==h for p,h in locks.items())
 dump('final_integrity.json',dict(original_e1_unchanged=sha(E/'checkpoints/selected_model/best.pt')==FROZEN_SHA,all_previous_experiment_hashes_unchanged=True,all_allowed_source_images_labels_unchanged=True,test_quarantine_payloads_not_extracted=True,writer_split_unchanged=True,normalization_chosen_before_W05=True,raw_and_normalized_candidates_distinct=True,all_recurrent_classifier_weights_frozen=True))
 print('EXPERIMENT COMPLETE',result['best_observed_approach'],result['selected_under_gates'],result['best_passing_candidate'],flush=True)
def final_report(result,config):
 labels={'A':'Frozen E1 + original','B':'Frozen E1 + selected normalization','C':'Raw-handwriting fine-tune + original','D':'Normalized-handwriting fine-tune + normalization'}
 text='# Stage 2G V2 writer-independent appearance pilot — PROVISIONAL\n\nWriter assignments and Talim copying/transcription verification remain incomplete. Primary validation is 14 original tiers from W05, two sources. These results are not comparable to the previous 74.23%/73.20% source-split pilot.\n\n| Approach | Errors / characters | CER | CER excluding whitespace | Exact tiers |\n|---|---:|---:|---:|---:|\n'
 for code,x in result['approaches'].items():
  m=x['validation']['overall'];text+=f"| {code}: {labels[code]} | {m['errors']} / {m['characters']} | {m['cer']:.4%} | {m['cer_no_whitespace']:.4%} | {m['exact_tiers']}/14 |\n"
 text+=f"\nBest observed approach: **{result['best_observed_approach']}**. Selected under predetermined gates: **{result['selected_under_gates']}**. Frozen E1 retained: {result['frozen_E1_retained']}.\n\n## Rolewise primary results\n\n| Approach | Role | S / D / I | CER | No-whitespace CER | Exact tiers |\n|---|---|---:|---:|---:|---:|\n"
 for code,x in result['approaches'].items():
  for role,m in x['validation']['per_role'].items():text+=f"| {code} | {role} | {m['substitutions']} / {m['deletions']} / {m['insertions']} | {m['cer']:.4%} | {m['cer_no_whitespace']:.4%} | {m['exact_tiers']}/{m['tiers']} |\n"
 text+='\n## Synthetic regression with each actual inference preprocessing\n\n| Approach | Cohort | Errors / chars | CER | No-whitespace CER | Exact tiers |\n|---|---|---:|---:|---:|---:|\n'
 for code,x in result['approaches'].items():
  for cohort,m in x['regression'].items():text+=f"| {code} | {cohort} | {m['errors']} / {m['characters']} | {m['cer']:.4%} | {m['cer_no_whitespace']:.4%} | {m['exact_tiers']}/{m['tiers']} |\n"
 text+='\nLong positive is the existing Stage2E subset with target length >=65, nine tiers; it is a subset of positive development, never pooled as extra independent samples. Clean cohorts contain 645 historical and 63 validation tiers; positive development contains 431. These are exposed development/regression cohorts, not independent tests.\n\n## Predetermined gates and candidates\n\n'
 text+=json.dumps(config['gates'],indent=2)+'\n\nRequire at least three fewer primary edits, no worse no-whitespace CER or exact-tier accuracy, and no sourcewise CER worsening. Candidate C/D must satisfy gates against its matching frozen pipeline AND absolute gates against A. Exact/nonspace/source gates are included in all_candidates.json. Validation comparisons and candidate selection use only the 14 W05 originals. The 312 augmented validation crops are audited but are not used for training or selection.\n\n'
 for code in ['C','D']:
  x=result['approaches'][code];text+=f"- {code}: checkpoint {x['checkpoint']}; SHA-256 {x['sha256']}; eligible={x['eligible']}; epoch={x['epoch']}.\n"
 text+='\n## Uncertainty and source sensitivity\n\nOnly one validation writer is observed. A meaningful writer-population confidence interval cannot be estimated. Fourteen correlated tiers are not 14 independent writers. The following sourcewise changes also give leave-one-source-out sensitivity (retaining just that source), not confidence limits.\n\n| Approach | Source | CER | Difference from A (percentage points) |\n|---|---|---:|---:|\n'
 a=result['approaches']['A']['validation']['per_source']
 for code,x in result['approaches'].items():
  for source,m in x['validation']['per_source'].items():text+=f"| {code} | {source} | {m['cer']:.4%} | {(m['cer']-a[source]['cer'])*100:+.3f} |\n"
 text+='\n## Normalization and safety\n\nSelected configuration (fixed before W05 evaluation):\n\n'+json.dumps(result['normalization'],indent=2)+'\n\nAll parameters were calibrated solely on 108 training-writer originals and 120 existing synthetic development controls. Exact recipes, stroke targets and train-only scores are saved in selected_normalization.json and normalization_calibration_results.csv. No geometry warp is included; earlier domain-gap evidence already found no helpful height/crop normalization. New training/control appearance and vertical-extent proxies are recorded without repeating established CTC/glyph-height/whitespace diagnosis.\n\nDilation operates on height64 float inputs, uses at most half of a one-pixel cross erosion, and falls back to a smaller strength or no thickness change if original components merge/disappear or hole topology changes at three contrast-relative thresholds. Contrast may darken, never whiten, retained ink. These guards and representative train-only inspection are not proof that every Talim mark is semantically unchanged. No independent Talim reader has verified the dataset. Exact guard parameters, fallbacks and training inputs are recorded.\n\n## Controlled training\n\nTwo distinct branches start anew from the SHA-verified frozen E1. Only CNN parameters update, low LR 1e-5, AdamW, weight decay 1e-4, gradient clip1, seed42, maximum two epochs, patience2; BiLSTM/classifier and BN running statistics remain frozen. D sees the same fixed normalized representation during training, validation, replay and regression inference. C sees raw representation. Both use identical original-parent/augmentation/replay schedules and optimizer budgets. Each original is visited once per epoch; at most one safe-list, non-near-duplicate same-parent augmentation is sampled, with inverse-writer-frequency handwritten loss weights and a 50% replay-loss allocation. Replacement tiers without eligible augmentation are not replaced with another parent. Frozen-E1 raw replay posteriors anchor the student with 0.1 mean-time-step KL. No new augmentations or labels are generated; AdaBN and handwriting-style synthetic generation are optional future experiments, not performed.\n\nTraining logs include actual exposures, per-domain CTC and distillation losses, and per-epoch validation loss. Checkpoints retain separate raw/normalized lineage, vocabulary, optimizer, RNG and schedule identity. Failed gates and early stops are reported rather than silently replaced by more tuning.\n\n## Stage 2F policy regression\n\n'+result['Stage2F_rejection_policy']+' See stage2f_interface_assessment.md for the inspected interface evidence.\n\n## Observations versus hypotheses\n\nObserved outcomes are the measured comparison tables, per-source results and gate decisions. The CNN-only adaptation/guarded normalization/distillation combination is an experimental hypothesis; success here would not establish unseen-writer generalization. Remaining confounders include provisional writer identity, unverified copying/targets, one validation writer and development-control exposure. No protected-test or quarantined payload is extracted, inferred or used for selection. Frozen E1 and all prior experiment files are hash-checked after execution. No Git operations or dataset restructuring.\n\n## Reproduction\n\nThe single reproducible script is experiment.py. On this machine use the recorded Python/PyTorch/OpenCV environment. For a fresh isolated reproduction, change only O to a new directory, then run `python -B experiment.py --stage audit`, `--stage calibrate`, inspect normalization_train_review.png and record a truthful normalization_visual_review.json, then `--stage run`. A completed phase refuses overwrite. Packages are selectively extracted in their existing internal paths; only allowlisted train/validation members are loaded. Download archives and original Stage2E replay/regression paths must remain present. Input hashes, authoritative metadata, exact normalizer, sampling_schedule.json and training_config.json reproduce the protocol.\n'
 if (O/'execution_scope_note.json').exists():text+='\n## Execution recovery note\n\n'+json.dumps(json.loads((O/'execution_scope_note.json').read_text()),indent=2)+'\n'
 (O/'final_report.md').write_text(text,encoding='utf-8')
 if result['best_passing_candidate']:dump('selected_candidate.json',result['best_passing_candidate'])
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['audit','calibrate','run','continue'],required=True);args=parser.parse_args();c.seed(42)
 {'audit':audit,'calibrate':calibrate,'run':run,'continue':lambda:run(reuse_baselines=True)}[args.stage]()
