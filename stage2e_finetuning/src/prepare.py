"""Load original model; audit historical evidence; freeze data without training."""
from context import *
from collections import Counter,defaultdict
import platform,html,importlib.metadata as metadata
_raw_zip=None
_raw_parents={}
def synthetic_row(r,cohort):
 global _raw_zip
 from talim.talim_rules import canonical_row
 if _raw_zip is None:_raw_zip=zipfile.ZipFile(dc.old.SOURCE_CONFIG['normal']['clean'])
 path=SYN/r['image_path'];truth=r['training_label'];parent=r['parent_image_id'];split=r['split']
 if parent not in _raw_parents:
  prefix=f"datasets/talim_100000/{split}/"
  data=json.loads(_raw_zip.read(prefix+'json/'+parent+'.json'))
  parent_txt=_raw_zip.read(prefix+'labels/'+parent+'.txt').decode('utf-8').replace('\r\r\n','\n').replace('\r\n','\n').replace('\r','\n')
  expected='\n\n'.join('\n'.join((x['upper'],x['count_line'],x['lower'])) for x in data['logical_rows'])
  assert parent_txt.rstrip('\n')==expected.rstrip('\n')
  for x in data['logical_rows']:assert canonical_row(x['runs'])==(x['upper'],x['count_line'],x['lower'])
  _raw_parents[parent]=data
 key={'count':'count_line','upper':'upper','lower':'lower'}[r['tier_name']]
 raw=_raw_parents[parent]['logical_rows'][int(r['row_index'])][key]
 assert raw==r['raw_label'] and raw.lstrip(' ')==truth
 return row_from_image(path,truth,id=r['tier_id'],cohort=cohort,split=r['split'],parent_id=parent,case_id=parent,kind='clean',augmentation='clean',role=r['tier_name'],raw_truth=raw,leading_offset=r['leading_offset'],verification='original Stage1 approved manifest; independently crosschecked raw parent TXT/JSON/canonical labels')
def main():
 seed()
 if (E/'configs/input_lock.json').exists():raise RuntimeError('Protocol already frozen; preserve it')
 if not (E/'configs/protected_before.json').exists():dump('configs/protected_before.json',protect_snapshot())
 inv=json.loads((D/'reports/artifact_inventory.json').read_text())
 assert all(digest(f['path'])==f['sha256'] for f in inv['files'])
 dc.old.verify_lock(code=True)
 m,v,state=model_load();m.eval()
 with torch.inference_mode():logits=m(torch.ones(1,1,64,128))
 assert logits.shape==(32,1,66)
 inventory=dict(checkpoint=str(CHECKPOINT),sha256=digest(CHECKPOINT),checkpoint_keys=list(state),parameter_count=sum(p.numel() for p in m.parameters()),vocabulary_hash=state['vocabulary_hash'],architecture=str(m),framework=torch.__version__,python=sys.version,cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,checkpoint_training_config=state.get('config',state.get('args',{})),blank=0,forward_shape=list(logits.shape))
 # torch checkpoint metadata is primitives; retain relevant exact metadata.
 dump('reports/model_inventory.json',inventory)
 (E/'reports/model_inventory.md').write_text('# Original model inventory\n\n'+f"Original checkpoint: {CHECKPOINT}\n\nSHA256: {digest(CHECKPOINT)}\n\nPyTorch {torch.__version__}; CPU available, CUDA={torch.cuda.is_available()}. Successfully loaded strict state dict and verified [1,1,64,128] -> [32,1,66].\n\n"+'Six CNN blocks reduce height64 to1, width /4; two bidirectional LSTM layers, hidden256, dropout0.1; linear 512->66. Original AdamW training implementation and warmup/cosine schedule are in training/manifest_training.py; checkpoint metadata retained in model_inventory.json.\n\n65 fixed character indices1..65, blank0. Grayscale float32 /255, bilinear aspect-preserving height64, white right padding and true sequence widths. No label truncation or internal-space normalization. Original greedy decoder data/ctc_decode.py, edit accounting stage2/positive_edge_cases/evaluate.py. Exact match means the complete tier string matches.\n\n'+str(m),encoding='utf-8')
 image_manifest=read(D/'manifests/fine_tuning_images.csv');tier_manifest=read(D/'manifests/fine_tuning_tiers.csv')
 assert len(tier_manifest)==1794 and sum(r['bucket']=='training_ready' for r in image_manifest)==714
 assert sum(r['bucket']=='manual_review' for r in image_manifest)==14 and sum(r['bucket']=='excluded' for r in image_manifest)==184
 pos=[]
 pmeta={r['image_id']:r for r in image_manifest}
 for r in tier_manifest:
  assert digest(r['crop_path'])==r['crop_png_sha256'];assert pmeta[r['image_id']]['experiment_partition']=='tuning'
  pos.append(row_from_image(r['crop_path'],r['ground_truth'],id=r['image_id']+f"_r{r['row']}_{r['tier_role']}",cohort='positive_train',split='train',parent_id=r['source_parent'],case_id=r['image_id'],kind='clean' if r['image_id'].endswith('__clean') else 'scanned',augmentation=next(p['augmentation'] for p in dc.pages() if p['case_id']==r['image_id'] and p['dataset']=='positive'),role=r['tier_role'],raw_truth=r['raw_ground_truth'],leading_offset=r['leading_offset'],verification='Stage2D automated independent labels/geometry/intended components; not human reviewed'))
 write('manifests/positive_train_audited.csv',pos)
 # Preserve complete Stage2D page-level evaluation outcome ledger.
 from pipeline_d1 import process_image
 from intended_components import intended_retention
 config=json.loads((D/'configs/D1.json').read_text());sources=dc.Sources();evalrows=[];ledger=[]
 cases=[p for p in dc.pages() if p['partition']=='validation' or p['partition']=='clean_regression']
 previous={dc.key(r):r for r in read(D/'reports/evaluation_v1/pages.csv')}
 if (E/'manifests/page_outcomes.csv').exists():
  ledger=read(E/'manifests/page_outcomes.csv');assert len(ledger)==len(cases)
  savedpred=read(D/'reports/evaluation_v1/predictions.csv');by={}
  for r in savedpred:by.setdefault((r['dataset'],r['case_id']),[]).append(r)
  clean_archives={d:zipfile.ZipFile(c['clean']) for d,c in dc.old.SOURCE_CONFIG.items()}
  annotations={}
  for p in cases:
   prev=previous[dc.key(p)]
   if prev['correct_segmentation']!='True':continue
   cohort='positive_development' if p['dataset']=='positive' else 'scanned_exploratory' if p['dataset']=='general' else 'clean_historical_regression'
   k=(p['dataset'],p['parent_id'])
   if k not in annotations:annotations[k]=json.loads(clean_archives[p['dataset']].read(dc.old.SOURCE_CONFIG[p['dataset']]['prefix']+p['source_json']))
   for t in by[dc.key(p)]:
    path=E/'derived_evaluation'/p['dataset']/p['case_id']/f"r{t['row']}_{t['role']}.png"
    im=cv2.imread(str(path),0);assert im is not None
    assert hashlib.sha256(str(im.shape).encode()+im.tobytes()).hexdigest()==t['crop_sha256']
    raw=annotations[k]['logical_rows'][int(t['row'])][{'upper':'upper','count':'count_line','lower':'lower'}[t['role']]]
    assert raw.lstrip(' ')==t['truth']
    evalrows.append(row_from_image(path,t['truth'],id=p['dataset']+'_'+p['case_id']+f"_r{t['row']}_{t['role']}",cohort=cohort,split='development',parent_id=p['parent_id'],case_id=p['case_id'],kind=p['kind'],augmentation=p['augmentation'],role=t['role'],raw_truth=raw,leading_offset=t['leading_offset'],verification='exact Stage2D crop pixel hash and independent source annotation; recovered completed preprocessing audit'))
  print('Recovered validated evaluation crops',len(evalrows),flush=True)
 else:
  for i,p in enumerate(cases):
   image,ref,entries=sources.get(p);out=process_image(image,config);a=dc.reference.assess(out,ref,entries)
   prior=previous[dc.key(p)];assert a['correct']==(prior['correct_segmentation']=='True') and out['accepted']==(prior['proposal']=='True')
   cohort='positive_development' if p['dataset']=='positive' else 'scanned_exploratory' if p['dataset']=='general' else 'clean_historical_regression'
   ledger.append(dict(dataset=p['dataset'],parent_id=p['parent_id'],case_id=p['case_id'],kind=p['kind'],augmentation=p['augmentation'],proposal=out['accepted'],correct=a['correct'],reason=a['reason'],cohort=cohort))
   if a['correct']:
    assert intended_retention(out,ref,entries)['role_components_lost']==0
    folder=E/'derived_evaluation'/p['dataset']/p['case_id'];folder.mkdir(parents=True)
    for t in a['tiers']:
     x0,y0,x1,y1=t['box'];path=folder/f"r{t['row']}_{t['role']}.png";assert cv2.imwrite(str(path),out['image'][y0:y1,x0:x1])
     evalrows.append(row_from_image(path,t['truth'],id=p['dataset']+'_'+p['case_id']+f"_r{t['row']}_{t['role']}",cohort=cohort,split='development',parent_id=p['parent_id'],case_id=p['case_id'],kind=p['kind'],augmentation=p['augmentation'],role=t['role'],raw_truth=t['raw'],leading_offset=t['leading_offset'],verification='frozen D1; independent offline geometry, labels and intended components'))
   if (i+1)%40==0:print('Freeze evaluation pages',i+1,'/',len(cases),flush=True)
  write('manifests/page_outcomes.csv',ledger)
 # Additional original Stage1 validation parents, untouched by fine-tuning/replay.
 originalval=read(SYN/'manifests/stratified_validation_10k.csv')
 valparents=sorted({r['parent_image_id'] for r in originalval},key=lambda x:hashlib.sha256(('stage2e:42:cleanval:'+x).encode()).hexdigest())[:60]
 cleanval=[synthetic_row(r,'clean_validation') for r in originalval if r['parent_image_id'] in valparents]
 evalrows+=cleanval
 write('manifests/evaluation.csv',evalrows)
 write('manifests/positive_development.csv',[r for r in evalrows if r['cohort']=='positive_development'])
 evalhash={r['input_hash'] for r in evalrows}
 # Remove training pixel-identical inputs to any fixed eval crop, never remove eval hard cases.
 excluded=[dict(id=r['image_id'],parent_id=r['source_parent'],reason=r['reason'],scope='image',source_bucket=r['bucket']) for r in image_manifest if r['bucket']!='training_ready']
 train=[];seen=set()
 for r in pos:
  reason='normalized input duplicates fixed evaluation crop' if r['input_hash'] in evalhash else 'duplicate normalized training input and target' if (r['input_hash'],r['truth']) in seen else ''
  if reason:excluded.append(dict(id=r['id'],parent_id=r['parent_id'],reason=reason,scope='tier',source_bucket='training_ready'))
  else:train.append(r);seen.add((r['input_hash'],r['truth']))
 write('manifests/positive_train.csv',train);write('manifests/excluded_samples.csv',excluded)
 # Replay only from original train manifest, excluding historical-regression source parents.
 originaltrain=read(SYN/'manifests/stratified_train_150k.csv')
 forbidden={r['parent_id'] for r in evalrows}
 pool=[r for r in originaltrain if r['split']=='train' and r['parent_image_id'] not in forbidden]
 assert set(''.join(r['training_label'] for r in pool))==set(v.char_to_index)
 # Stratification followed by round-robin: role x label-length x width stratum.
 strata=defaultdict(list)
 for r in sorted(pool,key=lambda r:hashlib.sha256(('stage2e:42:replay:'+r['tier_id']).encode()).hexdigest()):strata[r['stratum_id']].append(r)
 selected=[];replayseen=set();characters=set();ordinal=0
 while len(selected)<1024 or characters!=set(v.char_to_index):
  progress=False
  for name,group in sorted(strata.items()):
   if not group:continue
   r=group.pop();progress=True
   t=synthetic_row(r,'synthetic_replay')
   if t['input_hash'] in evalhash or t['input_hash'] in replayseen:continue
   selected.append(t);replayseen.add(t['input_hash']);characters.update(t['truth'])
  if not progress:raise RuntimeError('Replay pool exhausted before vocabulary coverage')
 write('manifests/synthetic_replay_train.csv',selected)
 assert {r['parent_id'] for r in train}.isdisjoint({r['parent_id'] for r in evalrows if r['cohort']=='positive_development'})
 assert not {r['input_hash'] for r in train+selected}&evalhash
 # Deterministic small parent-disjoint selection cohort; keep the long parent diagnostic represented.
 development=[r for r in evalrows if r['cohort']=='positive_development']
 pids=sorted({r['parent_id'] for r in development},key=lambda p:hashlib.sha256(('stage2e:42:select:'+p).encode()).hexdigest())[:12]
 if 'edge_00323' not in pids:pids[-1]='edge_00323'
 selection=[r for r in development if r['parent_id'] in pids]+cleanval
 write('manifests/selection_development.csv',selection)
 # Human QC gallery is a review aid; no human-verification flags are changed.
 qc=[]
 for category,fn in [('small_symbols',lambda r:len(r['truth'])<=2),('rare_symbols',lambda r:any(x in r['truth'] for x in '{}|*')),('long',lambda r:len(r['truth'])>=80),('repeated',lambda r:any(a==b for a,b in zip(r['truth'],r['truth'][1:]))),('spacing',lambda r:'  ' in r['truth']),('perspective',lambda r:'perspective' in r['augmentation']),('degraded',lambda r:r['kind']=='scanned'),('upper',lambda r:r['role']=='upper'),('lower',lambda r:r['role']=='lower')]:
  for r in [r for r in pos+development if fn(r)][:4]:qc.append(dict(category=category,**r))
 write('manifests/quality_review.csv',qc)
 panels=[]
 for i,r in enumerate(qc):
  im=cv2.imread(r['path'],0);path=E/'galleries'/f"qc_{i:03}.png";cv2.imwrite(str(path),im)
  panels.append(f'<section><h2>{html.escape(r["category"])} — {html.escape(r["id"])}</h2><a href="{path.name}"><img src="{path.name}" style="max-width:95%;max-height:250px"></a><pre>{html.escape(r["truth"])}</pre><p>{r["width"]}x{r["height"]}; parent {r["parent_id"]}; {r["verification"]}</p></section>')
 (E/'galleries/data_quality_review.html').write_text('<!doctype html><meta charset="utf-8"><title>Stage2E crop/label QC</title><h1>Algorithmically verified; human review pending</h1>'+''.join(panels),encoding='utf-8')
 counts=dict(original_positive_images=714,original_positive_parents=179,original_positive_tiers=1794,manual_review_images=14,excluded_images=184,positive_train_after_duplicate_audit=len(train),positive_training_parents=len({r['parent_id'] for r in train}),replay_tiers=len(selected),replay_parents=len({r['parent_id'] for r in selected}),replay_vocabulary_coverage=len(characters),evaluation_tiers=len(evalrows),evaluation_by_cohort=dict(Counter(r['cohort'] for r in evalrows)),selection_tiers=len(selection),human_verified=0,independent_test=False,duplicate_policy='Training exact normalized pixel duplicates against any evaluation cohort and within training removed with explicit provenance; all evaluable hard evaluation tiers retained. Shared glyph shapes across different parents are expected, not unseen-symbol generalization.')
 dump('reports/data_audit.json',counts)
 (E/'reports/data_audit.md').write_text('# Stage2E data audit\n\n'+json.dumps(counts,indent=2)+'\n\nPositive development is the existing frozen 46-parent exposed internal-validation partition, never reassigned or called unseen. All original positive validation/test parents and exposed Stage2B test parents remain absent. General sources remain exploratory. Added clean validation uses 60 original Stage1 validation parents; replay comes only from its original training manifest. Source labels, crop/input hashes, normalization, vocabulary and repeated-character CTC feasibility were checked. Exact normalized input leakage is removed from training, never by pruning evaluation. Near-identical isolated glyphs remain by design; this is not unseen-font or unseen-character evaluation.\n',encoding='utf-8')
 locks={str(p):digest(p) for p in list((E/'manifests').glob('*.csv'))+[CHECKPOINT,VOCAB,SYN/'manifests/stratified_train_150k.csv',SYN/'manifests/stratified_validation_10k.csv',D/'configs/D1.json',D/'reports/artifact_inventory.json']}
 for r in train+selected+evalrows:locks[r['path']]=r['file_sha256']
 dump('configs/input_lock.json',locks)
 dump('reports/environment.json',dict(python=sys.version,platform=platform.platform(),packages={p:metadata.version(p) for p in ['torch','numpy','opencv-python','Pillow','scipy']},cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),threads=4))
 print('FROZEN DATA',json.dumps(counts),flush=True)
if __name__=='__main__':main()

