"""Isolated, provisional Stage 2G pilot; never evaluates protected test data."""
import sys, csv, json, hashlib, re, math, time, zipfile
from pathlib import Path
sys.dont_write_bytecode=True
P=Path('C:/NAQSHKASH'); O=P/'stage2/stage2g_final_handwritten_e1_20261109'
R=O/'dataset/naqshkash_stage2g_handwritten_release_v1'
sys.path[:0]=[str(P),str(P/'stage2/stage2e_finetuning/src')]
import context as c
import evaluation as ev
import metrics as met
from datasets import FrozenTiers
from checkpoint_utils import set_finetuning_mode,forward_training
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def dump(n,x): (O/n).write_text(json.dumps(x,indent=2),encoding='utf-8')
def write(n,rows):
 with (O/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def aggregate(rows):
 m=met.aggregate(rows);counts=dict(substitutions=0,deletions=0,insertions=0);chars=0
 for r in rows:
  a=re.sub(r'\s+','',r['truth']);b=re.sub(r'\s+','',r['prediction']);chars+=len(a)
  for k,v in c.dc.old.edit_counts(a,b).items():counts[k]+=v
 return dict(m,cer_excluding_whitespace=sum(counts.values())/chars,no_space_edit_counts=counts,mean_loss=sum(r['loss'] for r in rows)/len(rows))
def evaluate(m,v,rows,name,device):
 _,pred=ev.evaluate(m,v,rows,device=device,progress=True)
 write(name+'_predictions.csv',pred)
 result=dict(status='PROVISIONAL',overall=aggregate(pred),per_source={s:aggregate([r for r in pred if r['parent_id']==s]) for s in sorted({r['parent_id'] for r in pred})})
 dump(name+'_metrics.json',result);return result
def main():
 assert not (O/'baseline_metrics.json').exists(),'Preserve completed run; use a new isolated directory'
 c.seed(42);audit={'status':'PROVISIONAL','independent_talim_reader_verified':False,'checks':[]}
 def check(ok,msg):
  audit['checks'].append(dict(check=msg,passed=bool(ok)))
  if not ok or len(audit['checks'])%100==0:dump('dataset_import_audit.json',audit)
  assert ok,msg
 z=Path.home()/'Downloads'/'naqshkash_stage2g_handwritten_release_v1.zip'
 assert z.is_file(),f'Dataset not found: {z}'
 audit['dataset_zip']=str(z);audit['dataset_zip_sha256']=sha(z)
 sums=read(R/'checksums.csv')
 check(all(sha(R/r['path'])==r['sha256'] and (R/r['path']).stat().st_size==int(r['bytes']) for r in sums),'377 release checksums and byte sizes')
 check({str(p.relative_to(R)).replace('\\','/') for p in R.rglob('*') if p.is_file()}=={r['path'] for r in sums}|{'checksums.csv'},'No unlisted payloads')
 split=read(R/'evidence/split_assignment_authoritative.csv');orig=read(P/'stage2/stage2g_handwritten_dataset_preparation_20261108/source_split.csv')
 sm={r['source_id']:r['split'] for r in split};check(sm=={r['source_id']:('protected_test' if r['split']=='test' else r['split']) for r in orig},'Authoritative split equals existing source split (test normalized to protected_test)')
 release=json.loads((c.E/'reports/selected_checkpoint.json').read_text());cp=Path(release['selected_checkpoint'])
 check(release['selected']=='E1' and sha(cp)==release['sha256'],'Selected frozen E1 SHA-256')
 audit['frozen_checkpoint']=str(cp);audit['frozen_sha256']=sha(cp)
 m,v,state=c.model_load(cp);device='cuda' if c.torch.cuda.is_available() else 'cpu';m.to(device)
 audit['device']=device;audit['torch']=c.torch.__version__;audit['pairs']=[];groups={}
 for s,n in [('train',165),('validation',14)]:
  rr=read(R/(s+'_manifest.csv'));check(len(rr)==n and len({r['tier_id'] for r in rr})==n,s+' unique expected tiers');prepared=[]
  for r in rr:
   check(r['split']==s and sm[r['source_id']]==s and r['renderer']=='keep_faint_v1','Split and renderer '+r['tier_id'])
   check(r['crop_path']==f"{s}/tier_crops/{r['tier_id']}.png" and r['label_path']==f"{s}/tier_labels/{r['tier_id']}.txt",'Exact image-label path '+r['tier_id'])
   im=c.cv2.imread(str(R/r['crop_path']),0);raw=(R/r['label_path']).read_text(encoding='utf-8-sig')
   check(sha(R/r['crop_path'])==r['crop_sha256'] and sha(R/r['label_path'])==r['label_sha256'] and raw==r['raw_label'] and raw.lstrip(' ')==r['ocr_text'],'Hashes and whitespace '+r['tier_id'])
   ann=json.loads((P/'stage2/stage2g_handwritten_dataset_preparation_20261108/annotations'/(r['source_id']+'.json')).read_text())
   check(raw==ann['logical_rows'][int(r['row_index'])]['count_line' if r['tier']=='count' else r['tier']],'Existing source label '+r['tier_id'])
   check(im is not None and im.shape==(int(r['crop_height']),int(r['crop_width'])) and im.min()<im.max(),'Image decode and dimensions '+r['tier_id'])
   prepared.append(c.row_from_image(R/r['crop_path'],r['ocr_text'],id=r['tier_id'],cohort='handwritten_'+s,parent_id=r['source_id'],case_id=r['source_id'],kind='handwritten',augmentation='none',role=r['tier']))
   audit['pairs'].append(dict(tier_id=r['tier_id'],split=s,light_ink_pixels=int(((im>200)&(im<255)).sum()),min_ink_margin_px=r['min_ink_margin_px']))
  groups[s]=prepared
 check(not ({r['parent_id'] for r in groups['train']}&{r['parent_id'] for r in groups['validation']}),'No handwritten source leakage')
 check(not any(s in str(p) for s,t in sm.items() if t=='protected_test' for p in R.rglob('*') if p.is_file()),'No protected source filenames')
 excluded=read(R/'excluded_tiers.csv');check(len(excluded)==4 and not ({r['tier_id'] for r in excluded}&{r['id'] for r in groups['train']}),'Four excluded tiers absent')
 replay=read(c.E/'manifests/synthetic_replay_train.csv');dev=[r for r in read(c.E/'manifests/evaluation.csv') if r['cohort'] in ['clean_historical_regression','clean_validation','positive_development']]
 check(not ({r['parent_id'] for r in replay}&{r['parent_id'] for r in dev}),'Replay and regression parent isolation')
 for r in replay+dev:
  check(r['split'] in (['train'] if r in replay else ['train','development','validation']) and sha(r['path'])==r['file_sha256'],'Synthetic/development input '+r['id'])
 write('handwritten_train_inputs.csv',groups['train']);write('handwritten_validation_inputs.csv',groups['validation']);write('replay_inputs.csv',replay);write('regression_inputs.csv',dev)
 audit['passed']=True;dump('dataset_import_audit.json',audit)
 print('AUDIT PASS',len(audit['checks']),'device',device,flush=True)
 # Representative contact sheet preserves aspect ratio; original pixels remain unchanged.
 from PIL import Image,ImageDraw
 reps=groups['validation']+[next(r for r in groups['train'] if r['parent_id']==s) for s in sorted({r['parent_id'] for r in groups['train']})]
 sheet=Image.new('RGB',(1200,len(reps)*100),'#eeeeee');draw=ImageDraw.Draw(sheet)
 for i,r in enumerate(reps):
  im=Image.open(r['path']).convert('RGB');im.thumbnail((1180,74));sheet.paste(im,(10,i*100+24));draw.text((10,i*100+4),r['id'],fill='black')
 sheet.save(O/'visual_review.png')
 if '--audit-only' in sys.argv:return
 baseline=evaluate(m,v,groups['validation'],'baseline',device);print('BASELINE',baseline['overall'],flush=True)
 base_reg={}
 for cohort in sorted({r['cohort'] for r in dev}):base_reg[cohort]=evaluate(m,v,[r for r in dev if r['cohort']==cohort],'baseline_'+cohort,device)['overall']
 dump('baseline_regression.json',base_reg)
 config=dict(seed=42,learning_rate=1e-5,weight_decay=1e-4,epochs=2,microbatch=1,effective_batch=4,handwritten_per_update=2,replay_per_update=2,augmentations='none',batchnorm='frozen running statistics, affine trainable',selection='lowest handwritten CER, tie no-space CER; report all regression; readiness requires clean CER delta <=0.001 and positive delta <=0.005',frozen_e1_sha256=sha(cp),status='PROVISIONAL')
 dump('training_config.json',config)
 opt=c.torch.optim.AdamW(m.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay']);enc=c.CTCLabelEncoder(v);criterion=c.torch.nn.CTCLoss(blank=0,zero_infinity=False)
 train=FrozenTiers(groups['train']);rep=FrozenTiers(replay);rng=c.random.Random(42);history=[];usage=[];candidates=[];step=0;ri=0;order=list(range(len(replay)));rng.shuffle(order)
 for epoch in range(2):
  idx=list(range(165));rng.shuffle(idx)
  for start in range(0,165,2):
   batchidx=[('handwritten',i) for i in idx[start:start+2]]+[('replay',order[(ri+j)%len(order)]) for j in range(2)];ri+=2
   set_finetuning_mode(m);opt.zero_grad(set_to_none=True);losses=[];t=time.time()
   for source,i in batchidx:
    ds=train if source=='handwritten' else rep;b=c.talim_collate_fn([ds[i]]);encoded=enc.encode_batch(b['label_texts']);logits=forward_training(m,b['images'].to(device),2048);lengths,_=c.runtime_ctc_lengths(m,b,encoded,logits.shape[0]);loss,_=c.fp32_ctc_loss(logits,encoded['targets'].to(device),lengths,encoded['target_lengths'],criterion)
    assert c.torch.isfinite(loss);(loss/len(batchidx)).backward();losses.append(float(loss.detach()));usage.append(dict(step=step+1,epoch=epoch+1,source=source,id=ds.rows[i]['id'],parent_id=ds.rows[i]['parent_id']))
   norm=c.torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);assert c.torch.isfinite(norm);opt.step();step+=1
   history.append(dict(step=step,epoch=epoch+1,train_loss=sum(losses)/len(losses),gradient_norm=float(norm),seconds=time.time()-t))
   dump('training_history.json',history);write('training_exposures.csv',usage)
   if step%8==0:print('TRAIN',step,'loss',history[-1]['train_loss'],flush=True)
  result=evaluate(m,v,groups['validation'],f'epoch_{epoch+1}',device);path=O/f'candidate_epoch_{epoch+1}.pt'
  candidate=dict(state);candidate.update(model_state_dict=m.state_dict(),optimizer_state_dict=opt.state_dict(),stage2g_config=config,stage2g_step=step,stage2g_parent_sha256=release['sha256'],torch_rng=c.torch.get_rng_state(),python_rng=c.random.getstate());c.torch.save(candidate,path)
  candidates.append(dict(epoch=epoch+1,path=str(path),sha256=sha(path),metrics=result));dump('candidates.json',candidates);print('EPOCH',epoch+1,result['overall'],flush=True)
 best=min(candidates,key=lambda x:(x['metrics']['overall']['cer'],x['metrics']['overall']['cer_excluding_whitespace']))
 m,v,_=c.model_load(Path(best['path']),device=device);reg={}
 for cohort in sorted(base_reg):reg[cohort]=evaluate(m,v,[r for r in dev if r['cohort']==cohort],'selected_'+cohort,device)['overall']
 dump('selected_regression.json',reg);check(sha(cp)==release['sha256'],'Frozen E1 unchanged after experiment')
 ready=best['metrics']['overall']['cer']<baseline['overall']['cer'] and all(reg[k]['cer']-base_reg[k]['cer']<=(.005 if k=='positive_development' else .001) for k in reg)
 final=dict(status='PROVISIONAL',baseline=baseline['overall'],selected=best,regression_before=base_reg,regression_after=reg,ready_for_further_development_testing=ready,handwritten_exposures=sum(r['source']=='handwritten' for r in usage),replay_exposures=sum(r['source']=='replay' for r in usage),protected_test_evaluations=0)
 dump('final_comparison.json',final)
 report='# Stage 2G final handwritten pilot — PROVISIONAL\n\nIndependent Talim-reader semantic verification remains incomplete. Recognition of manually corrected crops only; 14 validation tiers from three sources cannot establish generalization. Historical 74.23% CER is retained separately and is not the comparable baseline.\n\n| Metric | Frozen E1, final ink-preserving release | Selected fine-tune |\n|---|---:|---:|\n'
 for label,key in [('CER including spaces','cer'),('CER excluding whitespace','cer_excluding_whitespace'),('Exact-tier accuracy','exact_accuracy')]:report+=f"| {label} | {baseline['overall'][key]:.4%} | {best['metrics']['overall'][key]:.4%} |\n"
 report+='\nRegression CER:\n\n| Cohort | Before | After |\n|---|---:|---:|\n'
 for k in reg:report+=f"| {k} | {base_reg[k]['cer']:.4%} | {reg[k]['cer']:.4%} |\n"
 report+=f"\nSelected checkpoint: {best['path']}\n\nSHA-256: {best['sha256']}\n\nReady for further development testing under prespecified regression gates: {ready}. No deployment endorsement. Original E1 and prior results preserved. No Git operations.\n"
 (O/'final_report.md').write_text(report,encoding='utf-8');print(report,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  dump('blocker.json',dict(error=repr(e),status='STOPPED; do not train if import integrity failed'));raise
