import sys,json,time,copy,statistics
from pathlib import Path
import v5b_engine as e
from PIL import Image,ImageDraw
def series(records,key):
 return [d for r in records for d in r['dot_diameters']] if key=='dot_diameter' else [r[key] for r in records]
def summary(records):
 return {key:dict(n=len(vals),median=float(e.np.median(vals)),p10=float(e.np.quantile(vals,.1)),p90=float(e.np.quantile(vals,.9))) for key in ['stroke_width','ink_darkness','dot_diameter'] for vals in [series(records,key)] if vals}
def plot(groups,path):
 im=Image.new('RGB',(1650,700),'white');d=ImageDraw.Draw(im)
 for panel,(key,limit) in enumerate([('stroke_width',15),('ink_darkness',150),('dot_diameter',35)]):
  x0=65+panel*550;y0=610;d.text((x0,15),key+' (64px input)',fill='black');d.line((x0,100,x0,y0,x0+440,y0),fill='black',width=2)
  for k,(name,records,color) in enumerate(groups):
   vals=series(records,key);hist,edges=e.np.histogram(vals,bins=40,range=(0,limit));freq=hist/max(1,len(vals));pts=[(x0+int((i+.5)*440/40),y0-int(v*440)) for i,v in enumerate(freq)]
   d.line(pts,fill=color,width=3);d.text((x0,40+17*k),name,fill=color)
  for tick in range(6):
   x=x0+tick*88;d.line((x,y0,x,y0+5),fill='black');d.text((x-8,y0+12),str(round(limit*tick/5,1)),fill='black')
  d.text((x0,650),'Y: fraction per histogram bin',fill='black')
 im.save(path)
def prepare():
 cfg=e.config();rows,replay,dev=e.inputs();originals=[r for r in rows if r['sample_type']=='original'];out=e.ROOT/'quality_safe4';out.mkdir(exist_ok=True)
 assert not (out/'acceptance.json').exists(),'Preserve completed safe4 QA'
 # Protect V5 after the specifically authorized stop-record update.
 lock={str(p):e.sha(p) for p in e.V5.rglob('*') if p.is_file()};e.dump(e.ROOT/'V5_preservation_lock.json',lock)
 e.dump(e.ROOT/'input_integrity.json',dict(originals=105,augmented=2413,writers=sorted({r['writer_id'] for r in originals}),frozen_E1_sha256=e.sha(e.c.E/'checkpoints/selected_model/best.pt'),forbidden_images_opened=False,V5_modified_only_by_authorized_stop_record=True))
 real=[dict(id=r['id'],role=r['role'],**e.appearance(e.v2.normalize(e.v2.tensors(r),r['role'],e.h.CONTRAST)[0])) for r in originals]
 realraw=[dict(id=r['id'],role=r['role'],**e.appearance(e.v2.tensors(r))) for r in originals]
 clean=[dict(id=r['id'],role=r['role'],**e.appearance(e.v2.tensors(r))) for r in replay[:200]]
 e.write(out/'real_handwriting_measurements.csv',real);e.write(out/'real_raw_measurements.csv',realraw);e.write(out/'clean_synthetic_measurements.csv',clean)
 darkness=[r['ink_darkness'] for r in real];e.dump(out/'darkness_reference.json',darkness)
 realstats=summary(real);rounds=[];cache=e.ImageCache(cfg['image_cache_size'])
 for n in range(1,cfg['augmentation']['qa_max_rounds']+1):
  roundcfg=copy.deepcopy(cfg['augmentation']);roundcfg['darkness_offset']-=7*(n-1)
  sampler=e.Safe4(roundcfg,darkness);rng=e.np.random.default_rng(20261110+n);folder=out/f'round_{n:02}'
  if folder.exists():assert not any(folder.rglob('*.png')),'Preserve prior QA sample outputs'
  folder.mkdir(exist_ok=True);(folder/'samples').mkdir(exist_ok=True);(folder/'contact_sheets').mkdir(exist_ok=True);records=[]
  for page in range(10):
   sheet=Image.new('RGB',(1600,3000),'white');draw=ImageDraw.Draw(sheet)
   for slot in range(20):
    for retry in range(200):
     r=replay[int(rng.integers(len(replay)))];source=cache.get(r)
     try:aug,diag=sampler.sample(source,rng,r['id']);break
     except RuntimeError:continue
    else:raise RuntimeError('No safe4 example after resampling 200 sources')
    k=len(records);a=e.appearance(aug);a['dot_diameters']=diag['dot_diameters'];records.append(dict(index=k,id=r['id'],role=r['role'],**{**diag,**a}))
    Image.fromarray(e.np.uint8(aug.numpy()[0]*255)).save(folder/'samples'/f'{k:03}.png')
    for col,t in enumerate([source,aug]):
     image=Image.fromarray(e.np.uint8(e.np.clip(t.numpy()[0]*255,0,255))).convert('RGB');image.thumbnail((780,110));sheet.paste(image,(col*800,slot*150+32))
    draw.text((5,slot*150+3),r['id']+f" | source / safe4 | core {a['ink_darkness']:.0f} | dots "+str([round(v,1) for v in diag['dot_diameters']]),fill='black')
   sheet.save(folder/'contact_sheets'/f'pairs_{page:02}.jpg',quality=94)
   print('SAFE4 QA',n,'accepted',len(records),flush=True)
  stats=summary(records);vis=sampler.counters['visibility_rejections']/max(1,sampler.counters['visibility_checked'])
  checks=dict(darkness_median=abs(stats['ink_darkness']['median']-realstats['ink_darkness']['median'])<=cfg['augmentation']['darkness_median_tolerance'],dot_diameter_median=realstats['dot_diameter']['p10']<=stats['dot_diameter']['median']<=realstats['dot_diameter']['p90'],visibility_rejection_rate=vis<cfg['augmentation']['maximum_visibility_rejection_rate'])
  result=dict(round=n,passed=all(checks.values()),checks=checks,visibility_rejection_rate=vis,counters=dict(sampler.counters),statistics={'augmented':stats,'real_handwriting':realstats,'clean_synthetic':summary(clean)},configuration=roundcfg)
  e.write(folder/'augmented_measurements.csv',records);e.dump(folder/'round_report.json',result);plot([('Safe4 augmented',records,'blue'),('Real handwriting (contrast)',real,'red'),('Clean synthetic',clean,'green')],folder/'distributions.png')
  table=[dict(domain=domain,measure=key,**values) for domain,data in result['statistics'].items() for key,values in data.items()];e.write(folder/'distribution_summary.csv',table)
  rounds.append(result);e.dump(out/'all_rounds.json',rounds)
  if result['passed']:
   e.dump(out/'accepted_config.json',roundcfg);e.dump(out/'acceptance.json',dict(passed=True,accepted_round=n,checks=checks,real_reference_preprocessing=cfg['real_QA_preprocessing'],visibility_rejection_rate=vis,statistics=result['statistics']));break
 else:e.dump(out/'acceptance.json',dict(passed=False,rounds=rounds));raise RuntimeError('safe4 did not pass QA within three rounds; training is blocked')
 lines=['# Safe4 augmentation QA','',f'Accepted round: {n}. All three specified acceptance checks passed.','', '| Domain | Measure | Median | p10 | p90 | N |','|---|---|---:|---:|---:|---:|']
 lines += [f"| {r['domain']} | {r['measure']} | {r['median']:.2f} | {r['p10']:.2f} | {r['p90']:.2f} | {r['n']} |" for r in table]
 lines+=['',f'Visibility rejection rate among candidates reaching the visibility check: {vis:.2%}. Boundary, topology, correspondence and dot-size rejection counts are reported separately. Rejected candidates are resampled; this changes the accepted parameter distribution.','', 'Core darkness is median grayscale at distance-transform ridges. Dot diameter is the geometric mean of the outer bounding-box width/height, so hollow interiors do not artificially reduce diameter. Real reference statistics use the 105 training-writer originals after the established V3 contrast; raw-original measurements are also retained. The model evaluation pipeline is independently fixed to identity preprocessing for both E1 and candidates.','', 'Compact detached marks are redrawn separately, never skeletonized. Ring thickness equals the sampled body stroke; if this would eliminate the hole, the mark is filled. Every component must be visible (darkest pixel at most 120), map one-to-one, and retain body-hole topology. All transformed ink support remains in frame. Unconfirmed variants are disabled. These are automatic topology guards, not independent Talim-reader verification. Scale, rotation and pitch use whole-tier affine transforms; geometry rejection is explicitly logged.','',f'![Overlaid distributions](round_{n:02}/distributions.png)']
 lines += [f'![Paired samples {i}](round_{n:02}/contact_sheets/pairs_{i:02}.jpg)' for i in range(10)]
 (out/'augmentation_quality_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('SAFE4 ACCEPTED',json.dumps(result['checks']),flush=True)
if __name__=='__main__':e.c.seed(42);prepare()
