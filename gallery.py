"""Before/after galleries with full-resolution pixels and diagnostic sidecars."""
from dcore import *
from pipeline_d1 import process_image
import html
def main():
 config=json.loads((ROOT/'configs/D1.json').read_text());changes=read(ROOT/'reports/d1_case_changes.csv')
 prior=json.loads((ROOT/'galleries/selection.json').read_text());chosen=[(g['folder'].split('_',1)[0],g['case']) for g in prior]
 for part in ['tuning','validation','clean_regression']:
  for ds in ['positive','general','normal']:
   for event in [('wrong','correct'),('rejected','correct'),('correct','rejected'),('correct','wrong'),('wrong','wrong'),('rejected','rejected')]:
    rs=[r for r in changes if r['partition']==part and r['dataset']==ds and (r['C5'],r['D1'])==event]
    if rs and key(rs[0]) not in chosen:chosen.append(key(rs[0]))
 lookup={key(p):p for p in pages()};sources=Sources();items=[]
 preds=read(ROOT/'reports/evaluation_v1/predictions.csv')
 for k in chosen:
  p=lookup[k];image,ref,entries=sources.get(p);proc=Processor(image);out=process_image(image,config)
  folder=ROOT/'galleries'/(p['dataset']+'_'+p['case_id']);folder.mkdir(exist_ok=True)
  cv2.imwrite(str(folder/'original.png'),image)
  info=dict(page=p,expected=[{k:e[k] for k in ['row','role','raw','truth','leading_offset']} for e in entries],predictions=[r for r in preds if key(r)==k],candidates={})
  for name,result in [('C4',proc.run(old.CANDIDATES['C4'])),('C5',proc.run(old.CANDIDATES['C5'])),('D1',out)]:
   a=reference.assess(result,ref,entries);panel=cv2.cvtColor(result['image'],cv2.COLOR_GRAY2BGR);color=(0,140,0) if a['correct'] else (0,0,220)
   for row in result['rows']:
    for t in row:
     x0,y0,x1,y1=t['box'];cv2.rectangle(panel,(x0,y0),(x1-1,y1-1),color,1);cv2.putText(panel,t['role'],(x0,max(10,y0-2)),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
   cv2.imwrite(str(folder/(name+'.png')),panel);cv2.imwrite(str(folder/(name+'_pixels.png')),result['image'])
   cv2.imwrite(str(folder/(name+'_detail.png')),panel[:,:min(1000,panel.shape[1])])
   if panel.shape[1]>1000:cv2.imwrite(str(folder/(name+'_right.png')),panel[:,-1000:])
   info['candidates'][name]=dict(accepted=result['accepted'],assessment=a,rows=result['rows'],diagnostics=result['diagnostics'])
  old.write_json(folder/'evidence.json',info)
  items.append(dict(folder=folder.name,dataset=p['dataset'],case_id=p['case_id'],partition=p['partition'],kind=p['kind'],D1_accepted=out['accepted'],D1_correct=reference.assess(out,ref,entries)['correct'],reason=reference.assess(out,ref,entries)['reason']))
 sections=[]
 for g in items:
  folder=g['folder'];caption=html.escape(g['case_id']);status='correct' if g['D1_correct'] else 'wrong acceptance' if g['D1_accepted'] else 'rejection'
  imgs=''.join(f'<figure><figcaption>{name}</figcaption><a href="{folder}/{name}.png"><img loading="lazy" src="{folder}/{name}.png"></a>'+(f'<p><a href="{folder}/{name}_pixels.png">unaltered processed pixels</a> · <a href="{folder}/{name}_detail.png">left detail</a></p>' if name!='original' else '')+'</figure>' for name in ['original','C4','C5','D1'])
  info=json.loads((ROOT/'galleries'/folder/'evidence.json').read_text());truth='; '.join(f"row{e['row']} {e['role']}: {e['truth']}" for e in info['expected'])
  predictions='; '.join(f"row{e['row']} {e['role']}: {e['prediction']}" for e in info['predictions'])
  sections.append(f'<section><h2>{caption}</h2><p>{g["partition"]}; D1 {status}: {html.escape(g["reason"])}. <a href="{folder}/evidence.json">full evidence and predictions</a></p><div class="panels">{imgs}</div><details><summary>Expected / D1 predicted labels</summary><pre>Expected: {html.escape(truth)}\nPredicted (only correct, feasible crops): {html.escape(predictions)}</pre></details></section>')
 (ROOT/'galleries/index.html').write_text('<!doctype html><meta charset="utf-8"><title>NAQSHKASH Stage 2D comparisons</title><style>body{font:16px sans-serif;margin:24px;color:#19242b}section{border-top:1px solid #aaa;padding:16px 0}.panels{display:flex;gap:12px}figure{margin:0;width:25%}img{width:100%;max-height:650px;object-fit:contain;object-position:top}a{color:#146}pre{white-space:pre-wrap;overflow-wrap:anywhere}figcaption{font-weight:bold}</style><h1>Stage 2D segmentation diagnosis and D1 comparisons</h1><p>Click for full-resolution boundaries. Unaltered processed-pixel images preserve the evidence underneath overlays. Rejections produce no accepted crops. Expected labels and predictions are offline annotations; they never choose segmentation. All validation here was previously exposed; this gallery does not select D1.</p>'+''.join(sections),encoding='utf-8')
 dump('galleries/index.json',items)
 print('Gallery cases',len(items),flush=True)
if __name__=='__main__':main()

