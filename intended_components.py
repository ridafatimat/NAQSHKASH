"""Offline per-component retention in its intended semantic tier, not crop union."""
from dcore import *
def intended_retention(output,ref,entries):
 if not output['accepted']:return dict(role_components=0,role_components_lost=0,minimum_intended_retention=None)
 boxes={(t['row'],t['role']):t['box'] for row in output['rows'] for t in row}
 if set(boxes)!={(e['row'],e['role']) for e in entries}:return dict(role_components=0,role_components_lost=0,minimum_intended_retention=None)
 number=0;lost=0;minimum=1.
 for e in entries:
  n,labels,stats,_=cv2.connectedComponentsWithStats((ref==e['id']).astype(np.uint8))
  valid=stats[:,cv2.CC_STAT_AREA]>=3;valid[0]=False
  ys,xs=np.where(valid[labels]);ids=labels[ys,xs]
  if not len(ids):continue
  points=cv2.perspectiveTransform(np.column_stack([xs,ys]).astype(np.float32)[None],np.asarray(output['matrix']))[0]
  x0,y0,x1,y1=boxes[(e['row'],e['role'])]
  inside=(points[:,0]>=x0)&(points[:,0]<x1)&(points[:,1]>=y0)&(points[:,1]<y1)
  total=np.bincount(ids,minlength=n);kept=np.bincount(ids[inside],minlength=n)
  retention=kept[valid]/np.maximum(total[valid],1)
  number+=int(valid.sum());lost+=int((retention<.95).sum());minimum=min(minimum,float(retention.min()))
 return dict(role_components=number,role_components_lost=lost,minimum_intended_retention=minimum if number else None)
def main():
 diag={key(r):r for r in json.loads((ROOT/'reports/evaluation_v1/diagnostics.json').read_text())}
 src=Sources();records=[]
 for p in pages():
  d=diag[key(p)];a=d['assessment']
  if not a['structure']:continue
  im,ref,entries=src.get(p);out=dict(accepted=True,rows=d['rows'],matrix=np.asarray(d['matrix']))
  records.append(dict(**{k:p[k] for k in ['dataset','partition','parent_id','case_id','kind']},correct_segmentation=a['correct'],**intended_retention(out,ref,entries)))
 csvout('reports/intended_component_audit.csv',records)
 dump('reports/intended_component_summary.json',dict(structurally_aligned_pages=len(records),role_components=sum(r['role_components'] for r in records),role_components_lost=sum(r['role_components_lost'] for r in records),reference_correct_pages_with_component_loss=sum(r['correct_segmentation'] and r['role_components_lost']>0 for r in records),minimum_pixels=3,minimum_retention=.95))
 print('Intended-tier component audit',len(records),'aligned pages,',sum(r['role_components_lost'] for r in records),'losses',flush=True)
if __name__=='__main__':main()

