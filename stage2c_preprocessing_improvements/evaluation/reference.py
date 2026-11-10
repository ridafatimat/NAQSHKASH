"""Offline annotations from audited renderer slots and recorded augmentation geometry.
Never imported by the deployable pipeline. No images/datasets are regenerated.
"""
import json
import cv2,numpy as np
from PIL import Image,ImageOps


def clean_reference(image,logical_rows):
 h,w=image.shape[:2];n=len(logical_rows)
 pitch=(h-48-(n-1)*20)/(3*n)
 if abs(pitch-round(pitch))>1e-6 or pitch<8:raise ValueError('Unknown renderer geometry')
 pitch=int(round(pitch));labels=np.zeros((h,w),np.uint16);entries=[];index=0
 for ri,row in enumerate(logical_rows):
  for ti,(name,key) in enumerate([('upper','upper'),('count','count_line'),('lower','lower')]):
   y=24+ri*(3*pitch+20)+ti*pitch
   ink=image[y:y+pitch]<128;present=bool(row[key].strip())
   if bool(ink.any())!=present:raise ValueError('Renderer slot/label inconsistency')
   if present:
    index+=1;region=labels[y:y+pitch];region[ink]=index
    entries.append(dict(id=index,row=ri,role=name,raw=row[key],truth=row[key].lstrip(' '),leading_offset=len(row[key])-len(row[key].lstrip(' '))))
 if np.any((image<128)&(labels==0)):raise ValueError('Unassigned source ink outside renderer slots')
 return labels,entries


def perspective_coefficients(w,h,op):
 dx=w*float(op['inset']);dy=h*float(op['inset'])
 src=np.float32([(0,0),(w-1,0),(w-1,h-1),(0,h-1)])
 if op['axis']=='horizontal':
  dst=[(dx,dy),(w-1-dx,0),(w-1-dx,h-1),(dx,h-1-dy)] if not op['reverse'] else [(dx,0),(w-1-dx,dy),(w-1-dx,h-1-dy),(dx,h-1)]
 else:
  dst=[(dx,dy),(w-1-dx,dy),(w-1,h-1-dy),(0,h-1-dy)] if not op['reverse'] else [(0,dy),(w-1,dy),(w-1-dx,h-1-dy),(dx,h-1-dy)]
 matrix=cv2.getPerspectiveTransform(np.float32(dst),src);matrix=matrix/matrix[2,2]
 return tuple(matrix.reshape(-1)[:8])


def augmented_reference(labels,recipe,expected_shape):
 im=ImageOps.expand(Image.fromarray(labels.astype(np.int32),mode='I'),border=int(recipe['padding']),fill=0)
 for op in recipe['operations']:
  if op['type']=='rotation':im=im.rotate(float(op['degrees']),resample=Image.Resampling.NEAREST,expand=True,fillcolor=0)
  elif op['type']=='perspective':im=im.transform(im.size,Image.Transform.PERSPECTIVE,perspective_coefficients(*im.size,op),resample=Image.Resampling.NEAREST,fillcolor=0)
 out=np.asarray(im,dtype=np.uint16)
 if tuple(out.shape)!=tuple(expected_shape):raise ValueError('Recorded transform dimensions differ from sample')
 return out


def assess(result,reference,entries):
 if not result['accepted']:return dict(correct=False,reason=result['reason'],structure=False,geometry=False,tiers=[])
 proposed=result['rows'];expected_rows=1+max(e['row'] for e in entries)
 if len(proposed)!=expected_rows:return dict(correct=False,reason='row_count_mismatch',structure=False,geometry=False,tiers=[])
 for ri,row in enumerate(proposed):
  if [t['role'] for t in row]!=[e['role'] for e in entries if e['row']==ri]:
   return dict(correct=False,reason='tier_role_or_grouping_mismatch',structure=False,geometry=False,tiers=[])
 ys,xs=np.where(reference!=0);ids=reference[ys,xs]
 points=cv2.perspectiveTransform(np.column_stack([xs,ys]).astype(np.float32)[None],result['matrix'])[0]
 px,py=points[:,0],points[:,1]
 checks=[];all_ok=True
 for row in proposed:
  for tier in row:
   entry=next(e for e in entries if e['row']==tier['row'] and e['role']==tier['role'])
   x0,y0,x1,y1=tier['box'];inside_box=(px>=x0)&(px<x1)&(py>=y0)&(py<y1)
   total=int((ids==entry['id']).sum());inside=int(((ids==entry['id'])&inside_box).sum());other=int(((ids!=entry['id'])&inside_box).sum())
   retention=inside/max(total,1);contamination=other/max(inside+other,1)
   ok=total>0 and retention>=.98 and contamination<=.02
   all_ok=all_ok and ok
   checks.append(dict(**entry,box=tier['box'],retention=retention,contamination=contamination,geometry_ok=ok))
 return dict(correct=all_ok,reason='' if all_ok else 'ink_loss_or_neighbour_contamination',structure=True,geometry=all_ok,tiers=checks)
