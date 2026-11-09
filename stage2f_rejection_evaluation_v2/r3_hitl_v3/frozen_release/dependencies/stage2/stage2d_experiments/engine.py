"""D1 image-only band grouping and confidence-aware row grammar.
Scores are heuristics, never calibrated probabilities. No OCR or labels.
"""
import math,cv2,numpy as np
GRAMMAR=(('count',),('upper','count'),('count','lower'),('upper','count','lower'))
def runs(active):
 d=np.diff(np.pad(active.astype(np.int8),(1,1)))
 return list(zip(np.where(d==1)[0].tolist(),np.where(d==-1)[0].tolist()))
def search(bands,features,scale):
 """Exact k=2 DP: state is consumed band index; additive suffix-independent scores."""
 pitch=scale+max(6,scale*.25);dp={0:[(0.,[])]}
 for i in range(len(bands)):
  if i not in dp:continue
  for roles in GRAMMAR:
   size=len(roles)
   if i+size>len(bands):continue
   score=-1.4
   if i:
    separation=bands[i][0]-bands[i-1][1]
    if separation<scale*.65:continue
    score+=float(np.clip((separation-scale*.55)/scale,-1.,1.))
   allowed=True
   for j,role in enumerate(roles):
    evidence=features[i+j]['prob'];score+=math.log(evidence if role=='count' else 1-evidence)
    if j:
     distance=(sum(bands[i+j])-sum(bands[i+j-1]))/2
     if distance>1.8*pitch:allowed=False;break
     score-=.35*((distance-pitch)/pitch)**2
   if not allowed:continue
   for previous,path in dp[i]:dp.setdefault(i+size,[]).append((previous+score,path+[(i,list(roles))]))
   dp[i+size]=sorted(dp[i+size],key=lambda v:v[0],reverse=True)[:2]
 return dp.get(len(bands),[])
def group_runs(raw,scale,extent,gap_ratio=.25):
 bands=[];gap=max(2,min(12,round(scale*gap_ratio)))
 for y0,y1 in raw:
  if bands and y0-bands[-1][1]<=gap and y1-bands[-1][0]<=extent*scale:bands[-1][1]=y1
  else:bands.append([y0,y1])
 return bands
def propose(image,configuration):
 _,ink=cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
 n,cc,stats,_=cv2.connectedComponentsWithStats(ink)
 if n<2:raise ValueError('empty:no_ink')
 valid=stats[1:,cv2.CC_STAT_AREA]>=3
 if not valid.any():raise ValueError('components:insufficient_support')
 binary=np.concatenate(([False],valid))[cc]
 raw=runs(binary.sum(axis=1)>=max(1,round(image.shape[1]*.0008)))
 if not raw or len(raw)>100:raise ValueError('bands:ambiguous_or_excessive')
 component_scale=max(8.,float(np.percentile(stats[1:,cv2.CC_STAT_HEIGHT][valid],85)))
 # Whole-band extent prevents tiny disconnected components from setting count pitch.
 band_scale=float(np.percentile([b-a for a,b in raw],85))
 scale=max(component_scale,band_scale)
 bands=group_runs(raw,scale,configuration['merge_extent'],configuration['gap_ratio'])
 if any(b-a>configuration['maximum_band_extent']*scale for a,b in bands):raise ValueError('bands:unresolved_vertical_merge')
 covered=np.zeros(image.shape[0],bool)
 for a,b in bands:covered[a:b]=True
 if np.any(binary[~covered]):raise ValueError('components:unassigned_foreground')
 features=[]
 for y0,y1 in bands:
  roi=binary[y0:y1].astype(np.uint8)*255
  count,labels,props,_=cv2.connectedComponentsWithStats(roi)
  props=props[1:];props=props[props[:,cv2.CC_STAT_AREA]>=3]
  contours=cv2.findContours(roi,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)[1]
  holes=0 if contours is None else int((contours[0,:,3]>=0).sum())
  width=float(np.percentile(props[:,cv2.CC_STAT_WIDTH],70)) if len(props) else 1.
  ch=float(np.percentile(props[:,cv2.CC_STAT_HEIGHT],80)) if len(props) else 1.
  effective=max(ch,configuration['band_height_weight']*(y1-y0))
  rawscore=1.6*(effective/scale-.8)+1.3*(width/scale-.75)+1.4*min(1.,holes/max(len(props),1))
  evidence=float(np.clip(1/(1+math.exp(-rawscore)),.03,.97))
  features.append(dict(prob=evidence,height=y1-y0,component_height=ch,width=width,mass=int((roi>0).sum())))
 solutions=search(bands,features,scale)
 diag=dict(component_scale=component_scale,band_aware_scale=scale,bands=bands,features=features,legal_alternatives=len(solutions),heuristic_scores=[s[0] for s in solutions],score_is_calibrated_probability=False)
 if not solutions:return [],dict(**diag,rejection='grammar:no_legal_layout')
 score,path=solutions[0];margin=score-solutions[1][0] if len(solutions)>1 else None
 diag.update(role_margin=margin,path=path,inferred_rows=len(path))
 if margin is not None and margin<configuration['margin']:return [],dict(**diag,rejection='confidence:ambiguous_row_or_role')
 # A sole path has no margin measurement; image/band checks still apply.
 if any(features[i+j]['prob']<configuration['minimum_count_evidence'] for i,roles in path for j,role in enumerate(roles) if role=='count'):
  return [],dict(**diag,rejection='confidence:weak_count_morphology')
 rows=[];h,w=image.shape
 for ri,(start,roles) in enumerate(path):
  ys,xs=np.where(binary[bands[start][0]:bands[start+len(roles)-1][1]])
  x0=max(0,int(xs.min())-5);x1=min(w,int(xs.max())+6);row=[]
  for j,role in enumerate(roles):
   index=start+j;a,b=bands[index];lower=(bands[index-1][1]+a)//2 if index else 0;upper=(b+bands[index+1][0])//2 if index+1<len(bands) else h
   top=max(lower,a-4);bottom=min(upper,b+4)
   if top>=bottom:return [],dict(**diag,rejection='crop:no_safe_boundary')
   row.append(dict(row=ri,role=role,box=[x0,top,x1,bottom]))
  rows.append(row)
 return rows,dict(**diag,rejection='')

