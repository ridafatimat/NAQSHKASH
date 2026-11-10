"""Image-only component-aware row grammar and adaptive tier crop boundaries."""
import math,cv2,numpy as np


def _runs(active):
 d=np.diff(np.pad(active.astype(np.int8),(1,1)))
 return list(zip(np.where(d==1)[0].tolist(),np.where(d==-1)[0].tolist()))


def propose(image,strict=False):
 _,ink=cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
 n,cc,stats,_=cv2.connectedComponentsWithStats(ink)
 if n<2:raise ValueError('no_ink')
 areas=stats[1:,cv2.CC_STAT_AREA];heights=stats[1:,cv2.CC_STAT_HEIGHT]
 valid=(areas>=3)
 if not valid.any():raise ValueError('insufficient_components')
 scale=max(8.,float(np.percentile(heights[valid],85)))
 keep=np.concatenate(([False],valid));binary=keep[cc]
 active=binary.sum(axis=1)>=max(1,round(image.shape[1]*.0008))
 runs=_runs(active);bands=[]
 gap=max(2,min(6,round(scale*.18)))
 for y0,y1 in runs:
  if bands and y0-bands[-1][1]<=gap:bands[-1][1]=y1
  else:bands.append([y0,y1])
 if not bands or len(bands)>100:raise ValueError('ambiguous_component_bands')
 features=[]
 for y0,y1 in bands:
  roi=(binary[y0:y1].astype(np.uint8)*255)
  count,labels,props,_=cv2.connectedComponentsWithStats(roi)
  props=props[1:];props=props[props[:,cv2.CC_STAT_AREA]>=3]
  holes=cv2.findContours(roi,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)[1]
  hole_count=0 if holes is None else int((holes[0,:,3]>=0).sum())
  median_width=float(np.percentile(props[:,cv2.CC_STAT_WIDTH],70)) if len(props) else 1.
  height=y1-y0
  component_height=float(np.percentile(props[:,cv2.CC_STAT_HEIGHT],80)) if len(props) else 1.
  raw=1.6*(component_height/scale-.8)+1.3*(median_width/scale-.75)+1.4*min(1.,hole_count/max(len(props),1))
  probability=float(np.clip(1/(1+math.exp(-raw)),.03,.97))
  features.append(dict(prob=probability,height=height,mass=int(roi.sum()/255)))
 # Dynamic programming over optional upper, mandatory count, optional lower.
 pitch=scale+max(6,scale*.25)
 dp={0:[(0.,[])]}
 for i in range(len(bands)):
  if i not in dp:continue
  for size,roles in [(1,['count']),(2,['upper','count']),(2,['count','lower']),(3,['upper','count','lower'])]:
   if i+size>len(bands):continue
   score=-1.4
   if i>0:
    separation=bands[i][0]-bands[i-1][1]
    if separation<scale*.65:continue
    score+=float(np.clip((separation-scale*.55)/scale,-1.,1.))
   allowed=True
   for j,role in enumerate(roles):
    prob=features[i+j]['prob'];score+=math.log(prob if role=='count' else 1-prob)
    if j:
     distance=(sum(bands[i+j])-sum(bands[i+j-1]))/2
     if distance>1.8*pitch:allowed=False;break
     score-=.35*((distance-pitch)/pitch)**2
   if not allowed:continue
   for previous,path in dp[i]:
    target=i+size;option=(previous+score,path+[(i,roles)])
    dp.setdefault(target,[]).append(option)
   dp[target]=sorted(dp[target],key=lambda x:x[0],reverse=True)[:2]
 if len(bands) not in dp:raise ValueError('no_valid_row_grammar')
 solutions=dp[len(bands)];best,path=solutions[0]
 margin=best-solutions[1][0] if len(solutions)>1 else 99.
 if strict and margin<.2:raise ValueError('ambiguous_role_or_row_assignment')
 rows=[];h,w=image.shape
 for ri,(start,roles) in enumerate(path):
  y_start=bands[start][0];y_end=bands[start+len(roles)-1][1]
  ys,xs=np.where(binary[y_start:y_end])
  x0=max(0,int(xs.min())-5);x1=min(w,int(xs.max())+6)
  row=[]
  for j,role in enumerate(roles):
   index=start+j;a,b=bands[index]
   # Only shrink padding into observed whitespace; never bisect foreground ink.
   lower=(bands[index-1][1]+a)//2 if index else 0
   upper=(b+bands[index+1][0])//2 if index+1<len(bands) else h
   top=max(lower,a-4);bottom=min(upper,b+4)
   if top>=bottom:raise ValueError('no_safe_tier_boundary')
   row.append(dict(row=ri,role=role,box=[x0,top,x1,bottom]))
  rows.append(row)
 return rows,dict(role_margin=float(margin),band_count=len(bands),component_scale=scale)
