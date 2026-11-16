"""Offline independent component retention/content-support diagnostics."""
import cv2,numpy as np
def audit(output,reference):
 if not output['accepted']:return dict(components=0,components_lost=0,components_weak_content=0,minimum_retention=None,minimum_content_support=None)
 n,labels,stats,_=cv2.connectedComponentsWithStats((reference>0).astype(np.uint8))
 valid=stats[:,cv2.CC_STAT_AREA]>=3;valid[0]=False
 ys,xs=np.where(valid[labels]);ids=labels[ys,xs]
 if len(ids)==0:return dict(components=0,components_lost=0,components_weak_content=0,minimum_retention=None,minimum_content_support=None)
 pts=cv2.perspectiveTransform(np.column_stack([xs,ys]).astype(np.float32)[None],output['matrix'])[0]
 px,py=pts[:,0],pts[:,1];inside=np.zeros(len(ids),bool)
 for row in output['rows']:
  for t in row:
   x0,y0,x1,y1=t['box'];inside|=(px>=x0)&(px<x1)&(py>=y0)&(py<y1)
 totals=np.bincount(ids,minlength=n);kept=np.bincount(ids[inside],minlength=n)
 fractions=kept[valid]/np.maximum(totals[valid],1)
 _,mask=cv2.threshold(output['image'],0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
 support=cv2.dilate(mask,np.ones((3,3),np.uint8))>0
 ix=np.rint(px).astype(int);iy=np.rint(py).astype(int);in_image=(ix>=0)&(iy>=0)&(ix<support.shape[1])&(iy<support.shape[0])
 dark=np.zeros(len(ids),bool);dark[in_image]=support[iy[in_image],ix[in_image]]
 supported=np.bincount(ids[dark&inside],minlength=n)[valid]/np.maximum(totals[valid],1)
 return dict(components=int(valid.sum()),components_lost=int((fractions<.95).sum()),components_weak_content=int((supported<.95).sum()),minimum_retention=float(fractions.min()),minimum_content_support=float(supported.min()))

