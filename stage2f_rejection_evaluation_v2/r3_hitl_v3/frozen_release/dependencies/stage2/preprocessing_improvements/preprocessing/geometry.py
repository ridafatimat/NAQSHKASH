"""Image-only baseline-convergence correction with a no-warp fallback."""
import cv2,numpy as np
from core import PROJECT
from preprocessing.deskew import deskew


def mask(gray):
 _,ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
 return ink


def energy(ink):
 profile=(ink>0).sum(axis=1).astype(float)
 return float((profile*profile).sum()/max(profile.sum()**2,1))


def correct(image):
 result=deskew(image)
 first=result.image;M=np.eye(3)
 if result.matrix is not None:M[:2,:]=result.matrix
 ink=mask(first);h,w=ink.shape
 if w<100 or h<60:return first,M,dict(perspective_applied=False,reason='insufficient_extent')
 closed=cv2.morphologyEx(ink,cv2.MORPH_CLOSE,np.ones((1,max(9,min(35,w//25))),np.uint8))
 lines=cv2.HoughLinesP(closed,1,np.pi/1800,threshold=max(20,w//15),minLineLength=max(35,w//6),maxLineGap=30)
 observations=[]
 if lines is not None:
  for x0,y0,x1,y1 in lines[:,0]:
   if x1==x0:continue
   slope=(y1-y0)/(x1-x0)
   if abs(slope)>.15:continue
   at_center=y0+slope*(w/2-x0)
   observations.append((at_center,slope,abs(x1-x0)))
 if len(observations)<5:return first,M,dict(perspective_applied=False,reason='weak_line_support')
 ys=np.array([r[0] for r in observations]);slopes=np.array([r[1] for r in observations])
 if np.ptp(ys)<h*.2:return first,M,dict(perspective_applied=False,reason='insufficient_vertical_support')
 fit=np.polyfit(ys,slopes,1);residual=np.abs(slopes-np.polyval(fit,ys));keep=residual<=max(.006,3*np.median(residual))
 if keep.sum()<5:return first,M,dict(perspective_applied=False,reason='inconsistent_baselines')
 b,a=np.polyfit(ys[keep],slopes[keep],1)
 if abs(b*w)<.005 or abs(b*w)>.24 or abs(a)>.12:return first,M,dict(perspective_applied=False,reason='no_safe_projective_evidence')
 cx=w/2;T=np.array([[1.,0.,-cx],[0.,1.,0.],[0.,0.,1.]])
 inv=np.array([[1.,0.,cx],[0.,1.,0.],[0.,0.,1.]])
 H=inv@np.array([[1.,0.,0.],[-a,1.,0.],[b,0.,1.]])@T
 corners=np.float32([[[0,0],[w,0],[w,h],[0,h]]]);warped=cv2.perspectiveTransform(corners,H)[0]
 lower=np.floor(warped.min(axis=0));upper=np.ceil(warped.max(axis=0));nw,nh=(upper-lower).astype(int)
 if nw*nh>1.6*w*h or min(nw,nh)<10:return first,M,dict(perspective_applied=False,reason='unsafe_extent')
 shift=np.array([[1.,0.,-lower[0]],[0.,1.,-lower[1]],[0.,0.,1.]])
 H=shift@H
 test=cv2.warpPerspective(ink,H,(nw,nh),flags=cv2.INTER_NEAREST,borderValue=0)
 if energy(test)<energy(ink)*1.015:return first,M,dict(perspective_applied=False,reason='no_image_only_score_gain')
 output=cv2.warpPerspective(first,H,(nw,nh),flags=cv2.INTER_CUBIC,borderValue=255)
 return output,H@M,dict(perspective_applied=True,reason='supported_baseline_convergence',slope_a=float(a),slope_b=float(b))
