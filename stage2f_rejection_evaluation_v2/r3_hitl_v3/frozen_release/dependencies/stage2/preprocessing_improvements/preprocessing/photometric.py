"""Image-only photometry. No OCR, annotations, recipes or sample identities."""
import cv2,numpy as np


def flatten(image):
 gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image.copy()
 if np.percentile(gray,50)>=250 and np.percentile(gray,95)>=254:return gray
 k=max(15,(min(gray.shape)//12)|1)
 background=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
 background=cv2.GaussianBlur(background,(0,0),max(2,k/4))
 return cv2.divide(gray,np.maximum(background,1),scale=255)


def conservative_denoise(image):
 white=image[image>220]
 texture=float(np.median(np.abs(white-np.median(white)))) if len(white) else 0
 if texture<2:return image.copy(),False
 return cv2.bilateralFilter(image,5,12,2),True
