"""Deployable image+frozen-config API. No annotation, recipe or text input."""
import hashlib
import cv2,numpy as np
from core import CANDIDATES,load_local,baseline_preprocess,baseline_tiers
from preprocessing.pipeline import crop_and_separate_rows
photo=load_local('stage2c_photometric','preprocessing/photometric.py')
geo=load_local('stage2c_geometry','preprocessing/geometry.py')
engine=load_local('stage2c_engine','segmentation/engine.py')

class Processor:
 def __init__(self,image):
  if not isinstance(image,np.ndarray) or image.size==0:raise ValueError('Invalid image')
  self.original=image.copy();self.cache={}
 def run(self,config):
  try:
   image=self.original
   if config['normalize']:
    if 'normalized' not in self.cache:self.cache['normalized']=photo.flatten(image)
    image=self.cache['normalized']
   denoise_applied=False
   if config['conservative']:image,denoise_applied=photo.conservative_denoise(image)
   signature=hashlib.sha256(image.tobytes()).hexdigest()
   key=(signature,config['geometry'],denoise_applied)
   if key not in self.cache:
    extra=dict(perspective_applied=False)
    if config['geometry']:
     corrected,M,extra=geo.correct(image)
     processed=crop_and_separate_rows(corrected,apply_deskew=False,apply_cleaning=not denoise_applied)
    else:
     processed=baseline_preprocess(image);M=np.eye(3)
     if processed.deskew_matrix is not None:M[:2]=processed.deskew_matrix
    self.cache[key]=(processed,M,extra)
   processed,M,extra=self.cache[key]
   if config['robust']:rows,diagnostics=engine.propose(processed.deskewed_image,strict=config['conservative'])
   else:
    rows=[];diagnostics={}
    for ri,(row,box) in enumerate(zip(processed.rows,processed.global_row_bboxes)):
     items=[];y,x,_,_=box
     for role,tier in baseline_tiers(row.image):
      offset=tier.__array_interface__['data'][0]-row.image.__array_interface__['data'][0]
      dy=offset//row.image.strides[0]
      items.append(dict(row=ri,role=role,box=[x,y+dy,x+tier.shape[1],y+dy+tier.shape[0]]))
     rows.append(items)
   return dict(accepted=True,reason='',rows=rows,image=processed.deskewed_image,matrix=M,diagnostics=dict(**diagnostics,**extra,noise_filter_applied=denoise_applied))
  except (ValueError,RuntimeError,cv2.error) as exc:
   return dict(accepted=False,reason=str(exc),rows=[],image=self.original,matrix=np.eye(3),diagnostics={})

def process_image(image,configuration):
 return Processor(image).run(configuration)
