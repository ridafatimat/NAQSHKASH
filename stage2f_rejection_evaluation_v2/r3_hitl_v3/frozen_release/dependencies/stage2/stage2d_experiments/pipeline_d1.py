"""Deployable API: pixels plus frozen configuration only."""
import sys
from pathlib import Path
import cv2,numpy as np
C=Path(__file__).resolve().parent.parent/'preprocessing_improvements'
sys.path.insert(0,str(C))
import core
photo=core.load_local('stage2d_photo_readonly','preprocessing/photometric.py')
geo=core.load_local('stage2d_geo_readonly','preprocessing/geometry.py')
from preprocessing.pipeline import crop_and_separate_rows
# Explicit local import avoids collision with the historical engine module.
import importlib.util
spec=importlib.util.spec_from_file_location('stage2d_image_engine',Path(__file__).with_name('engine.py'))
engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
def prepare(image):
 pixels=photo.flatten(image);pixels,applied=photo.conservative_denoise(pixels)
 corrected,M,extra=geo.correct(pixels)
 result=crop_and_separate_rows(corrected,apply_deskew=False,apply_cleaning=not applied)
 return result.deskewed_image,M,dict(**extra,noise_filter_applied=applied)
def process_prepared(pixels,matrix,configuration,extra=None):
 try:
  rows,diag=engine.propose(pixels,configuration)
  reason=diag['rejection'];return dict(accepted=not bool(reason),reason=reason,rows=rows if not reason else [],image=pixels,matrix=matrix,diagnostics=dict(**(extra or {}),**diag))
 except (ValueError,RuntimeError,cv2.error) as exc:
  return dict(accepted=False,reason=str(exc),rows=[],image=pixels,matrix=matrix,diagnostics=dict(extra or {}))
def process_image(image,configuration):
 if not isinstance(image,np.ndarray) or image.ndim!=2 or image.dtype!=np.uint8 or not image.size:raise ValueError('Expected nonempty uint8 grayscale pixels')
 try:pixels,M,extra=prepare(image.copy())
 except (ValueError,RuntimeError,cv2.error) as exc:return dict(accepted=False,reason='preprocessing:'+str(exc),rows=[],image=image.copy(),matrix=np.eye(3),diagnostics={})
 return process_prepared(pixels,M,configuration,extra)

