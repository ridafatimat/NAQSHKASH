"""Record a matched clean/corrupted crop for the reviewed background failure."""
import json,zipfile
import cv2,numpy as np
from common import *
from modes import preprocess,assisted_segment,align_after_segmentation
root=HERE/'reports/evaluation_v1'
rows=read_csv(root/'predictions.csv')
record=next(r for r in rows if r['sample_id']=='all_001893__aug_03_brightness_noise' and r['mode']=='assisted' and r['truth']=='BH')
parent=next(r for r in read_csv(HERE/'manifests/parents.csv') if r['dataset']=='general' and r['parent_id']==record['parent_id'])
config=DATASETS['general'];vocab,_=load_vocabulary(VOCABULARY)
with zipfile.ZipFile(config['clean']) as z:
 data=json.loads(z.read(config['clean_prefix']+parent['source_json']))
 image=cv2.imdecode(np.frombuffer(z.read(config['clean_prefix']+parent['source_image']),np.uint8),0)
processed=preprocess(image);tiers=align_after_segmentation(assisted_segment(processed,data['logical_rows']),data['logical_rows'],vocab)
clean=next(t['image'] for t in tiers if t['row_index']==int(record['row_index']) and t['tier_name']==record['tier_name'])
example=next(r for r in read_csv(root/'failure_examples.csv') if r['sample_id']==record['sample_id'] and r['mode']=='assisted' and r['event']=='recognition')
scanned=cv2.imread(str(root/example['image_path']),0)
cv2.imwrite(str(root/'examples/reviewed_background_clean_tier.png'),clean)
clean_prediction=record['clean_prediction']
review=dict(sample_id=record['sample_id'],mode='assisted',tier_name=record['tier_name'],row_index=int(record['row_index']),truth=record['truth'],scanned_prediction=record['prediction'],clean_prediction=clean_prediction,
 clean_p50=float(np.percentile(clean,50)),clean_p95=float(np.percentile(clean,95)),scanned_p50=float(np.percentile(scanned,50)),scanned_p95=float(np.percentile(scanned,95)),
 notes=['Visual review confirms the two symbol clusters are present in the recorded BH tier crop; scanned prediction U differs from paired clean BH.',
 'The reviewed scanned crop has a visibly grey grainy background whereas the clean reference has a white background. Quantiles describe the full crop, not a certified background-only mask.',
 'Perspective examples show sloping ink bands with overlapping vertical extents; fixed row projection and padded tier boundaries can become ambiguous.',
 'These observations support investigation of illumination/background normalization and geometric segmentation before any training; they are not isolated causal ablations.'])
(root/'qualitative_review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
print(json.dumps(review,indent=2))
