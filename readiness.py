"""Offline training-readiness gate, independent label and geometry verification."""
from dcore import *
from collections import Counter
from talim.talim_rules import canonical_row
def main():
 outcomes={key(r):r for r in read(ROOT/'reports/evaluation_v1/pages.csv')}
 diagnostics={key(r):r for r in json.loads((ROOT/'reports/evaluation_v1/diagnostics.json').read_text())}
 predictions=read(ROOT/'reports/evaluation_v1/predictions.csv')
 intended={key(r):r for r in read(ROOT/'reports/intended_component_audit.csv')}
 parents={r['parent_id']:r for r in read(C/'manifests/parents.csv') if r['dataset']=='positive'}
 original=json.loads((old.A/'manifests/parent_assignments_complete_v1.json').read_text())['assignments']
 c=old.SOURCE_CONFIG['positive'];archive=zipfile.ZipFile(c['clean']);verified={}
 for parent,p in parents.items():
  data=json.loads(archive.read(c['prefix']+p['source_json']))
  raw=archive.read(c['prefix']+p['source_label']).decode('utf-8').replace('\r\r\n','\n').replace('\r\n','\n').replace('\r','\n')
  expected='\n\n'.join('\n'.join((r['upper'],r['count_line'],r['lower'])) for r in data['logical_rows'])
  assert raw.rstrip('\n')==expected.rstrip('\n')
  for row in data['logical_rows']:assert canonical_row(row['runs'])==(row['upper'],row['count_line'],row['lower'])
  verified[parent]=data['logical_rows']
 records=[];crops=[]
 for p in [p for p in pages() if p['dataset']=='positive']:
  r=outcomes[key(p)];diag=diagnostics[key(p)]
  # Original positive split stays training; observed internal validation remains excluded.
  assignment=original[p['parent_id']]
  if isinstance(assignment,dict):assignment=assignment.get('split')
  assert assignment=='train',assignment
  correct=r['correct_segmentation']=='True'
  compok=correct and int(r['components_lost'])==0 and int(r['components_weak_content'])==0 and int(intended[key(p)]['role_components_lost'])==0
  ready=p['partition']=='tuning' and compok and int(r['ctc_failures'])==0
  status='training_ready' if ready else 'excluded' if p['partition']!='tuning' else 'manual_review'
  reason='automated label, independent geometry, component/content support and CTC checks passed' if ready else 'preserve internal-validation parent isolation; never train on this cohort' if status=='excluded' else 'segmentation rejected: '+r['reason'] if r['proposal']!='True' else 'offline segmentation verification failed: '+r['reason'] if not correct else 'individual component/content-support review required' if not compok else 'CTC feasibility review required'
  entries=diag['assessment']['tiers'] if correct else []
  tierpaths=[]
  if ready:
   for t in entries:
    path=ROOT/'derived_crops'/p['case_id']/f"row{t['row']}_{t['role']}.png"
    assert path.is_file(),path
    pred=next(x for x in predictions if key(x)==key(p) and int(x['row'])==t['row'] and x['role']==t['role'])
    image=cv2.imread(str(path),0);cropkey=hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
    assert cropkey==pred['crop_sha256']
    tierpaths.append(str(path));crops.append(dict(image_id=p['case_id'],source_parent=p['parent_id'],dataset_split='train',experiment_partition='tuning',row=t['row'],tier_role=t['role'],ground_truth=t['truth'],raw_ground_truth=t['raw'],leading_offset=t['leading_offset'],processing_candidate='D1',crop_path=str(path),crop_png_sha256=old.digest(path),crop_pixel_sha256=cropkey,ctc_steps=pred['ctc_steps'],ctc_required=pred['ctc_required'],label_verified=True,segmentation_verified=True,verification_scope='automated offline renderer-ink and component checks; not human reviewed',fine_tuning_eligible=True))
  records.append(dict(image_id=p['case_id'],original_archive=c[p['kind']],original_image_member=p['image_member'],original_image_sha256=p['image_sha256'],source_parent=p['parent_id'],dataset_split=assignment,experiment_partition=p['partition'],ground_truth_transcription=json.dumps(verified[p['parent_id']],ensure_ascii=False),processing_candidate='D1',preprocessed_crop_paths=json.dumps(tierpaths),label_verification_status='TXT_JSON_canonical_verified',segmentation_verification_status='independent_geometry_components_content_verified' if compok else 'geometry_pass_component_review' if correct else 'not_verified',fine_tuning_eligible=ready,bucket=status,reason=reason,human_reviewed=False))
 csvout('manifests/fine_tuning_images.csv',records);csvout('manifests/fine_tuning_tiers.csv',crops)
 counts={}
 for bucket in ['training_ready','manual_review','excluded']:
  rs=[r for r in records if r['bucket']==bucket];counts[bucket]=dict(images=len(rs),parents=len({r['source_parent'] for r in rs}))
 dump('manifests/readiness_summary.json',dict(positive_images=len(records),source_parents=len(parents),buckets=counts,eligible_tiers=len(crops),unique_crop_pixel_hashes=len({r['crop_pixel_sha256'] for r in crops}),unique_target_strings=len({r['ground_truth'] for r in crops}),tiers_by_role=dict(Counter(r['tier_role'] for r in crops)),label_parents_reverified=len(verified),human_reviewed=0,annotation_limit='Recorded renderer/augmentation ideal-ink reference, not manual polygons; 3px minimum component, 1px content-support tolerance',no_training_performed=True))
 lines=['# Stage 2D fine-tuning readiness','', '| Bucket | Images | Unique parents |','|---|---:|---:|']
 for bucket,m in counts.items():lines.append(f"| {bucket} | {m['images']} | {m['parents']} |")
 lines+=['',f"{len(crops)} verified training-eligible tier crops ({len({r['crop_pixel_sha256'] for r in crops})} unique pixel hashes; {len({r['ground_truth'] for r in crops})} distinct target strings) were exported. Counts include clean and existing scanned variants, not independent pages from distinct sources. Parent counts overlap between ready and review buckets; variants of one training parent can have different processing outcomes.",'','Eligibility requires original training assignment AND Stage 2C tuning partition, independently reverified TXT/JSON/canonical labels, complete offline segmentation alignment, >=98% tier-ink retention and <=2% contamination, >=95% retention/content support for each ideal component >=3px, and repeat-aware CTC feasibility. Algorithm acceptance alone never establishes ground truth. Content support uses Otsu ink within a 1px neighbourhood and is an approximation on degraded pixels. Human-reviewed count is zero.','', 'All 46 previously exposed internal-validation positive parents remain excluded from training; original positive validation/test parents are absent entirely. All variants retain parent ownership. Labels, internal spaces and offsets are unchanged. Image manifests include archive/member paths, exact source hashes, full transcriptions, split, verification status and review/exclusion reason. Tier manifests include local crop paths, PNG/pixel hashes, row/role, raw/normalized label offsets and CTC lengths.','', 'The eligible subset is suitable for preparing a controlled fine-tuning loader/pilot, subject to reviewing representative difficult cases and freezing the training, validation, resource and stopping protocol. Its size does not establish that fine-tuning will improve recognition. Long spacing failures are a recognition target, while rejected/wrong segmentation pages require preprocessing/manual review. No end-to-end deployment approval and no independent generalization claim follows. Do not automatically start training; the checkpoint remains frozen.']
 (ROOT/'reports/fine_tuning_readiness.md').write_text('\n'.join(lines),encoding='utf-8')
 print(json.dumps(counts),len(crops),'tiers',flush=True)
if __name__=='__main__':main()

