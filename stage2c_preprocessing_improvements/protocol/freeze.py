"""Freeze parent-isolated development cohorts before candidate experiments."""
import hashlib,json,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import *
import cv2,numpy as np
ref=load_local('stage2c_reference','evaluation/reference.py')

def rank(pid):return hashlib.sha256(('stage2c:42:'+pid).encode()).hexdigest()

def main():
 if (HERE/'protocol/lock.json').exists():raise ValueError('Protocol already frozen')
 before=protected();write_json(HERE/'protocol/protected_before.json',before)
 audit=json.loads((B/'reports/dataset_audit.json').read_text())
 for name in ('positive','general'):
  assert digest(SOURCE_CONFIG[name]['clean'])==audit['datasets'][name]['clean_archive_sha256']
  assert digest(SOURCE_CONFIG[name]['scanned'])==audit['datasets'][name]['archive_sha256']
 parent_records=read_csv(B/'manifests/parents.csv');variants=read_csv(B/'manifests/all_variants.csv');exposed=read_csv(B/'manifests/evaluation_pages.csv')
 banned={(r['dataset'],r['parent_id']) for r in exposed}
 assignments=json.loads((A/'manifests/parent_assignments_complete_v1.json').read_text())['assignments']
 positives=sorted([r for r in parent_records if r['dataset']=='positive' and assignments[r['parent_id']]=='train'],key=lambda r:rank(r['parent_id']))
 generals=sorted([r for r in parent_records if r['dataset']=='general' and r['cohort']=='general_unassigned_exploratory' and ('general',r['parent_id']) not in banned],key=lambda r:rank(r['parent_id']))[:150]
 training_manifest=Path(r'C:\FYP Dataset Generation\NAQSHKASH_Stage1_Kaggle_Training\manifests\stratified_train_150k.csv')
 with training_manifest.open(encoding='utf-8-sig',newline='') as f:
  reader=csv.DictReader(f);training_ids=sorted({r['parent_image_id'] for r in reader},key=rank)[:100]
 normal=[dict(dataset='normal',parent_id=pid,source_image=f'train/images/{pid}.png',source_json=f'train/json/{pid}.json',source_label=f'train/labels/{pid}.txt',stage1_splits='train',stage2a_split='',cohort='confirmed_training_clean_regression') for pid in training_ids]
 selected=[]
 for group,records in [('positive',positives),('general',generals),('normal',normal)]:
  for index,parent in enumerate(records):
   part='clean_regression' if group=='normal' else 'tuning' if index<(182 if group=='positive' else 100) else 'validation'
   selected.append(dict(dataset=group,parent_id=parent['parent_id'],partition=part,source_image=parent['source_image'],source_json=parent['source_json'],source_label=parent['source_label'],source_identity=parent.get('source_sha256',''),scope='exploratory_development_only' if group=='general' else 'confirmed_training_parent'))
 assert len(positives)==228 and len(selected)==478
 assert not any((r['dataset'],r['parent_id']) in banned for r in selected)
 cases=[];page_keys={};hash_partitions={};semantic_partitions={}
 for dataset in SOURCE_CONFIG:
  config=SOURCE_CONFIG[dataset]
  with zipfile.ZipFile(config['clean']) as clean:
   for parent in [r for r in selected if r['dataset']==dataset]:
    image_bytes=clean.read(config['prefix']+parent['source_image']);image=cv2.imdecode(np.frombuffer(image_bytes,np.uint8),0)
    digest_image=hashlib.sha256(image_bytes).hexdigest()
    if parent['source_identity']:assert parent['source_identity']==digest_image
    parent['source_identity']=digest_image
    data=json.loads(clean.read(config['prefix']+parent['source_json']));labels,entries=ref.clean_reference(image,data['logical_rows'])
    semantic=hashlib.sha256(json.dumps([[(r['symbol'],r['count'],r['direction']) for r in row['runs']] for row in data['logical_rows']],separators=(',',':')).encode()).hexdigest()
    # Identical complete source content is not allowed to cross candidate-selection partitions.
    for lookup,key in [(hash_partitions,digest_image),(semantic_partitions,semantic)]:
     previous=lookup.get(key)
     if previous and previous!=parent['partition']:raise ValueError('Related source content crosses partitions')
     lookup[key]=parent['partition']
    key=(dataset,parent['parent_id']);page_keys[key]=parent
    base=dict(dataset=dataset,parent_id=parent['parent_id'],partition=parent['partition'],scope=parent['scope'],source_image=parent['source_image'],source_json=parent['source_json'],source_sha256=digest_image)
    cases.append(dict(**base,case_id=parent['parent_id']+'__clean',kind='clean',image_member=config['prefix']+parent['source_image'],image_sha256=digest_image,augmentation='clean',parameters='',width=image.shape[1],height=image.shape[0]))
 for variant in variants:
  key=(variant['dataset'],variant['parent_id'])
  if key not in page_keys:continue
  parent=page_keys[key]
  cases.append(dict(dataset=key[0],parent_id=key[1],partition=parent['partition'],scope=parent['scope'],source_image=parent['source_image'],source_json=parent['source_json'],source_sha256=parent['source_identity'],case_id=variant['sample_id'],kind='scanned',image_member=variant['image_member'],image_sha256=variant['augmented_sha256'],augmentation=variant['augmentation'],parameters=variant['parameters'],width=int(variant['width']),height=int(variant['height'])))
 assert len(cases)==1612
 write_csv(HERE/'manifests/parents.csv',selected);write_csv(HERE/'manifests/pages.csv',sorted(cases,key=lambda r:(r['partition'],r['dataset'],r['parent_id'],r['case_id'])))
 write_json(HERE/'protocol/candidates.json',CANDIDATES)
 specification=dict(seed=42,retention_min=.98,contamination_max=.02,reference='Audited renderer slots (padding24, gap8, rowgap20), verified all clean ink and tier presence; nearest-neighbour label-only recorded geometric transforms offline; no regenerated datasets.',
  selection='Rank candidate configurations only on tuning parents. Highest correctly segmented scanned-page fraction; then lowest wrong acceptance, then common-subset CER; validation not used in selection.',
  claims='Development validation, not an independent test. General sources remain exploratory. Exposed Stage2A/2B positive tests and general Stage2B parents excluded.',
  split_summary={part:{d:sum(r['partition']==part and r['dataset']==d for r in selected) for d in SOURCE_CONFIG} for part in ['tuning','validation','clean_regression']},
  page_count=len(cases),parent_count=len(selected),exposed_parent_exclusion_count=len(banned),candidate_ids=list(CANDIDATES))
 write_json(HERE/'protocol/design.json',specification)
 files=[HERE/'manifests/parents.csv',HERE/'manifests/pages.csv',HERE/'protocol/candidates.json',HERE/'protocol/design.json',B/'manifests/parents.csv',B/'manifests/evaluation_pages.csv',B/'manifests/all_variants.csv',B/'reports/dataset_audit.json',A/'manifests/parent_assignments_complete_v1.json',CHECKPOINT,VOCABULARY,training_manifest,PROJECT/'data/ctc_decode.py',PROJECT/'data/collate.py']
 files += [Path(c[k]) for c in SOURCE_CONFIG.values() for k in ['clean','scanned'] if c[k]]
 files += [Path(r'C:\FYP Dataset Generation\dataset_tools\talim\renderer.py'),Path(r'C:\FYP Dataset Generation\talamat_scan_scripts_v1_1\talamat_scan_augment.py')]
 lock=dict(locked_files={str(p):digest(p) for p in set(files)},scope='Parent-isolated preprocessing development; no independent test claim',stage1_checkpoint_sha256=digest(CHECKPOINT))
 write_json(HERE/'protocol/lock.json',lock)
 assert protected()==before
 print(json.dumps(specification,indent=2))

if __name__=='__main__':main()
