"""Full archive integrity and parent/split audit. Must finish before inference."""
import csv,hashlib,io,json,zipfile
from collections import Counter,defaultdict
from pathlib import Path
import cv2,numpy as np
from common import *
from talim.talim_rules import canonical_row

def main():
 destination=HERE/'reports/dataset_audit.json'
 if destination.exists():raise ValueError('Audit exists; preserve it')
 before=protected_snapshot()
 (HERE/'reports/protected_before.json').write_text(json.dumps(before,indent=2),encoding='utf-8')
 parents={};variants=[];summaries={};source_hash_targets=defaultdict(list);semantic_targets=defaultdict(list)
 assignments=json.loads((POSITIVE/'manifests/parent_assignments_complete_v1.json').read_text())['assignments']
 for dataset,config in DATASETS.items():
  with zipfile.ZipFile(config['scanned']) as scanned,zipfile.ZipFile(config['clean']) as clean:
   pre=config['prefix'];cp=config['clean_prefix']
   manifest=list(csv.DictReader(io.StringIO(scanned.read(pre+'augmentation_manifest.csv').decode('utf-8-sig'))))
   summary=json.loads(scanned.read(pre+'augmentation_summary.json'))
   inventory=json.loads(scanned.read(pre+'source_inventory.json'))
   clean_digest=sha256(config['clean'])
   assert clean_digest==summary['official_archive_sha256'],'Wrong clean archive'
   # Source inventory records hashes of clean image/label/JSON/metadata bytes.
   for member,digest in inventory['files'].items():
    assert hashlib.sha256(clean.read(cp+member)).hexdigest()==digest,(dataset,member)
   sources={row['source_id']:row for row in inventory['sources']}
   for pid,source in sources.items():
    data=json.loads(clean.read(cp+source['json_path']))
    assert data['sample_id']==pid
    text=clean.read(cp+source['label_path']).decode('utf-8').replace('\r\r\n','\n').replace('\r\n','\n').replace('\r','\n')
    expected='\n\n'.join('\n'.join((r['upper'],r['count_line'],r['lower'])) for r in data['logical_rows'])
    assert text.rstrip('\n')==expected.rstrip('\n'),(dataset,pid,'label')
    for row in data['logical_rows']:assert canonical_row(row['runs'])==(row['upper'],row['count_line'],row['lower'])
    raw=clean.read(cp+source['source_filename']);image=cv2.imdecode(np.frombuffer(raw,np.uint8),0)
    assert image is not None
    digest=hashlib.sha256(raw).hexdigest();key=(dataset,pid)
    parents[key]=dict(dataset=dataset,parent_id=pid,source_image=source['source_filename'],source_json=source['json_path'],source_label=source['label_path'],
        source_sha256=digest,semantic_sha256=semantic(data),rows=len(data['logical_rows']),stage1_matches=[],stage1_splits=set(),stage2a_split=assignments.get(pid,'') if dataset=='positive' else '')
    source_hash_targets[digest].append(key);semantic_targets[semantic(data)].append(key)
   image_members={n for n in scanned.namelist() if n.startswith(pre+'images/') and n.lower().endswith(('.jpg','.jpeg','.png'))}
   declared=set();counts=Counter()
   for i,row in enumerate(manifest):
    pid=row['source_id'];parent=parents[(dataset,pid)]
    assert row['source_sha256']==parent['source_sha256']
    assert row['source_filename']==parent['source_image'] and row['label_path']==parent['source_label'] and row['json_path']==parent['source_json']
    original=json.loads(row['original_metadata']);assert original['sample_id']==pid
    member=pre+row['augmented_filename'];assert member not in declared;declared.add(member)
    raw=scanned.read(member);assert hashlib.sha256(raw).hexdigest()==row['augmented_sha256']
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),0)
    assert image is not None and image.shape==(int(row['height']),int(row['width']))
    parameters=json.loads(row['augmentation_parameters']);ops=[o['type'] for o in parameters['operations']]
    assert row['augmentation_type'].split('+')==ops
    counts[pid]+=1
    variants.append(dict(dataset=dataset,sample_id=Path(row['augmented_filename']).stem,parent_id=pid,image_member=member,
      augmented_sha256=row['augmented_sha256'],source_sha256=row['source_sha256'],width=row['width'],height=row['height'],augmentation=row['augmentation_type'],
      operations=';'.join(ops),severity=row['severity'],variant_index=row['variant_index'],variant_seed=row['variant_seed'],parameters=row['augmentation_parameters']))
    if (i+1)%5000==0:print(dataset,'integrity checked',i+1,flush=True)
   assert declared==image_members and set(counts)==set(sources) and set(counts.values())=={3}
   assert len(manifest)==summary['total_augmented_images']
   summaries[dataset]=dict(config,archive_sha256=sha256(config['scanned']),clean_archive_sha256=clean_digest,variants=len(manifest),parents=len(sources),
      image_members=len(image_members),non_dataset_image_members=[n for n in scanned.namelist() if n.lower().endswith(('.jpg','.png')) and n not in image_members],
      embedded_label_files=sum(n.lower().endswith('.txt') for n in scanned.namelist()),labels='external clean TXT and JSON verified via source_inventory hashes',
      augmentation_combinations=dict(Counter(r['augmentation_type'] for r in manifest)),operation_counts=dict(Counter(op for r in manifest for op in r['augmentation_type'].split('+'))),
      severities=dict(Counter(r['severity'] for r in manifest)),raw_manifest_declares_splits=False,source_inventory_files_verified=len(inventory['files']),generation_summary=summary)
 print('Scanning all frozen Stage1 image hashes and logical structures',flush=True)
 matches=defaultdict(set);scanned_counts=Counter()
 with zipfile.ZipFile(STAGE1_ZIP) as stage1:
  for info in stage1.infolist():
   name=info.filename
   split=next((s for s in ('train','validation','test') if '/'+s+'/' in name),None)
   if split is None:continue
   if '/json/' in name and name.endswith('.json'):
    key=semantic(json.loads(stage1.read(info)));scanned_counts['json']+=1
    for parent in semantic_targets.get(key,[]):matches[parent].add((split,name,'logical_structure'))
   elif '/images/' in name and name.endswith('.png'):
    key=hashlib.sha256(stage1.read(info)).hexdigest();scanned_counts['images']+=1
    for parent in source_hash_targets.get(key,[]):matches[parent].add((split,name,'image_sha256'))
   else:continue
   if sum(scanned_counts.values())%20000==0:print('Stage1 audited',dict(scanned_counts),flush=True)
 parent_rows=[]
 for key,parent in parents.items():
  found=sorted(matches[key]);splits=sorted({s for s,n,m in found})
  parent['stage1_matches']=json.dumps(found);parent['stage1_splits']=';'.join(splits)
  parent['cohort']=eligible(splits,parent['stage2a_split'] if key[0]=='positive' else None,identity_verified=any(m=='image_sha256' for s,n,m in found))
  parent_rows.append(parent)
 lookup={(r['dataset'],r['parent_id']):r for r in parent_rows}
 for row in variants:
  parent=lookup[(row['dataset'],row['parent_id'])]
  row.update(cohort=parent['cohort'],stage1_splits=parent['stage1_splits'],stage2a_split=parent['stage2a_split'])
 # One fixed exploratory sample, only from parents without any confirmed Stage1 split.
 unknown=[r for r in parent_rows if r['cohort']=='general_unassigned_exploratory']
 selected_exploratory={r['parent_id'] for r in sorted(unknown,key=lambda r:hashlib.sha256(('42:'+r['parent_id']).encode()).hexdigest())[:100]}
 selected=[]
 for row in variants:
  keep=row['cohort'] in {'general_stage1_test','positive_frozen_test','general_test_structure_match_exploratory'} or (row['cohort']=='general_unassigned_exploratory' and row['parent_id'] in selected_exploratory)
  row['selected_for_inference']=keep
  if keep:selected.append(row)
 write_csv(HERE/'manifests/parents.csv',parent_rows)
 write_csv(HERE/'manifests/all_variants.csv',variants)
 write_csv(HERE/'manifests/evaluation_pages.csv',selected)
 assert protected_snapshot()==before
 result=dict(status='passed',datasets=summaries,stage1_archive=str(STAGE1_ZIP),stage1_scan=dict(scanned_counts),
    parents_by_cohort=dict(Counter(r['cohort'] for r in parent_rows)),variants_by_cohort=dict(Counter(r['cohort'] for r in variants)),
    selected_by_cohort=dict(Counter(r['cohort'] for r in selected)),selected_pages=len(selected),
    selection_protocol='All byte-confirmed test parents and positive frozen test parents, plus separately exploratory test-content-only matches, all 3 variants; plus first 100 unmapped general parents by SHA256(42:parent_id), all 3 variants, exploratory only. Frozen before inference.',
    duplicate_augmented_hashes=len(variants)-len({r['augmented_sha256'] for r in variants}),
    cross_split_parent_augmentation_leakage=False,split_note='No random augmented-image split. All variants inherit the same source-parent classification. Matches to any train/validation split are excluded.',
    caveats=['General requested filename talim_scanned_augmented_v1.zip not found; actual manifest-verified 30k archive is talim_10000_scanned.zip.',
      'No Stage1 match does not establish eligibility under a previously frozen test protocol; such general parents are exploratory.',
      'Positive test parents inherit all 326 Stage2A frozen assignments; prior test exposure is retained and disclosed.',
      'Whole-page image hashes and full logical structures detect exact relationships; arbitrary partial or transformed overlaps are not ruled out.'])
 destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k!='datasets'},indent=2))

if __name__=='__main__':main()
