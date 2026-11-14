"""Tighten the pre-inference protocol: content equality is not source identity."""
import json
from collections import Counter
from common import HERE,read_csv,write_csv
p=HERE/'common.py';t=p.read_text(encoding='utf-8-sig')
t=t.replace('def eligible(stage1_splits,positive_split=None):','def eligible(stage1_splits,positive_split=None,identity_verified=True):')
t=t.replace("if splits=={'test'}:return 'general_stage1_test'","if splits=={'test'}:return 'general_stage1_test' if identity_verified else 'general_test_structure_match_exploratory'")
p.write_text(t,encoding='utf-8')
p=HERE/'audit.py';t=p.read_text(encoding='utf-8-sig')
t=t.replace("eligible(splits,parent['stage2a_split'] if key[0]=='positive' else None)","eligible(splits,parent['stage2a_split'] if key[0]=='positive' else None,identity_verified=any(m=='image_sha256' for s,n,m in found))")
t=t.replace("{'general_stage1_test','positive_frozen_test'} or", "{'general_stage1_test','positive_frozen_test','general_test_structure_match_exploratory'} or")
t=t.replace('All confirmed test parents, all 3 variants; plus first 100 unmapped general parents','All byte-confirmed test parents and positive frozen test parents, plus separately exploratory test-content-only matches, all 3 variants; plus first 100 unmapped general parents')
p.write_text(t,encoding='utf-8')
parents=read_csv(HERE/'manifests/parents.csv')
for row in parents:
 if row['cohort']=='general_stage1_test' and not any(x[2]=='image_sha256' for x in json.loads(row['stage1_matches'])):
  row['cohort']='general_test_structure_match_exploratory'
lookup={(r['dataset'],r['parent_id']):r['cohort'] for r in parents}
write_csv(HERE/'manifests/parents.csv',parents)
for name in ('all_variants.csv','evaluation_pages.csv'):
 rows=read_csv(HERE/'manifests'/name)
 for row in rows:row['cohort']=lookup[(row['dataset'],row['parent_id'])]
 write_csv(HERE/'manifests'/name,rows)
a=json.loads((HERE/'reports/dataset_audit.json').read_text())
a['parents_by_cohort']=dict(Counter(r['cohort'] for r in parents))
a['variants_by_cohort']=dict(Counter(r['cohort'] for r in read_csv(HERE/'manifests/all_variants.csv')))
a['selected_by_cohort']=dict(Counter(r['cohort'] for r in read_csv(HERE/'manifests/evaluation_pages.csv')))
a['source_identity_rule']='A complete logical-structure match without image-hash identity is explicitly exploratory, even when the content match is confined to Stage1 test.'
a['selection_protocol']='All 49 positive frozen-test parents (147 variants); 100 unknown general parents by SHA256(42:parent_id) (300 variants); one test-content-only general match (3 variants), separately exploratory. All frozen before inference.'
(HERE/'reports/dataset_audit.json').write_text(json.dumps(a,indent=2),encoding='utf-8')
print(json.dumps(a['selected_by_cohort'],indent=2))
