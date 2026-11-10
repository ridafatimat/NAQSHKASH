"""Independent selected-source TXT/JSON/rule correspondence audit."""
import json,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import *
from talim.talim_rules import canonical_row
parents=read_csv(HERE/'manifests/parents.csv');checked=[]
for dataset,c in SOURCE_CONFIG.items():
 with zipfile.ZipFile(c['clean']) as z:
  for p in [r for r in parents if r['dataset']==dataset]:
   data=json.loads(z.read(c['prefix']+p['source_json']))
   raw=z.read(c['prefix']+p['source_label']).decode('utf-8').replace('\r\r\n','\n').replace('\r\n','\n').replace('\r','\n')
   expected='\n\n'.join('\n'.join((r['upper'],r['count_line'],r['lower'])) for r in data['logical_rows'])
   assert raw.rstrip('\n')==expected.rstrip('\n'),p['parent_id']
   for r in data['logical_rows']:assert canonical_row(r['runs'])==(r['upper'],r['count_line'],r['lower'])
   checked.append(dict(dataset=dataset,parent_id=p['parent_id'],partition=p['partition'],txt_json_canonical_match=True))
write_json(HERE/'reports/annotation_correspondence.json',dict(parents_checked=len(checked),all_passed=True,records=checked))
print('TXT/JSON/production Talim rules agree for',len(checked),'selected source parents')
