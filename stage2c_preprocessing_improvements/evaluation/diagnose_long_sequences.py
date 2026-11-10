"""Explain character-weighted long-sequence failures without removing them."""
import json,sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import *
root=HERE/'reports/experiments_v1';rows=read_csv(root/'predictions.csv');totals=defaultdict(int)
for r in rows:
 if r['candidate']=='C4' and r['partition']=='validation' and r['dataset']=='positive' and r['kind']=='scanned':totals[r['parent_id']]+=len(r['truth'])
longest=max(totals,key=totals.get);output={}
for candidate in CANDIDATES:
 group=[r for r in rows if r['candidate']==candidate and r['partition']=='validation' and r['dataset']=='positive' and r['kind']=='scanned']
 by=defaultdict(lambda:[0,0])
 for r in group:
  by[r['parent_id']][0]+=sum(int(r[k]) for k in ['substitutions','deletions','insertions']);by[r['parent_id']][1]+=len(r['truth'])
 error=sum(v[0] for v in by.values());chars=sum(v[1] for v in by.values());specific=by.get(longest,[0,0])
 output[candidate]=dict(scored_parents=len(by),macro_parent_cer=sum(v[0]/v[1] for v in by.values())/len(by),micro_cer=error/chars,errors=error,characters=chars,dense_parent=longest,dense_errors=specific[0],dense_characters=specific[1],error_share=specific[0]/error if error else 0,character_share=specific[1]/chars)
clean=[r for r in rows if r['candidate']=='C4' and r['dataset']=='positive' and r['kind']=='clean' and r['parent_id']==longest]
output['clean_dense_diagnostic']=dict(parent_id=longest,errors=sum(sum(int(r[k]) for k in ['substitutions','deletions','insertions']) for r in clean),all_nonspace_sequences_equal=all(r['truth'].replace(' ','')==r['prediction'].replace(' ','') for r in clean),main_metrics_unchanged=True,notes='Removing spaces is a diagnostic equality check only; actual targets/CER remain unchanged.')
write_json(root/'long_sequence_diagnostic.json',output);print(json.dumps(output['clean_dense_diagnostic'],indent=2))
