"""Deterministic edit traces for space, count-code and symbol error diagnosis."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from context import *
from collections import Counter,defaultdict
def trace(a,b):
 table=[list(range(len(b)+1))]
 for i,x in enumerate(a,1):
  row=[i]
  for j,y in enumerate(b,1):row.append(min(table[i-1][j-1]+(x!=y),table[i-1][j]+1,row[-1]+1))
  table.append(row)
 i,j=len(a),len(b);ops=[]
 while i or j:
  if i and j and a[i-1]==b[j-1] and table[i][j]==table[i-1][j-1]:i-=1;j-=1
  elif i and j and table[i][j]==table[i-1][j-1]+1:ops.append(('substitution',a[i-1],b[j-1]));i-=1;j-=1
  elif i and table[i][j]==table[i-1][j]+1:ops.append(('deletion',a[i-1],''));i-=1
  else:ops.append(('insertion','',b[j-1]));j-=1
 return ops[::-1]
def main():
 selection=json.loads((E/'reports/selected_checkpoint.json').read_text());selected={'E0':'baseline','E1':'e1','E2':'e2'}[selection['selected']]
 confusion=Counter();spaces=[];summaries=[]
 for name in ['baseline','e1','e2']:
  rows=read(E/'metrics'/f'{name}_per_sample.csv');allops=[]
  for r in rows:
   ops=trace(r['truth'],r['prediction']) if r['exact']!='True' else []
   assert len(ops)==sum(int(r[k]) for k in ['substitutions','deletions','insertions'])
   for op,a,b in ops:confusion[(name,r['cohort'],r['role'],op,a,b)]+=1
   if r['cohort']=='positive_development':
    spaces.append(dict(candidate=name,id=r['id'],parent_id=r['parent_id'],role=r['role'],truth_spaces=r['truth'].count(' '),predicted_spaces=r['prediction'].count(' '),deleted_spaces=sum(op=='deletion' and a==' ' for op,a,b in ops),inserted_spaces=sum(op=='insertion' and b==' ' for op,a,b in ops),space_substitutions=sum(op=='substitution' and (a==' ' or b==' ') for op,a,b in ops),truth_length=len(r['truth']),prediction_length=l