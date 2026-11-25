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
    spaces.append(dict(candidate=name,id=r['id'],parent_id=r['parent_id'],role=r['role'],truth_spaces=r['truth'].count(' '),predicted_spaces=r['prediction'].count(' '),deleted_spaces=sum(op=='deletion' and a==' ' for op,a,b in ops),inserted_spaces=sum(op=='insertion' and b==' ' for op,a,b in ops),space_substitutions=sum(op=='substitution' and (a==' ' or b==' ') for op,a,b in ops),truth_length=len(r['truth']),prediction_length=len(r['prediction'])))
    allops.extend(ops)
  group=[r for r in spaces if r['candidate']==name]
  summaries.append(dict(candidate=name,positive_deleted_spaces=sum(r['deleted_spaces'] for r in group),positive_inserted_spaces=sum(r['inserted_spaces'] for r in group),positive_space_substitutions=sum(r['space_substitutions'] for r in group),positive_nonspace_deletions=sum(op=='deletion' and a!=' ' for op,a,b in allops),positive_symbol_count_substitutions=sum(op=='substitution' and a!=' ' and b!=' ' for op,a,b in allops)))
 write('metrics/character_confusions.csv',[dict(candidate=k[0],cohort=k[1],role=k[2],operation=k[3],expected=k[4],predicted=k[5],count=n) for k,n in sorted(confusion.items())])
 write('metrics/space_error_details.csv',spaces);dump('metrics/space_error_summary.json',summaries)
 path=E/'reports/error_analysis.md';text=path.read_text(encoding='utf-8')
 text+='\n## Measured character/space attribution\n\n'+json.dumps(summaries,indent=2)+'\n\nEdit traces are deterministic minimum-distance alignments (substitution, deletion, insertion tie preference). Character attribution in ambiguous alignments is diagnostic; official CER/SDI remains unchanged. Count-tier errors are repetition-code recognition errors; upper/lower-tier errors are symbol recognition errors. Exact confusion pairs and distortion/length/role breakdowns are saved, not inferred from image appearance alone.\n\nTop remaining positive substitutions for selected checkpoint:\n\n'
 top=[(k,n) for k,n in confusion.items() if k[0]==selected and k[1]=='positive_development' and k[3]=='substitution']
 for k,n in sorted(top,key=lambda v:(-v[1],v[0]))[:15]:text+=f"- tier {k[2]}: expected {k[4]!r}, predicted {k[5]!r}, count {n}\n"
 path.write_text(text,encoding='utf-8')
 print(json.dumps(summaries),flush=True)
if __name__=='__main__':main()

