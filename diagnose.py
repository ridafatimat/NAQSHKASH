"""Reproduce saved counts, compare source parents, and audit role search."""
from dcore import *
from collections import Counter,defaultdict
import math,random
def exhaustive(bands,features,scale):
 pitch=scale+max(6,scale*.25)
 def visit(i):
  if i==len(bands):return [(0.,[])]
  options=[]
  for roles in [['count'],['upper','count'],['count','lower'],['upper','count','lower']]:
   size=len(roles)
   if i+size>len(bands):continue
   score=-1.4
   if i:
    sep=bands[i][0]-bands[i-1][1]
    if sep<scale*.65:continue
    score+=float(np.clip((sep-scale*.55)/scale,-1.,1.))
   for j,role in enumerate(roles):
    prob=features[i+j]['prob'];score+=math.log(prob if role=='count' else 1-prob)
    if j:
     dist=(sum(bands[i+j])-sum(bands[i+j-1]))/2
     if dist>1.8*pitch:break
     score-=.35*((dist-pitch)/pitch)**2
   else:
    options.extend((score+s,[(i,roles)]+p) for s,p in visit(i+size))
  return options
 return sorted(visit(0),key=lambda v:v[0],reverse=True)[:2]
def dp_reference(bands,features,scale):
 # Literal Stage 2C search, separate from exhaustive recursion.
 pitch=scale+max(6,scale*.25);dp={0:[(0.,[])]}
 for i in range(len(bands)):
  if i not in dp:continue
  for roles in [['count'],['upper','count'],['count','lower'],['upper','count','lower']]:
   size=len(roles)
   if i+size>len(bands):continue
   score=-1.4
   if i:
    sep=bands[i][0]-bands[i-1][1]
    if sep<scale*.65:continue
    score+=float(np.clip((sep-scale*.55)/scale,-1.,1.))
   allowed=True
   for j,role in enumerate(roles):
    prob=features[i+j]['prob'];score+=math.log(prob if role=='count' else 1-prob)
    if j:
     dist=(sum(bands[i+j])+0-sum(bands[i+j-1]))/2
     if dist>1.8*pitch:allowed=False;break
     score-=.35*((dist-pitch)/pitch)**2
   if not allowed:continue
   for score0,path in dp[i]:dp.setdefault(i+size,[]).append((score0+score,path+[(i,roles)]))
   dp[i+size]=sorted(dp[i+size],key=lambda v:v[0],reverse=True)[:2]
 return dp.get(len(bands),[])
def main():
 for name in ['reports','configs','galleries','manifests','tests']: (ROOT/name).mkdir(exist_ok=True)
 if not (ROOT/'configs/protected_before.json').exists():dump('configs/protected_before.json',snapshot())
 # Confirm every inventoried Stage 2C artifact, not just its runtime lock.
 inventory=json.loads((C/'reports/artifact_inventory.json').read_text())
 assert all(old.digest(v['path'])==v['sha256'] for v in inventory['files'])
 old.verify_lock(code=True)
 rows=read(C/'reports/experiments_v1/pages.csv')
 predictions=read(C/'reports/experiments_v1/predictions.csv')
 summaries=[]
 for s in read(C/'reports/experiments_v1/summary.csv'):
  g=[r for r in rows if all(r[k]==s[k] for k in ['partition','dataset','kind','candidate'])]
  assert len(g)==int(s['pages'])
  assert sum(r['correct_segmentation']=='True' for r in g)==int(s['correct_pages'])
  assert sum(r['proposal']=='True' and r['correct_segmentation']!='True' for r in g)==int(s['wrong_acceptances'])
  ps=[r for r in predictions if all(r[k]==s[k] for k in ['partition','dataset','kind','candidate'])]
  edits=sum(sum(int(r[k]) for k in ['substitutions','deletions','insertions']) for r in ps);chars=sum(len(r['truth']) for r in ps)
  if chars:assert abs(edits/chars-float(s['cer']))<1e-12
  summaries.append(s)
 by={(r['candidate'],*key(r)):r for r in rows};comparisons=[];categories=[]
 for p in pages():
  a=by[('C4',*key(p))];b=by[('C5',*key(p))]
  status=lambda r:'correct' if r['correct_segmentation']=='True' else 'wrong' if r['proposal']=='True' else 'rejected'
  comparisons.append(dict(**{k:p[k] for k in ['partition','dataset','parent_id','case_id','kind','augmentation']},C4=status(a),C5=status(b),C4_reason=a['reason'],C5_reason=b['reason'],C4_margin=a['role_margin'],C5_margin=b['role_margin']))
 for cand in ['C4','C5']:
  groups=defaultdict(list)
  for r in rows:
   if r['candidate']==cand and r['reason']:groups[(r['partition'],r['dataset'],r['kind'],r['reason'])].append(r)
  for (part,d,k,reason),rs in sorted(groups.items()):
   categories.append(dict(candidate=cand,partition=part,dataset=d,kind=k,category=reason,pages=len(rs),parents=len({r['parent_id'] for r in rs}),representative=rs[0]['case_id']))
 csvout('reports/c4_c5_comparison.csv',comparisons);csvout('reports/failure_categories.csv',categories)
 rng=random.Random(42);checked=0
 for n in range(1,10):
  for trial in range(150):
   bands=[];y=0;scale=rng.uniform(8,50)
   for i in range(n):
    h=rng.randint(2,40);bands.append([y,y+h]);y+=h+rng.randint(1,65)
   f=[dict(prob=rng.uniform(.03,.97)) for _ in bands]
   a=exhaustive(bands,f,scale);b=dp_reference(bands,f,scale)
   assert len(a)==len(b) and all(abs(x[0]-y[0])<1e-10 for x,y in zip(a,b)),(a,b)
   checked+=1
 dump('reports/search_audit.json',dict(random_sequences=checked,maximum_bands=9,top_two_scores_match=True,verified_search_defects=0,notes='Exhaustive full paths versus literal Stage 2C DP; tie path order is not a scoring defect. Morphology and priors can still be wrong.'))
 dump('reports/evidence_verification.json',dict(inventory_files_verified=len(inventory['files']),summary_groups_verified=len(summaries),saved_page_outcomes=len(rows),saved_predictions=len(predictions),source_software_lock_verified=True))
 counts=Counter((r['partition'],r['dataset'],r['kind'],r['C4'],r['C5']) for r in comparisons)
 for k,v in counts.items():
  if k[0]=='validation' and k[2]=='scanned':print(k,v)
 print('Search exhaustive tests:',checked,flush=True)
if __name__=='__main__':main()

