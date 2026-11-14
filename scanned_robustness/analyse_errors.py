"""Deterministic edit attribution on saved predictions, no further inference."""
import json
from collections import Counter,defaultdict
from common import HERE,read_csv,edit_counts


def operations(truth,prediction):
 n,m=len(truth),len(prediction)
 cost=[[0]*(m+1) for _ in range(n+1)];move={}
 for i in range(1,n+1):cost[i][0]=i;move[i,0]='D'
 for j in range(1,m+1):cost[0][j]=j;move[0,j]='I'
 for i in range(1,n+1):
  for j in range(1,m+1):
   if truth[i-1]==prediction[j-1]:cost[i][j]=cost[i-1][j-1];move[i,j]='E'
   else:
    value,action=min([(cost[i-1][j-1]+1,'S'),(cost[i-1][j]+1,'D'),(cost[i][j-1]+1,'I')],key=lambda p:p[0])
    cost[i][j]=value;move[i,j]=action
 out=[];i=n;j=m
 while i or j:
  action=move[i,j]
  if action=='E':i-=1;j-=1
  elif action=='S':out.append(('S',truth[i-1],prediction[j-1]));i-=1;j-=1
  elif action=='D':out.append(('D',truth[i-1],''));i-=1
  else:out.append(('I','',prediction[j-1]));j-=1
 return out


def main():
 groups=defaultdict(lambda:dict(substitutions=Counter(),deletions=Counter(),insertions=Counter(),error_sequences=0))
 for row in read_csv(HERE/'reports/evaluation_v1/recognition_errors.csv'):
  ops=operations(row['truth'],row['prediction']);counts=Counter(op for op,a,b in ops)
  assert counts['S']==int(row['substitutions']) and counts['D']==int(row['deletions']) and counts['I']==int(row['insertions'])
  group=groups[row['cohort']+'/'+row['mode']];group['error_sequences']+=1
  for op,a,b in ops:
   if op=='S':group['substitutions'][repr(a)+' -> '+repr(b)]+=1
   elif op=='D':group['deletions'][repr(a)]+=1
   else:group['insertions'][repr(b)]+=1
 output={key:{k:dict(v.most_common()) if isinstance(v,Counter) else v for k,v in value.items()} for key,value in groups.items()}
 (HERE/'reports/evaluation_v1/edit_attribution.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
 print(json.dumps(output,indent=2))
 metrics=json.loads((HERE/'reports/evaluation_v1/metrics.json').read_text())
 for c in ('positive_frozen_test','general_unassigned_exploratory'):
  print('Actual recipe ranking',c)
  rows=metrics['groups'][c]['assisted']['per_augmentation']
  for name,item in sorted(rows.items(),key=lambda pair:pair[1]['metrics']['cer'] or 0,reverse=True):
   print(name,'CER',item['metrics']['cer'],'processed',item['pages_processed'],'/',item['pages_attempted'])

if __name__=='__main__':main()
