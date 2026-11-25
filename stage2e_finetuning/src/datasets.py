"""Validated frozen tier inputs; original normalization and variable widths."""
from context import *
from collections import defaultdict
class FrozenTiers(torch.utils.data.Dataset):
 def __init__(self,rows):self.rows=read(rows) if isinstance(rows,(Path,str)) else list(rows)
 def __len__(self):return len(self.rows)
 def __getitem__(self,index):
  r=self.rows[index];im=cv2.imread(r['path'],0)
  if im is None:raise ValueError('Unreadable frozen crop '+r['path'])
  assert max(1,round(im.shape[1]*64/im.shape[0]))==int(r['resized_width'])
  return dict(image=tensor(im),label_text=r['truth'],sample_id=r['id'],tier_name=r['role'])
def order(rows,seed_value):
 # Metadata-only role/length stratification, no model or evaluation outcome.
 rng=random.Random(seed_value);groups=defaultdict(list)
 for i,r in enumerate(rows):groups[(r['role'],min(3,len(r['truth'])//16))].append(i)
 for g in groups.values():rng.shuffle(g)
 result=[]
 while any(groups.values()):
  for k,g in sorted(groups.items()):
   if g:result.append(g.pop())
 return result
def schedule(positive,replay,steps,mixture,seed_value):
 p=order(positive,seed_value);r=order(replay,seed_value+1) if replay else [];pi=ri=0;result=[]
 for step in range(steps):
  batch=[]
  for j in range(4):
   which='replay' if mixture and j%2 else 'positive'
   if which=='positive':batch.append((which,p[pi%len(p)]));pi+=1
   else:batch.append((which,r[ri%len(r)]));ri+=1
  result.append(batch)
 return result

