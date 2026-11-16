"""Reproduce the frozen algorithm with output-path relocation only."""
import argparse,hashlib,json,re
from dcore import *
def relocated_source(name):
 if not re.fullmatch(r'[A-Za-z0-9_-]+',name) or name=='evaluation_v1':raise ValueError('Choose a distinct simple result namespace')
 original=ROOT/'evaluate_d1.py';source=original.read_text(encoding='utf-8-sig')
 lock=json.loads((ROOT/'configs/evaluation_software_lock.json').read_text());assert old.digest(original)==lock[str(original)]
 replacements={
  "'reports/evaluation_v1'":f"'reports/{name}'",
  "'reports/evaluation_v1/":f"'reports/{name}/",
  "'derived_crops'":f"'derived_crops_{name}'",
  "'reports/pre_evaluation_hash_check.json'":f"'reports/pre_evaluation_hash_check_{name}.json'"}
 counts={}
 for a,b in replacements.items():
  counts[a]=source.count(a);assert counts[a]>0
  source=source.replace(a,b)
 return source,dict(original_sha256=old.digest(original),relocated_source_sha256=hashlib.sha256(source.encode()).hexdigest(),output=str(ROOT/'reports'/name),crop_output=str(ROOT/('derived_crops_'+name)),only_output_path_replacements=replacements,replacement_counts=counts,frozen_candidate_unchanged=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
 source,plan=relocated_source(a.name)
 if Path(plan['output']).exists() or Path(plan['crop_output']).exists():raise RuntimeError('Namespace exists; preserve old outputs')
 print(json.dumps(plan,indent=2))
 if not a.plan_only:
  dump('reports/reproduction_'+a.name+'_source.json',plan)
  exec(compile(source,str(ROOT/'evaluate_d1.py'),'exec'),dict(__name__='__main__',__file__=str(ROOT/'evaluate_d1.py')))

