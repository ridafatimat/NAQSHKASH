"""Verify original model, all historical files and all frozen Stage2E inputs/code."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from context import *
def main():
 expected=json.loads((E/'configs/protected_before.json').read_text())
 changed=[p for p,h in expected.items() if not Path(p).is_file() or digest(p)!=h]
 newoutside=[str(p) for p in PROJECT.rglob('*') if p.is_file() and not p.is_relative_to(E) and not {'.git','__pycache__','.pytest_cache'}.intersection(p.parts) and str(p) not in expected]
 sourcefiles=json.loads((E/'configs/software_lock.json').read_text())
 sourcechanged=[p for p,h in sourcefiles.items() if digest(p)!=h]
 locked=verify();dc.old.verify_lock(code=True)
 result=dict(checked=len(expected),changed=changed,new_files_outside_stage2e=newoutside,software_files_checked=len(sourcefiles),software_changed=sourcechanged,stage2e_locked_inputs=locked,original_checkpoint_sha256=digest(CHECKPOINT),original_unchanged=digest(CHECKPOINT)==EXPECTED,passed=not changed and not sourcechanged and not newoutside and digest(CHECKPOINT)==EXPECTED,preexisting_stage2d_archive_protected=str(PROJECT/'stage2/stage2d_experiments.zip') in expected)
 selection=json.loads((E/'reports/selected_checkpoint.json').read_text()) if (E/'reports/selected_checkpoint.json').exists() else None
 if selection:
  result['selected_checkpoint_sha256']=digest(selection['selected_checkpoint'])
  result['selected_checkpoint_unchanged']=result['selected_checkpoint_sha256']==selection['sha256']
  result['passed']=result['passed'] and result['selected_checkpoint_unchanged']
 dump('reports/protected_file_hash_check.json',result)
 print(json.dumps(result,indent=2))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()

