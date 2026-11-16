"""Verify frozen sources and every historical file; allow only the new D module."""
from dcore import *
import argparse
def verify():
 expected=json.loads((ROOT/'configs/protected_before.json').read_text())
 result=check_history(expected)
 additions=[str(p) for p in old.PROJECT.rglob('*') if p.is_file() and not p.is_relative_to(ROOT) and not {'.git','__pycache__','.pytest_cache'}.intersection(p.parts) and str(p) not in expected]
 result['unexpected_files_outside_stage2d']=additions
 result['passed']=result['passed'] and not additions
 inventory=json.loads((C/'reports/artifact_inventory.json').read_text())
 bad=[p['path'] for p in inventory['files'] if old.digest(p['path'])!=p['sha256']]
 software=json.loads((ROOT/'configs/evaluation_software_lock.json').read_text())
 codebad=[p for p,h in software.items() if old.digest(p)!=h]
 old.verify_lock(code=True)
 result.update(stage2c_artifacts_checked=len(inventory['files']),stage2c_changed=bad,stage2d_locked_software_checked=len(software),stage2d_software_changed=codebad,checkpoint_sha256=old.digest(old.CHECKPOINT),original_source_locks_passed=True)
 result['passed']=result['passed'] and not bad and not codebad
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--record',action='store_true');a=p.parse_args();r=verify()
 if a.record:dump('reports/protected_file_hash_check.json',r)
 print(json.dumps(r,indent=2))
 if not r['passed']:raise SystemExit(1)

