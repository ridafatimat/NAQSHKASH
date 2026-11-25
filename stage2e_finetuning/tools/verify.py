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
 result=dict(checked=len(expected),changed=changed,new_files_outside_stage2e=newoutside,software_files_checked=len(sourcefiles),software_changed=sourc