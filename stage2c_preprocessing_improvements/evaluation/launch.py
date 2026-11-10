"""Reproduction entry point: fail closed before any model or page is evaluated."""
import sys,runpy
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from core import load_local
verifier=load_local('stage2c_launch_verifier','protocol/verify.py')
print('Pre-inference source/history verification:',verifier.verify_history(),flush=True)
runpy.run_path(str(root/'evaluation/run.py'),run_name='__main__')
