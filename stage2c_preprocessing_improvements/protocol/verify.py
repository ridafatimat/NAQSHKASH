"""Read-only verification of frozen inputs, code, and protected historical files."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import *

def verify_history():
 lock=verify_lock(code=True)
 expected=json.loads((HERE/'protocol/protected_before.json').read_text())
 for relative,value in expected.items():
  if digest(PROJECT/relative)!=value:raise ValueError('Protected project file changed: '+relative)
 return dict(locked_inputs=len(lock['locked_files']),protected_historical_files=len(expected),software_files=len(json.loads((HERE/'protocol/software_lock.json').read_text())))

if __name__=='__main__':print(json.dumps(verify_history(),indent=2))
