from pathlib import Path
import json,hashlib,ast
WORK=Path(__file__).resolve().parent
ROOT=Path('C:/NAQSHKASH/stage2/stage2g_handwriting_adaptation_v5b')
source=ROOT/'source';source.mkdir(exist_ok=True)
names=['v5b_engine.py','v5b_prepare.py','v5b_accept_qa.py','v5b_generated.py','v5b_local_folds.py','v5b_local_final.py','v5b_verify_smoke.py','v5b_finish.py','v5b_bundle.py']
for name in names:
 data=(WORK/name).read_bytes();ast.parse(data.decode('utf-8'));(source/name).write_bytes(data)
text=(WORK/'v5b_generated.py').read_text(encoding='utf-8')
text=text.replace('import v5b_engine as e',"sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_run'))\nimport v5b_engine as e",1)
ast.parse(text);(ROOT/'generated_only/run_generated_only.py').write_text(text,encoding='utf-8')
(ROOT/'source_manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()},indent=2),encoding='utf-8')
print('Source bundle syntax checked and saved; generated-only resume entry point is ready')
