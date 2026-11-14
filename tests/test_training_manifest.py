"""Run: python -m unittest discover -s tests -p test_training_manifest.py"""
import ast
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1]/'data'/'build_training_manifest.py'
if not SCRIPT.exists():
    SCRIPT = Path(__file__).with_name('build_training_manifest.py')
spec = importlib.util.spec_from_file_location('manifest', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


class ManifestTests(unittest.TestCase):
    def test_scalar_semantics(self):
        self.assertEqual(m.resized_width(1, 128), 1)
        self.assertEqual(m.resized_width(5, 128), 2)  # Python tie-to-even
        self.assertEqual(m.ctc_required('AA BB'), 7)
        self.assertEqual(m.ctc_required(''), 0)

    def test_real_source_scalar_logic(self):
        # Extract only pure functions into AST, avoiding torch and all module
        # top-level code. This detects divergence from the actual project.
        project = SCRIPT.parents[1]
        if not (project/'models'/'crnn.py').exists():
            self.skipTest('Project source available after installation')
        class FakeTorch:
            @staticmethod
            def clamp(x, min): return max(x, min)
        tree = ast.parse((project/'models'/'crnn.py').read_text(encoding='utf-8-sig'))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'output_lengths')
        node.decorator_list = []
        env = {'torch': FakeTorch}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<extracted output_lengths>', 'exec'), env)
        for w in range(1, 5000):
            self.assertEqual(env['output_lengths'](w), max(1, w//2//2))
        tree = ast.parse((project/'data'/'collate.py').read_text(encoding='utf-8-sig'))
        resize = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'resize_preserve_aspect_ratio')
        assignment = next(n for n in resize.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'new_width' for t in n.targets))
        for w,h in [(1,128),(5,128),(101,35),(255,25),(501,64),(123,37)]:
            env = dict(width=w, scale=64/float(h))
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<extracted resize>', 'exec'), env)
            self.assertEqual(env['new_width'], m.resized_width(w,h))

    def test_fixture_inventory_roundtrip_and_determinism(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            chars = list(' ABC,"') + [chr(i) for i in range(1000,1059)]
            (root/'validation_report.json').write_text(json.dumps({'splits':{'train':{'vocabulary':chars}}}), encoding='utf-8')
            for split in m.SPLITS:
                base = root/split
                (base/'images').mkdir(parents=True)
                (base/'labels').mkdir()
                Image.new('L', (20,10)).save(base/'images'/'a.png')
                (base/'labels'/'a.txt').write_text('  AA, "B  B"\n',encoding='utf-8')
                row = dict(tier_id='a', parent_sample_id=split+'_page',row_id='r',row_index='0',tier_index='0',tier_name='upper',image_path='images/a.png',label_path='labels/a.txt',label='  AA, "B  B"',label_length='12',image_width='20',image_height='10')
                row['label_length'] = str(len(row['label']))
                write_csv(base/'metadata.csv', list(row), [row])
                layout = dict(row_id='r',parent_sample_id=split+'_page',row_index='0',upper_present='1',count_present='0',lower_present='0')
                write_csv(base/'layout_manifest.csv',list(layout),[layout])
                (base/'completed_pages.txt').write_text(split+'_page\nfailed\n',encoding='utf-8')
                write_csv(base/'tier_build_failures.csv',['sample_id','reason'],[dict(sample_id='failed',reason='ROW_COUNT_MISMATCH')])
            (root/'test'/'images'/'orphan.png').write_bytes(b'x')
            (root/'test'/'labels'/'orphan.txt').write_text('x', encoding='utf-8')
            result = m.build(root,root/'manifests',workers=2)
            original = (root/'manifests'/'full_tier_manifest.csv').read_bytes()
            self.assertEqual(result['total_rows'],3)
            self.assertEqual(result['issue_counts'],{})
            self.assertEqual(result['splits']['test']['images_without_metadata'],['test/images/orphan.png'])
            self.assertEqual(result['splits']['test']['completed_without_tiers'],['failed'])
            self.assertTrue(result['roundtrip']['passed'])
            self.assertIn('leading_spaces',result['roundtrip']['categories'])
            m.build(root,root/'manifests',workers=1)
            self.assertEqual(original,(root/'manifests'/'full_tier_manifest.csv').read_bytes())
            (root/'test'/'images'/'a.png').write_bytes(b'invalid')
            (root/'validation'/'labels'/'a.txt').unlink()
            result = m.build(root,root/'manifests',workers=2)
            self.assertEqual(result['issue_counts']['image_status:unreadable_header'],1)
            self.assertEqual(result['issue_counts']['label_status:missing'],1)


if __name__ == '__main__':
    unittest.main()
