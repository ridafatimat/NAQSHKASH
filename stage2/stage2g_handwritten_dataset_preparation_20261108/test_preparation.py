"""Lightweight preparation checks; no models or protected-test data."""
import csv, hashlib, json, unittest, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
class PreparationTests(unittest.TestCase):
    def test_inventory_and_paths(self):
        rows=read('manifest.csv');self.assertEqual(len(rows),920)
        self.assertEqual(len({r['source_id'] for r in rows}),30)
        self.assertEqual(sum(r['augmentation_type']=='original_photo' for r in rows),20)
        for r in rows:
            for k in ['image_path','transcription_path','annotation_path']:self.assertTrue(Path(r[k]).is_file(),r[k])
            self.assertEqual(hashlib.sha256(Path(r['image_path']).read_bytes()).hexdigest(),r['sha256'])
            self.assertFalse(r['decode_error'])
    def test_source_split_no_leakage(self):
        groups={}
        for r in read('manifest.csv'):groups.setdefault(r['source_id'],set()).add(r['split'])
        self.assertTrue(all(len(v)==1 for v in groups.values()))
        sources=read('source_split.csv');self.assertEqual(len(sources),30)
        known={r['split'] for r in sources if r['metadata_capture_writer']=='writer_01'}
        self.assertLessEqual(len(known),1)
        self.assertEqual({r['split'] for r in sources},{'train','validation','test'})
    def test_annotations_and_acceptance_gate(self):
        self.assertEqual(json.loads((ROOT/'annotation_issues.json').read_text()),[])
        for s in read('source_split.csv'):
            self.assertEqual(s['annotation_valid'],'True');self.assertGreater(int(s['image_copies']),0)
        self.assertEqual(read('accepted_manifest.csv'),[])
        self.assertTrue(all(r['status'] in ['excluded','manual_review'] for r in read('manifest.csv')))
        self.assertEqual(len(read('review_queue.csv')),920)
        self.assertEqual(len(list((ROOT/'review').glob('*.jpg'))),30)
    def test_original_inputs_unchanged(self):
        archives={}
        try:
            for row in read('input_hashes.csv'):
                origin=row['origin']
                if '!' in origin:
                    archive,member=origin.split('!',1)
                    if archive not in archives:archives[archive]=zipfile.ZipFile(archive)
                    b=archives[archive].read(member)
                else:b=Path(origin).read_bytes()
                self.assertEqual(hashlib.sha256(b).hexdigest(),row['sha256'],origin)
        finally:
            for z in archives.values():z.close()
if __name__=='__main__':unittest.main(verbosity=2)
