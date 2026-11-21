import csv,hashlib,importlib.util,json,sys,unittest,zipfile
from pathlib import Path
from html.parser import HTMLParser
import cv2,numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('screen',ROOT/'screen.py');screen=importlib.util.module_from_spec(spec);spec.loader.exec_module(screen)
class ScreeningTests(unittest.TestCase):
    def test_objective_blank_rule(self):
        self.assertTrue(screen.objective(np.full((60,80,3),127,np.uint8)))
        a=np.full((60,80,3),255,np.uint8);a[30,40]=0
        self.assertFalse(screen.objective(a),'Sparse handwriting must not be excluded as blank')
        with self.assertRaises(Exception):screen.decode(b'not an image')
    def test_known_crop_foreground_coverage(self):
        g=np.full((200,300),255,np.uint8);cv2.rectangle(g,(10,50),(35,150),0,-1);cv2.rectangle(g,(200,50),(220,150),0,-1)
        lost,_=screen.coverage(g,np.eye(3),(200,300));self.assertEqual(lost,0)
        H=np.eye(3);H[0,2]=-60;cropped,_=screen.coverage(g,H,(200,300))
        self.assertGreater(cropped,.1,'Known removed foreground must be detectable')
    def test_registration_and_unresolved_case(self):
        rng=np.random.default_rng(16);g=rng.integers(0,256,(400,500),dtype=np.uint8)
        c=screen.compare(g,g,screen.features(g),screen.features(g));self.assertEqual(c['registration'],'REGISTERED_PROXY');self.assertLess(c['reference_foreground_outside_fraction'],.01)
        blank=np.full_like(g,255);c=screen.compare(g,blank,screen.features(g),screen.features(blank));self.assertEqual(c['registration'],'UNRESOLVED')
    def test_complete_inventory_and_preserved_split(self):
        orig=screen.readcsv(screen.AUDIT/'manifest.csv');rows=screen.readcsv(ROOT/'automated_quality_report.csv');self.assertEqual(len(rows),920)
        before={r['image_path']:r for r in orig}
        for r in rows:
            self.assertEqual(r['split'],before[r['image_path']]['split']);self.assertEqual(r['source_id'],before[r['image_path']]['source_id'])
            self.assertEqual(r['sha256'],before[r['image_path']]['sha256']);self.assertNotEqual(r['decision'],'ACCEPT')
            if r['decision']=='OBJECTIVE_EXCLUDE':self.assertTrue('SPATIALLY_CONSTANT_BLANK' in r['automatic_flags'] or 'UNREADABLE_OR_CORRUPT' in r['automatic_flags'])
        summary=json.loads((ROOT/'readiness_summary.json').read_text());self.assertEqual(sum(summary['quality_results'].values()),920)
        self.assertEqual(len(screen.readcsv(ROOT/'flagged_images.csv')),sum(bool(r['automatic_flags']) for r in rows))
    def test_queue_and_dashboard_isolation(self):
        q=screen.readcsv(ROOT/'prioritized_review_queue.csv');self.assertEqual(sum(r['review_type']=='SOURCE_LABEL_AND_PROVENANCE' for r in q),30)
        self.assertTrue(all(r['review_scope']=='ISOLATED_TEST' for r in q if r['split']=='test'))
        dev=(ROOT/'review_dashboard.html').read_text(encoding='utf-8');test=(ROOT/'isolated_test_review.html').read_text(encoding='utf-8')
        class Images(HTMLParser):
            def __init__(self):super().__init__();self.paths=[]
            def handle_starttag(self,t,attrs):
                if t=='img':self.paths.append(dict(attrs)['src'])
        parser=Images();parser.feed(dev);self.assertTrue(parser.paths)
        sources=screen.readcsv(screen.AUDIT/'source_split.csv')
        for s in sources:
            if s['split']=='test':self.assertNotIn(s['source_id'],dev);self.assertIn(s['source_id'],test)
        self.assertIn('<pre>',dev);self.assertIn('Synthetic annotation reference',dev)
    def test_input_integrity(self):
        for p,h in json.loads((ROOT/'input_integrity.json').read_text()).items():
            if '!' in p:
                archive,n=p.split('!',1)
                with zipfile.ZipFile(archive) as z:b=z.read(n)
            else:b=Path(p).read_bytes()
            self.assertEqual(hashlib.sha256(b).hexdigest(),h,p)
if __name__=='__main__':unittest.main(verbosity=2)
