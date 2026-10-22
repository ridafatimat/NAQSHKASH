import csv
import hashlib
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import random
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1]/'data'/'build_stratified_subset.py'
if not SCRIPT.exists():
    SCRIPT = Path(__file__).with_name('build_stratified_subset.py')
spec = importlib.util.spec_from_file_location('stratified',SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

FIELDS = ['split','tier_id','image_path','label_path','raw_label','training_label','tier_name',
          'image_width','image_height','resized_width','label_length','parent_image_id','row_id',
          'row_index','tier_index','leading_offset','raw_label_length','ctc_required_timesteps',
          'crnn_output_length','ctc_feasible','unknown_characters','image_status','label_status']


class StratificationTests(unittest.TestCase):
    def test_duplicate_quantile_partitions(self):
        spec = m.bucket_spec([1]*90+[2]*5+[9]*5,5)
        self.assertEqual(spec['actual_buckets'],2)
        self.assertEqual(spec['boundaries'],[1.0])
        self.assertEqual(m.bucket(1,spec,'label'),'label_q1')
        self.assertEqual(m.bucket(2,spec,'label'),'label_q2')
        self.assertEqual(m.bucket_spec([7]*30,5)['actual_buckets'],1)

    def test_allocations_exact_and_bounded(self):
        rng = random.Random(7)
        for _ in range(100):
            sizes = {str(i):rng.randint(1,200) for i in range(10)}
            target = rng.randint(10,sum(sizes.values()))
            result,floors = m.allocate(sizes,target)
            self.assertEqual(sum(result.values()),target)
            self.assertTrue(all(floors[k]<=result[k]<=sizes[k] for k in sizes))
            self.assertEqual(result,m.allocate(sizes,target)[0])
        with self.assertRaises(ValueError): m.allocate({'a':4,'b':4},1)
        self.assertEqual(m.allocate({'a':1,'b':90},1000)[0],{'a':1,'b':90})

    def test_parent_rounds_and_global_preference(self):
        rows = [dict(parent_image_id=p,tier_id=str(i)) for i,p in enumerate(['a','a','a','b','b','c','a','d'])]
        selected,audit = m.parent_sample(rows,{'x':list(range(6)),'y':[6,7]},{'x':4,'y':1},42,'train')
        self.assertEqual(len(selected),5)
        self.assertEqual(audit['x']['initial_distinct_parents'],3)
        self.assertEqual(audit['y']['initial_distinct_parents'],1)
        self.assertEqual(len({rows[i]['parent_image_id'] for i in selected}),4)
        counts = {}
        for i in selected & set(range(6)):
            p = rows[i]['parent_image_id']
            counts[p] = counts.get(p,0)+1
        self.assertEqual(sorted(counts.values()),[1,1,2])

    def test_rare_repair_prefers_new_parent_and_does_not_overrepair(self):
        rows = [dict(tier_id=str(i),parent_image_id=p,stratum_id='s',training_label=t)
                for i,(p,t) in enumerate([('a','A'),('a','XXX'),('b','XXX'),('c','X')])]
        chosen = {0}
        report = m.repair(rows,chosen,{'s':4},['A','X'],3,42,'train')
        self.assertEqual(chosen,{0,2})
        self.assertEqual(report['repair_samples_added'],1)
        self.assertEqual(report['characters_repaired'],['X'])
        self.assertEqual(report['characters']['X']['subset_count'],3)
        report = m.repair(rows,chosen,{'s':4},['A','X'],3,42,'train')
        self.assertEqual(report['repair_samples_added'],0)

    def test_invalid_paths(self):
        self.assertTrue(m.valid_path('train/images/a.png','train','images'))
        for path in ('C:/data/a.png','/train/images/a.png','train/images/../a.png',
                     'train\\images\\a.png','validation/images/a.png','train/images//a.png'):
            self.assertFalse(m.valid_path(path,'train','images'))

    def fixture(self,root,leak=False):
        manifests = root/'manifests'
        manifests.mkdir()
        vocab = [chr(i) for i in range(32,97)]
        rows = []
        for split in ('train','validation','test'):
            for folder in ('images','labels'):
                (root/split/folder).mkdir(parents=True)
            for parent in range(4):
                for ti,tier in enumerate(('upper','count','lower')):
                    tid = '{}_{}_{}'.format(split,parent,tier)
                    target = 'A'+''.join(vocab) if parent==0 else 'AA  B,"'
                    raw = '  '+target
                    image_path,label_path = split+'/images/'+tid+'.png',split+'/labels/'+tid+'.txt'
                    (root/image_path).write_bytes(b'header reads are not needed')
                    (root/label_path).write_text(raw,encoding='utf-8')
                    row = dict.fromkeys(FIELDS,'')
                    row.update(split=split,tier_id=tid,image_path=image_path,label_path=label_path,
                               raw_label=raw,training_label=target,tier_name=tier,image_width='2000' if parent==0 else '400',
                               image_height='64',resized_width='2000' if parent==0 else '400',label_length=str(len(target)),
                               parent_image_id=('shared' if leak else split)+'_'+str(parent),row_id=tid+'_row',row_index='0',
                               tier_index=str(ti),leading_offset='2',raw_label_length=str(len(raw)),
                               ctc_required_timesteps=str(len(target)+sum(a==b for a,b in zip(target,target[1:]))),
                               crnn_output_length='500' if parent==0 else '100',ctc_feasible='1',image_status='ok',label_status='ok')
                    rows.append(row)
        manifest = manifests/'full_tier_manifest.csv'
        with manifest.open('w',encoding='utf-8',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=FIELDS,quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(rows)
        summary = dict(vocabulary=vocab,vocabulary_classes_including_blank=66,manifest_sha256=m.sha256(manifest))
        (manifests/'manifest_summary.json').write_text(json.dumps(summary),encoding='utf-8')
        return manifest

    def test_end_to_end_roundtrip_no_mutations_and_determinism(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root = Path(tmp)
            manifest = self.fixture(root)
            before = {p.relative_to(root).as_posix():m.sha256(p) for p in root.rglob('*') if p.is_file()}
            one,two = root/'manifests'/'one',root/'manifests'/'two'
            report = m.build(manifest,one,6,6,min_char_count=3)
            again = m.build(manifest,two,6,6,min_char_count=3)
            self.assertEqual(report,again)
            for p in one.iterdir():
                self.assertEqual(p.read_bytes(),(two/p.name).read_bytes())
            self.assertTrue(all(m.sha256(root/p)==h for p,h in before.items()))
            self.assertTrue(all(x['parent_overlap_count']==0 for x in report['leakage'].values()))
            self.assertTrue(report['integrity']['train']['passed'])
            self.assertTrue(report['outputs']['train']['all_fields_roundtrip_passed'])
            with self.assertRaises(FileExistsError): m.build(manifest,one,6,6,min_char_count=3)
            with self.assertRaises(ValueError): m.build(manifest,root/'train',6,6)
            (root/'train'/'images'/'train_0_upper.png').unlink()
            with self.assertRaises(ValueError): m.build(manifest,root/'manifests'/'missing',12,12,min_char_count=3)
            self.assertFalse((root/'manifests'/'missing').exists())

    def test_leakage_blocks_outputs(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root = Path(tmp)
            manifest = self.fixture(root,leak=True)
            output = root/'manifests'/'blocked'
            with self.assertRaisesRegex(ValueError,'Leakage'):
                m.build(manifest,output,6,6,min_char_count=3)
            self.assertFalse(output.exists())


if __name__=='__main__':
    unittest.main()
