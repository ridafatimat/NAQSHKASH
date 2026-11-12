"""Stage2B integrity, leakage, mode separation and deterministic-accounting tests."""
import collections,hashlib,inspect,json,sys,unittest,zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from common import DATASETS,CHECKPOINT,VOCABULARY,read_csv,sha256,eligible,metrics,edit_counts,load_vocabulary
from modes import blind_tiers,blind_segment,align_after_segmentation,SegmentationFailure

class Stage2BTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.audit=json.loads((HERE/'reports/dataset_audit.json').read_text())
  cls.parents=read_csv(HERE/'manifests/parents.csv')
  cls.all_pages=read_csv(HERE/'manifests/all_variants.csv')
  cls.selected=read_csv(HERE/'manifests/evaluation_pages.csv')

 def test_archive_counts_and_parent_integrity(self):
  self.assertEqual(self.audit['datasets']['general']['variants'],30000)
  self.assertEqual(self.audit['datasets']['positive']['variants'],978)
  counts=collections.Counter((r['dataset'],r['parent_id']) for r in self.all_pages)
  self.assertEqual(len(counts),10326);self.assertEqual(set(counts.values()),{3})
  self.assertEqual(len({r['augmented_sha256'] for r in self.all_pages}),30978)
  self.assertEqual(self.audit['stage1_scan'],dict(json=100000,images=100000))

 def test_external_image_label_and_json_links(self):
  from talim.talim_rules import canonical_row
  for dataset,config in DATASETS.items():
   row=next(r for r in self.selected if r['dataset']==dataset)
   parent=next(r for r in self.parents if r['dataset']==dataset and r['parent_id']==row['parent_id'])
   with zipfile.ZipFile(config['scanned']) as scanned,zipfile.ZipFile(config['clean']) as clean:
    self.assertEqual(hashlib.sha256(scanned.read(row['image_member'])).hexdigest(),row['augmented_sha256'])
    self.assertEqual(hashlib.sha256(clean.read(config['clean_prefix']+parent['source_image'])).hexdigest(),parent['source_sha256'])
    data=json.loads(clean.read(config['clean_prefix']+parent['source_json']))
    for logical in data['logical_rows']:
     self.assertEqual(canonical_row(logical['runs']),(logical['upper'],logical['count_line'],logical['lower']))

 def test_split_leakage_and_frozen_positive_parents(self):
  cohorts=collections.defaultdict(set)
  for row in self.all_pages:cohorts[(row['dataset'],row['parent_id'])].add(row['cohort'])
  self.assertTrue(all(len(c)==1 for c in cohorts.values()))
  self.assertEqual(len(self.selected),450)
  self.assertEqual(sum(r['cohort']=='positive_frozen_test' for r in self.selected),147)
  for row in self.selected:
   self.assertFalse(set(row['stage1_splits'].split(';')).intersection({'train','validation'}))
   if row['dataset']=='positive':self.assertEqual(row['stage2a_split'],'test')
  self.assertEqual(len({r['parent_id'] for r in self.selected if r['dataset']=='positive'}),49)

 def test_semantic_match_is_not_claimed_as_source_identity(self):
  self.assertEqual(eligible(['test'],identity_verified=False),'general_test_structure_match_exploratory')
  self.assertEqual(eligible(['test'],identity_verified=True),'general_stage1_test')
  self.assertEqual(eligible([]),'general_unassigned_exploratory')
  self.assertEqual(eligible(['train','test']),'excluded_seen_or_validation')
  self.assertEqual(eligible(['validation']),'excluded_seen_or_validation')
  self.assertEqual(eligible([],positive_split='train'),'excluded_stage2a_train')
  self.assertEqual(eligible([],positive_split='test'),'positive_frozen_test')
  self.assertNotIn('general_stage1_test',{r['cohort'] for r in self.selected})

 def test_deterministic_exploratory_selection_is_parent_level(self):
  unknown=[r for r in self.parents if r['cohort']=='general_unassigned_exploratory']
  expected={r['parent_id'] for r in sorted(unknown,key=lambda r:hashlib.sha256(('42:'+r['parent_id']).encode()).hexdigest())[:100]}
  actual={r['parent_id'] for r in self.selected if r['cohort']=='general_unassigned_exploratory'}
  self.assertEqual(expected,actual)
  self.assertEqual(collections.Counter(r['parent_id'] for r in self.selected if r['cohort']=='general_unassigned_exploratory'),collections.Counter({p:3 for p in expected}))

 def test_blind_interface_accepts_no_annotations_or_predictions(self):
  self.assertEqual(list(inspect.signature(blind_segment).parameters),['processed'])
  self.assertEqual(list(inspect.signature(blind_tiers).parameters),['image'])
  image=np.full((80,100),255,np.uint8);image[20:40,20:80]=0
  processed=SimpleNamespace(rows=[SimpleNamespace(image=image)])
  with patch('modes.blind_tiers',return_value=[('count',image)]) as spy:
   result=blind_segment(processed)
   spy.assert_called_once();self.assertIs(spy.call_args.args[0],image)
   self.assertEqual(result[0][0][0],'count')

 def test_alignment_rejects_instead_of_relabeling_blind_proposal(self):
  vocab,_=load_vocabulary(VOCABULARY)
  image=np.full((32,64),255,np.uint8)
  proposal=[[('count',image)]]
  truth=[dict(upper='A',count_line='a',lower='')]
  with self.assertRaisesRegex(SegmentationFailure,'tier_presence'):
   align_after_segmentation(proposal,truth,vocab)
  self.assertEqual(proposal[0][0][0],'count')
  with self.assertRaisesRegex(SegmentationFailure,'row_count'):
   align_after_segmentation([],truth,vocab)

 def test_ctc_and_leading_offset_alignment(self):
  vocab,_=load_vocabulary(VOCABULARY)
  image=np.full((64,16),255,np.uint8)
  result=align_after_segmentation([[('count',image)]],[dict(upper='',count_line='  aa',lower='')],vocab)
  self.assertEqual(result[0]['truth'],'aa');self.assertEqual(result[0]['leading_offset'],2)
  with self.assertRaisesRegex(SegmentationFailure,'ctc_infeasible'):
   align_after_segmentation([[('count',image)]],[dict(upper='',count_line='aaaaa',lower='')],vocab)

 def test_cer_exact_accuracy_and_no_data_are_distinct(self):
  rows=[dict(truth=t,prediction=p,**edit_counts(t,p)) for t,p in [('ABC','ADC'),('A','A')]]
  self.assertEqual(metrics(rows)['cer'],.25);self.assertEqual(metrics(rows)['exact_accuracy'],.5)
  self.assertEqual(edit_counts('AB','ABC')['insertions'],1)
  self.assertEqual(edit_counts('ABC','AB')['deletions'],1)
  self.assertIsNone(metrics([])['cer']);self.assertEqual(metrics([])['samples'],0)

 def test_frozen_checkpoint_and_vocabulary(self):
  import torch
  from models.crnn import CRNN
  torch.set_num_threads(4)
  vocab,digest=load_vocabulary(VOCABULARY)
  state=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
  model=CRNN(66);model.load_state_dict(state['model_state_dict'],strict=True)
  self.assertEqual(state['char_to_idx'],vocab.char_to_index)
  self.assertEqual(state['vocabulary_hash'],digest);self.assertEqual(len(vocab.char_to_index),65)
  self.assertEqual(model.rnn.hidden_size,256);self.assertEqual(model.rnn.num_layers,2)
  self.assertEqual(model.classifier.out_features,66)
  self.assertEqual(sha256(CHECKPOINT),'22450cbddd42968149739ea0293a6f4ad105f593e82387d707ecb6418a6bac0f')

 def test_completed_evaluation_failure_accounting(self):
  root=HERE/'reports/evaluation_v1'
  result=json.loads((root/'metrics.json').read_text())
  pages=read_csv(root/'page_processing.csv');predictions=read_csv(root/'predictions.csv');failures=read_csv(root/'segmentation_failures.csv')
  self.assertEqual(len(pages),900)
  self.assertEqual(sum(r['status']=='failed' for r in pages),len(failures))
  success={(r['sample_id'],r['mode']):r for r in pages if r['status']=='success'}
  counts=collections.Counter((r['sample_id'],r['mode']) for r in predictions)
  self.assertTrue(all(key in success for key in counts))
  for key,row in success.items():self.assertEqual(counts[key],int(row['tiers']))
  for cohort,modes in result['groups'].items():
   for mode,entry in modes.items():
    self.assertEqual(entry['pages_attempted'],entry['pages_processed']+entry['pages_failed'])
    self.assertEqual(entry['overall']['samples'],sum(r['cohort']==cohort and r['mode']==mode for r in predictions))

 def test_uncached_deterministic_replays_and_model_immutability(self):
  result=json.loads((HERE/'reports/evaluation_v1/metrics.json').read_text())
  expected={(c,m) for c,modes in result['groups'].items() for m,e in modes.items() if e['overall']['samples']>0}
  actual={(r['cohort'],r['mode']) for r in result['uncached_determinism_replays']}
  self.assertEqual(expected,actual)
  self.assertTrue(all(r['prediction_equal'] for r in result['uncached_determinism_replays']))
  self.assertTrue(result['deterministic_algorithms']);self.assertTrue(result['model_state_unchanged'])

if __name__=='__main__':unittest.main(verbosity=2)
