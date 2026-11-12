import copy
import tempfile
import unittest
from pathlib import Path
from common import *
from prepare import validate_prepared, inventory
from rejection_pipeline import decide
from evaluate_rejection import metrics, evaluate, load_baseline
from compare_candidates import compare
from calibrate import calibrate


def observation():
    return dict(preprocessing_status='ok', segmentation_accepted=True, ocr_status='ok',
                decoded_rows=[dict(upper='A', count='a')], confidence={'nonblank_geomean': .9})


class ContractTests(unittest.TestCase):
    def test_frozen_paths_schema_parent_integrity(self):
        report = validate_prepared()
        self.assertTrue(report['parent_disjoint'])
        self.assertGreater(report['historical_files_verified'], 20)

    def test_fixture_category_and_label_distinction(self):
        counts = load_json(ROOT / 'reports/audit_counts.json')
        self.assertEqual(counts['independent_source_parents'], 1)
        self.assertEqual(counts['categories'], 49)
        rows = read_csv(ROOT / 'manifests/negative_inventory.csv')
        self.assertEqual(len({r['source_parent_id'] for r in rows}), 1)
        base = next(r for r in rows if r['invalidity_category'] == 'count_zero' and r['augmentation'] == 'clean')
        self.assertEqual(base['image_ground_truth'], 'valid')
        self.assertEqual(base['expected_system_outcome'], 'ACCEPT')

    def test_three_way_decisions(self):
        obs = observation()
        self.assertEqual(decide(obs, 'F0')['outcome'], 'ACCEPT')
        obs['segmentation_accepted'] = False
        self.assertEqual(decide(obs)['outcome'], 'REVIEW_UNCERTAIN')
        obs['preprocessing_status'] = 'unreadable'
        self.assertEqual(decide(obs)['outcome'], 'REJECT_INVALID')
        with self.assertRaises(ValueError):
            decide(obs, 'F3')

    def test_confidence_does_not_claim_invalidity(self):
        policy = dict(schema_version=1, enabled=True, action='REVIEW_UNCERTAIN', minimum={'nonblank_geomean': .95})
        self.assertEqual(decide(observation(), 'F2', policy)['outcome'], 'REVIEW_UNCERTAIN')

    def test_metrics_three_way_denominators(self):
        def row(truth, outcome, expected, exact=None):
            return dict(image_ground_truth=truth, outcome=outcome, expected_system_outcome=expected, transcription_exact=exact)
        m = metrics([row('valid', 'ACCEPT', 'ACCEPT', True), row('valid', 'REVIEW_UNCERTAIN', 'ACCEPT', False),
                     row('corrupted_unreadable', 'REJECT_INVALID', 'REJECT_INVALID'),
                     row('empty_input', 'ACCEPT', 'REJECT_INVALID'), row('unadjudicated', 'REVIEW_UNCERTAIN', 'REVIEW_UNCERTAIN')])
        self.assertEqual(m['negative_rejection_recall'], .5)
        self.assertEqual(m['false_acceptance_rate'], .5)
        self.assertEqual(m['positive_review_rate'], .5)
        self.assertEqual(m['false_rejection_rate'], 0)
        self.assertIsNone(m['content_rejection_recall'])
        self.assertEqual(m['accepted_output_correctness'], 1.)
        self.assertIsNone(metrics([])['false_acceptance_rate'])

    def test_calibration_rejects_evaluation(self):
        with self.assertRaises(ValueError):
            calibrate([], dict(smoke=False, manifest_sha256='wrong'))

    def test_current_calibration_reports_not_trained(self):
        policy, report = calibrate([], dict(smoke=False, manifest_sha256=digest(ROOT / 'manifests/development_manifest.csv')))
        self.assertIsNone(policy)
        self.assertEqual(report['status'], 'NOT_CALIBRATED')

    def test_candidate_comparison_rejects_unpaired(self):
        # Small fabricated unit-test inputs only; never stored as experiment results.
        with tempfile.TemporaryDirectory() as tmp:
            dirs = []
            for name, sha in [('F0', 'a'), ('F1', 'b')]:
                p = Path(tmp) / name
                p.mkdir()
                result = dict(candidate=name, lineage=dict(manifest_sha256=sha, checkpoint_sha256=CHECKPOINT_SHA,
                    observations_sha256='same', smoke=False), rows=[])
                write_json(p / 'results.json', result)
                write_json(p / 'completion.json', dict(complete=True, rows=0, results_sha256=digest(p / 'results.json')))
                dirs.append(p)
            with self.assertRaises(ValueError):
                compare(dirs)

    def test_checkpoint_strict_compatible_frozen_no_forward(self):
        from evaluate_baseline import load_recognizer
        model, vocab, device = load_recognizer('cpu')
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertEqual(len(vocab.char_to_index), 65)
        self.assertEqual(vocab.blank_index, 0)


if __name__ == '__main__':
    unittest.main()
