"""Small synthetic records live only in temporary unit-test directories."""
import tempfile
import unittest
from pathlib import Path
from common import *
from calibrate import calibrate
from compare_candidates import compare
from evaluate_rejection import evaluate


class WorkflowTests(unittest.TestCase):
    def test_quality_gate_quantile_and_positive_only_scope(self):
        items = []
        for i in range(20):
            items.append(dict(record=dict(partition='development', image_ground_truth='valid', source_parent_id='p' + str(i)),
                observation=dict(preprocessing_status='ok', segmentation_accepted=True, ocr_status='ok',
                                 decoded_rows=[dict(upper='A', count='a')], confidence={'nonblank_geomean': .5 + i / 100})))
        lineage = dict(smoke=False, manifest_sha256=digest(ROOT / 'manifests/development_manifest.csv'), observations_sha256='unit-test')
        policy, report = calibrate(items, lineage, .05, True)
        self.assertAlmostEqual(policy['minimum']['nonblank_geomean'], .51)
        self.assertEqual(report['observed_additional_review_fraction'], .05)
        self.assertFalse(policy['calibrated_validity_probability'])
        items[0]['record']['partition'] = 'evaluation'
        with self.assertRaises(ValueError):
            calibrate(items, lineage, .05, True)

    def test_evaluate_comparison_roundtrip_and_groups(self):
        item = dict(record=dict(image_id='unit-test', source_parent_id='unit-parent', invalidity_category='none',
                    data_origin='unit', image_ground_truth='valid', expected_system_outcome='ACCEPT'),
                    observation=dict(preprocessing_status='ok', segmentation_accepted=True, ocr_status='ok',
                                     decoded_rows=[dict(upper='A', count='a')], confidence={}), transcription_exact=True)
        with tempfile.TemporaryDirectory() as tmp:
            directories = []
            for candidate in ('F0', 'F1'):
                rows, summary, groups = evaluate([item], candidate)
                self.assertEqual(summary['accepted_output_correctness'], 1.)
                self.assertIn('unit-parent', groups['source_parent_id'])
                directory = Path(tmp) / candidate
                directory.mkdir()
                result = dict(candidate=candidate, rows=rows,
                    lineage=dict(manifest_sha256='same', checkpoint_sha256=CHECKPOINT_SHA, observations_sha256='same', smoke=False))
                write_json(directory / 'results.json', result)
                write_json(directory / 'completion.json', dict(complete=True, rows=1, results_sha256=digest(directory / 'results.json')))
                directories.append(directory)
            table, _ = compare(directories)
            self.assertEqual([r['candidate'] for r in table], ['F0', 'F1'])
            self.assertEqual(table[0]['positive_acceptance_rate'], 1.)


if __name__ == '__main__':
    unittest.main()
