"""Focused policy tests; four saved development cases, no OCR/evaluation job."""
import copy
import json
import unittest
from pathlib import Path
from common import ROOT, digest, load_json
from rejection_pipeline import decide, load_decision_policy, DEFAULT_DECISION_POLICY
from talim_validator import validate_tiers
from evaluate_rejection import evaluate

CODE = 'DECODED_UNSUPPORTED_OR_WRONG_TIER_GLYPH'
DISABLED = dict(schema_version=1, enabled=False, action='REVIEW_UNCERTAIN', minimum={})


def observation(upper='A', lower='R'):
    return dict(preprocessing_status='ok', segmentation_accepted=True, segmentation_reason='',
                ocr_status='ok', decoded_rows=[dict(upper=upper, count='ja', lower=lower)], confidence={})


class F1PolicyV2Tests(unittest.TestCase):
    def test_wrong_tier_rule_still_detects_violation(self):
        validation = validate_tiers(observation('X')['decoded_rows'])
        self.assertEqual(validation['outcome'], 'REJECT_INVALID')
        result = decide(observation('X'))
        self.assertEqual(result['outcome'], 'REVIEW_UNCERTAIN')
        self.assertEqual(result['syntax_validation'], validation)
        self.assertEqual(result['evidence'], validation['evidence'])
        self.assertIn(CODE, result['reason_codes'])
        detail = result['reason_details'][0]
        self.assertEqual(detail['tier'], 'upper')
        self.assertEqual(detail['predicted_characters'], ['X'])
        self.assertEqual(detail['based_on'], 'ocr_output')
        self.assertEqual(detail['evidence_scope'], 'decoded_syntax')
        self.assertFalse(result['independent_image_invalidity_confirmed'])
        self.assertIn('does not independently prove', result['explanation'])

    def test_unsupported_glyph_is_review(self):
        result = decide(observation('?'))
        self.assertEqual(result['outcome'], 'REVIEW_UNCERTAIN')
        self.assertIn(CODE, result['reason_codes'])
        self.assertEqual(result['reason_details'][0]['predicted_characters'], ['?'])

    def test_independent_artifact_checks_keep_existing_actions(self):
        for state, expected in [('unreadable', 'REJECT_INVALID'), ('no_ink', 'REJECT_INVALID'),
                                ('missing_artifact', 'REVIEW_UNCERTAIN')]:
            with self.subTest(state=state):
                obs = observation()
                obs.update(preprocessing_status=state, segmentation_accepted=False, ocr_status='not_run', decoded_rows=[])
                result = decide(obs)
                self.assertEqual(result['outcome'], expected)
                if expected == 'REJECT_INVALID':
                    self.assertEqual(result['reason_codes'], ['INPUT_' + state.upper()])
                    self.assertEqual(result['reason_details'][0]['based_on'], 'independent_image_check')
                    self.assertTrue(result['independent_image_invalidity_confirmed'])

    def test_valid_recognition_accepts(self):
        result = decide(observation())
        self.assertEqual(result['outcome'], 'ACCEPT')
        self.assertEqual(result['reason_codes'], ['OBSERVABLE_CHECKS_PASSED'])

    def test_ocr_error_and_segmentation_uncertainty_preserve_both(self):
        obs = observation('X')
        obs.update(segmentation_accepted=False, segmentation_reason='confidence:ambiguous_row_or_role')
        result = decide(obs)
        self.assertEqual(result['outcome'], 'REVIEW_UNCERTAIN')
        self.assertEqual(result['reason_codes'], ['D1_NO_SAFE_PROPOSAL', CODE])
        self.assertEqual({d['based_on'] for d in result['reason_details']}, {'image_processing_uncertainty', 'ocr_output'})

    def test_independent_invalidity_overrides_ocr_only_error(self):
        for state in ('unreadable', 'no_ink'):
            obs = observation('X')
            obs['preprocessing_status'] = state
            result = decide(obs)
            self.assertEqual(result['outcome'], 'REJECT_INVALID')
            self.assertEqual(result['reason_codes'], ['INPUT_' + state.upper(), CODE])
            self.assertEqual(result['evidence_scope'], 'image_artifact')
            self.assertEqual(result['syntax_validation']['evidence']['glyphs'], ['X'])

    def test_ground_truth_and_claimed_invalid_flags_are_not_runtime_evidence(self):
        for obs in (observation(), observation('X')):
            expected = decide(obs)
            for truth in ('valid', 'invalid_talim_content'):
                poisoned = copy.deepcopy(obs)
                poisoned.update(image_ground_truth=truth, expected_system_outcome='REJECT_INVALID',
                    transcription_exact=True, truth_rows=[{'upper': 'Z'}],
                    image_invalid=True, independent_image_invalidity_confirmed=True)
                self.assertEqual(decide(poisoned), expected)

    def test_f0_exact_legacy_schema_and_actions(self):
        cases = [(observation('X'), dict(outcome='ACCEPT', reason_codes=['D1_PROPOSAL_E1_OUTPUT'], evidence_scope='baseline'))]
        for state in ('unreadable', 'no_ink'):
            obs = observation('X')
            obs['preprocessing_status'] = state
            cases.append((obs, dict(outcome='REJECT_INVALID', reason_codes=['INPUT_' + state.upper()], evidence_scope='image_artifact')))
        obs = observation('X')
        obs['segmentation_accepted'] = False
        cases.append((obs, dict(outcome='REVIEW_UNCERTAIN', reason_codes=['D1_NO_SAFE_PROPOSAL'], evidence_scope='segmentation')))
        obs = observation('X')
        obs['ocr_status'] = 'failed'
        cases.append((obs, dict(outcome='REVIEW_UNCERTAIN', reason_codes=['OCR_UNAVAILABLE'], evidence_scope='recognition')))
        for obs, expected in cases:
            self.assertEqual(decide(obs, 'F0'), expected)

    def test_f1_f2_consistent_provenance(self):
        for obs in (observation('X'), observation()):
            self.assertEqual(decide(obs, 'F1'), decide(obs, 'F2', DISABLED))

    def test_f2_retains_quality_and_ocr_reasons(self):
        policy = dict(schema_version=1, enabled=True, action='REVIEW_UNCERTAIN', minimum={'nonblank_geomean': .9})
        result = decide(observation('X'), 'F2', policy)
        self.assertEqual(result['outcome'], 'REVIEW_UNCERTAIN')
        self.assertEqual(result['reason_codes'], [CODE, 'CONFIDENCE_MISSING_NONBLANK_GEOMEAN'])

    def test_observation_not_mutated(self):
        obs = observation('X')
        before = copy.deepcopy(obs)
        decide(obs)
        self.assertEqual(obs, before)

    def test_four_saved_development_regressions(self):
        fixture = load_json(Path(__file__).parent / 'fixtures/f1_v2_development_cases.json')
        self.assertEqual(len(fixture['cases']), 4)
        self.assertEqual(digest(Path(fixture['source_baseline']) / 'observations.jsonl'), fixture['observations_sha256'])
        expected = {'normal_train_044361__clean': ('upper', 'X'),
            'positive_edge_00297__aug_01_perspective_shadow': ('upper', 'Z'),
            'positive_edge_00314__aug_02_resolution': ('lower', 'C'),
            'positive_edge_00314__aug_03_perspective': ('lower', 'C')}
        for item in fixture['cases']:
            with self.subTest(image_id=item['image_id']):
                validation = validate_tiers(item['observation']['decoded_rows'])
                self.assertEqual(validation['outcome'], 'REJECT_INVALID')
                result = decide(item['observation'])
                self.assertEqual(result['outcome'], 'REVIEW_UNCERTAIN')
                self.assertIn(CODE, result['reason_codes'])
                tier, glyph = expected[item['image_id']]
                self.assertEqual(result['evidence']['role'], tier)
                self.assertIn(glyph, result['evidence']['glyphs'])
                self.assertEqual(result['evidence'], validation['evidence'])
                self.assertEqual(result, decide(item['observation'], 'F2', DISABLED))

    def test_saved_output_evaluation_retains_provenance(self):
        item = dict(record=dict(image_id='unit', source_parent_id='unit', invalidity_category='none',
            data_origin='unit', image_ground_truth='valid', expected_system_outcome='ACCEPT'),
            observation=observation('X'), transcription_exact=False)
        rows, summary, _ = evaluate([item], 'F1', decision_policy=load_decision_policy())
        self.assertEqual(rows[0]['outcome'], 'REVIEW_UNCERTAIN')
        self.assertIn(CODE, rows[0]['reason_codes'])
        self.assertEqual(rows[0]['reason_details'][0]['tier'], 'upper')
        self.assertFalse(rows[0]['transcription_exact'])
        self.assertEqual(summary['false_rejection_rate'], 0)
        self.assertEqual(summary['positive_review_rate'], 1)

    def test_policy_is_versioned_and_cannot_change_image_rules(self):
        config = load_decision_policy(DEFAULT_DECISION_POLICY)
        self.assertEqual(config['policy_version'], 'f1_ocr_uncertainty_v2')
        config['independent_invalid_preprocessing_statuses'].append('ambiguous')
        with self.assertRaises(ValueError):
            decide(observation(), decision_policy=config)

    def test_historical_results_rules_checkpoint_and_manifests_unchanged(self):
        lock = load_json(ROOT / 'reports/f1_policy_v2/protected_before.json')
        for filename, expected in lock.items():
            with self.subTest(filename=filename):
                self.assertEqual(digest(filename), expected)


if __name__ == '__main__':
    unittest.main()
