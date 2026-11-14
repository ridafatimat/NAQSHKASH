import unittest
from confidence import check, from_logits


class ConfidenceTests(unittest.TestCase):
    def test_threshold_boundary_and_missing(self):
        policy = dict(schema_version=1, enabled=True, action='REVIEW_UNCERTAIN', minimum={'role_margin': 0.5})
        self.assertEqual(check({'role_margin': 0.5}, policy), [])
        self.assertTrue(check({'role_margin': 0.49}, policy))
        self.assertTrue(check({}, policy))
        self.assertTrue(check({'role_margin': float('nan')}, policy))

    def test_invalid_policy(self):
        with self.assertRaises(ValueError):
            check({}, dict(schema_version=1, enabled=True, action='REJECT_INVALID', minimum={}))

    def test_sequence_score_sums_alignments(self):
        import torch
        # Greedy target [1], three legal CTC paths: 11, 10 and 01.
        logits = torch.tensor([[0., 1.], [0., 1.]])
        signals = from_logits(logits)
        p = logits.softmax(-1)
        expected = p[0, 1] * p[1, 1] + p[0, 1] * p[1, 0] + p[0, 0] * p[1, 1]
        self.assertAlmostEqual(signals['ctc_log_sequence_score'], float(expected.log()), places=5)
        self.assertFalse(signals['calibrated_probability'])

    def test_blank_sequence_and_nonfinite(self):
        import torch
        self.assertIsNone(from_logits(torch.tensor([[10., -10.]]))['nonblank_geomean'])
        with self.assertRaises(ValueError):
            from_logits(torch.tensor([[float('nan'), 0.]]))


if __name__ == '__main__':
    unittest.main()
