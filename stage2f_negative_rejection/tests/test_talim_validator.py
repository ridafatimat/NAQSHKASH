import unittest
from talim_validator import validate_tiers, validate_aligned_row, validate_annotation, rules


class TalimTests(unittest.TestCase):
    def test_valid_observable(self):
        r = validate_tiers([dict(upper='A', count='ja', lower='R')])
        self.assertEqual(r['outcome'], 'ACCEPT')
        self.assertFalse(r['evidence']['image_validity_proven'])

    def test_wrong_tier(self):
        self.assertEqual(validate_tiers([dict(upper='R', count='ja')])['outcome'], 'REJECT_INVALID')

    def test_uncertain(self):
        for rows in ([], [dict(upper='A')], [dict(count='ja')], [dict(upper='O', count='ja')]):
            self.assertEqual(validate_tiers(rows)['outcome'], 'REVIEW_UNCERTAIN')

    def test_all_authentic_counts_aligned(self):
        for n in range(1, 260):
            text = rules.canonical_row([dict(symbol='A', count=n)])
            self.assertEqual(validate_aligned_row(*text, expected_cells=n)['outcome'], 'ACCEPT')

    def test_aligned_wrong_sum_and_direction(self):
        text = rules.canonical_row([dict(symbol='A', count=11), dict(symbol='R', count=19)])
        self.assertEqual(validate_aligned_row(*text, expected_cells=31)['reason_codes'], ['ROW_SUM_MISMATCH'])
        self.assertEqual(validate_aligned_row('R', 'a', '')['outcome'], 'REJECT_INVALID')

    def test_annotation_scope(self):
        r = validate_annotation({'logical_rows': []})
        self.assertEqual(r['outcome'], 'REJECT_INVALID')
        self.assertEqual(r['scope'], 'annotation_only')


if __name__ == '__main__':
    unittest.main()
