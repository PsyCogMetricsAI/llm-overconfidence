"""Hand-derived boundary checks for the A/B source metrics; no real data."""
import math
import unittest

import numpy as np

from metrics_ab import METRIC_IDS, compute_ab, compute_ab_splits, prepare_ab_items


def from_scores(scores, gold, ids=None, lengths=None, n_options=None):
    scores = np.asarray(scores, dtype=float)
    if lengths is None:
        lengths = np.where(np.isnan(scores), np.nan, 1.0)
    if ids is None:
        ids = [f'item{i}' for i in range(len(scores))]
    return prepare_ab_items(scores, lengths, np.asarray(gold), ids, n_options)


class ABFormulaTests(unittest.TestCase):
    def assertMetric(self, block, metric, expected):
        self.assertEqual(block['reasons'][metric], '')
        self.assertAlmostEqual(block['values'][metric], expected, delta=1e-10)

    def test_tied_binary_hand_computation_all_twelve(self):
        p = [[.9, .1], [.9, .1], [.6, .4], [.6, .4]]
        items = from_scores(np.log(p), [0, 1, 0, 1], ['b', 'a', 'd', 'c'])
        got = compute_ab(items)
        expected = dict(A1=.5, A2=2/3, A3=0, A4=1, A5=1,
                        B1=-math.log(.9*.1*.6*.4)/4, B2=.375,
                        B3=.2925, B4=(math.log(9)+math.log(1.5))/4,
                        B5=math.log(9), B6=.5, B7=0)
        self.assertEqual(tuple(got['values']), METRIC_IDS)
        for key, value in expected.items():
            self.assertMetric(got, key, value)

    def test_auc_perfect_reversed_and_single_class(self):
        items = from_scores([[2, 0], [1, 0]], [0, 1])
        self.assertMetric(compute_ab(items), 'A1', 1)
        items = from_scores([[1, 0], [2, 0]], [0, 1])
        self.assertMetric(compute_ab(items), 'A1', 0)
        correct = from_scores([[2, 0], [1, 0]], [0, 0])
        got = compute_ab(correct)
        self.assertTrue(math.isnan(got['values']['A1']))
        self.assertEqual(got['reasons']['A1'], 'SINGLE_CLASS')
        for key in ('B2', 'B3', 'B4', 'B5', 'B6'):
            self.assertMetric(got, key, 0)

    def test_option_ties_and_variable_k(self):
        items = from_scores([[0, 0, np.nan, np.nan, np.nan],
                             [0, 0, 0, np.nan, np.nan],
                             [0, 0, 0, 0, np.nan], [0, 0, 0, 0, 0]],
                            [0, 1, 3, 4], n_options=[2, 3, 4, 5])
        np.testing.assert_array_equal(items['y'], [True, False, False, False])
        np.testing.assert_allclose(items['gold_rank_depth'], [0, .5, 1, 1], atol=1e-12)
        np.testing.assert_allclose(items['z'], 0, atol=0)
        np.testing.assert_allclose(items['nonwinner_entropy'], [0, 1, 1, 1], atol=1e-12)

    def test_underflow_does_not_corrupt_nll_entropy_or_rank(self):
        # Gold u=-10001 ranks below u=-10000, although both probabilities underflow.
        items = from_scores([[0, -10001, -10000]], [1])
        self.assertMetric(compute_ab(items), 'B1', 10001)
        self.assertMetric(compute_ab(items), 'B4', 10001)
        self.assertMetric(compute_ab(items), 'B6', 1)
        q = 1/(1+math.exp(-1))
        h = -(q*math.log(q)+(1-q)*math.log(1-q))/math.log(2)
        self.assertMetric(compute_ab(items), 'B7', h)
        self.assertEqual(items['c'][0], 1)

    def test_character_normalization_not_raw_loglik(self):
        # Raw -1 beats -2, but normalized -.5 beats -1.
        items = from_scores([[-1, -2]], [1], lengths=[[1, 4]])
        self.assertTrue(items['y'][0])
        self.assertMetric(compute_ab(items), 'B4', 0)

    def test_bins_include_point_one_point_nine_and_one(self):
        # Prepared-stage test supplies exact representable boundary values.
        items = from_scores([[1, 0]]*4, [0, 1, 0, 1])
        items['c'] = np.array([.1, np.nextafter(.1, 0), .9, 1.0])
        # Bins 1 and 0 each pure; bin9 contains one correct and one error.
        self.assertMetric(compute_ab(items), 'A3', .125)

    def test_ceil_and_zero_padding_of_fixed_top_tail(self):
        items = from_scores([[math.log(9), 0]]*11, [1]+[0]*10)
        got = compute_ab(items)
        self.assertMetric(got, 'B5', math.log(9)/2)
        one = compute_ab(from_scores([[1, 0]], [1]))
        for key in ('A2', 'A4', 'A5'):
            self.assertMetric(one, key, 1)

    def test_split_alignment_and_empty_subset(self):
        items = from_scores([[2, 0], [1, 0], [0, 1]], [0, 1, 1], ['z', 'a', 'b'])
        parts = compute_ab_splits(items, np.array([1, 0, 1], dtype=np.int8))
        self.assertEqual([parts[k]['n_items'] for k in ('full', 'half0', 'half1')], [3, 1, 2])
        self.assertMetric(parts['half0'], 'A4', 1)
        self.assertMetric(parts['half1'], 'B2', 0)
        empty = compute_ab(items, np.zeros(3, dtype=bool))
        self.assertTrue(all(v == 'NO_VALID_ITEMS' for v in empty['reasons'].values()))
        with self.assertRaises(ValueError):
            compute_ab_splits(items, [0, 1, 2])
        with self.assertRaises(ValueError):
            compute_ab(items, [0, 1, 1])

    def test_validity_rejects_bad_items_without_dropping_real_options(self):
        scores = [[0, -1, np.nan], [0, -1, 0], [0, -1, np.nan],
                  [0, np.inf, np.nan], [0, -1, np.nan], [0, -1, np.nan]]
        lengths = [[1, 1, np.nan], [1, 1, np.nan], [1, 0, np.nan],
                   [1, 1, np.nan], [1, 1, np.nan], [1, 1, np.nan]]
        items = from_scores(scores, [0, 0, 0, 0, 1.5, 0], lengths=lengths,
                            n_options=[2, 2, 2, 2, 2, 3])
        np.testing.assert_array_equal(items['valid'], [True, False, False, False, False, False])
        self.assertTrue(np.isnan(items['z'][1:]).all())
        self.assertEqual(compute_ab(items)['n_items'], 1)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            from_scores([[0, 1], [0, 1]], [0, 0], ['same', 'same'])


if __name__ == '__main__':
    unittest.main()
