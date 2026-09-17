"""Constructed warm-extraction and sealed-selection interface checks."""
import unittest
import math
from warm_costs import base9, extract_all, method_selection, fixed_schedule
from metrics_ab import prepare_ab_items
import numpy as np


class WarmInterfaceTests(unittest.TestCase):
    def raw(self):
        return dict(loglik=np.log([[.8, .2], [.6, .4]]), char_lens=np.ones((2, 2)),
                    gold=np.array([0, 1]), item_ids=np.array(['c', 'a']), n_options=np.array([2, 2]))

    def test_nine_baseline_hand_values(self):
        raw = self.raw()
        got = base9(raw, prepare_ab_items(**raw))
        entropy = lambda x: -x*math.log(x)-(1-x)*math.log(1-x)
        expected = dict(A=.5, C=.7, O=.2, REL10=.2, R50=0, ECE10=.4,
                        Brier=.4, H=(entropy(.8)+entropy(.6))/(2*math.log(2)), Margin=.4)
        for key, value in expected.items():
            self.assertAlmostEqual(got[key], value, delta=1e-12)

    def test_all_twenty_flags_and_nontrivial_global_alignment(self):
        reference = dict(w=np.array([.1, .2, .8, .9]),
                         b_fit=np.full((2, 4), np.nan), b_eval=np.full((2, 4), np.nan),
                         direction_failures=np.array(['SYNTHETIC_FIT_FAILURE']*2))
        features, reasons = extract_all(self.raw(), ['a', 'b', 'c', 'd'], [0, 1, 0, 1], reference)
        self.assertEqual(len(features), 49)
        self.assertEqual(len(reasons), 20)
        self.assertAlmostEqual(features['C1'], .05, delta=1e-12)
        self.assertAlmostEqual(features['C2'], .03, delta=1e-12)
        self.assertAlmostEqual(features['C3'], .05*math.log(1.5), delta=1e-12)
        for metric in ('C4', 'D1', 'D2', 'D3', 'D4'):
            self.assertEqual(features['missing_' + metric], 1)
            self.assertTrue(math.isnan(features[metric]))
        self.assertEqual(features['missing_A1'], 0)

    def test_procedure_endpoints_resolve_different_sealed_packages(self):
        pr = dict(feature_groups={'B0': ['A'], 'P_A1': ['A', 'A1']},
                  procedures={'S0': ['B0', 'P_A1']}, endpoints=['Y_sel', 'Y_cal'])
        packages = [dict(outer_fold=0, group=g, endpoint=e,
                         best_estimator=dict(path=g+e, sha256=g+e))
                    for g in pr['feature_groups'] for e in pr['endpoints']]
        procedures = [dict(outer_fold=0, procedure='S0', endpoint=e, chosen_package=g,
                          best_estimator=dict(path=g+e, sha256=g+e))
                      for e, g in [('Y_sel', 'B0'), ('Y_cal', 'P_A1')]]
        result = method_selection(pr, [dict(packages=packages, procedures=procedures)])
        self.assertEqual(result[0, 'S0', 'Y_sel']['package'], 'B0')
        self.assertEqual(result[0, 'S0', 'Y_cal']['package'], 'P_A1')
        procedures[1]['best_estimator']['sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'ESTIMATOR_MISMATCH'):
            method_selection(pr, [dict(packages=packages, procedures=procedures)])

    def test_fixed_schedule_covers_each_model_method_three_times(self):
        ids = [f'org/model{i:03}' for i in range(110)]
        methods = ['B0', 'P_A1', 'S0']
        selected, schedule = fixed_schedule(ids, methods)
        second_selected, second_schedule = fixed_schedule(ids[::-1], methods)
        self.assertEqual((selected, schedule), (second_selected, second_schedule))
        self.assertEqual(len(selected), 100)
        self.assertEqual(len(schedule), 900)
        for repeat in range(3):
            for mid in selected:
                rows = [r for r in schedule if r['repetition'] == repeat and r['model_id'] == mid]
                self.assertEqual({r['method'] for r in rows}, set(methods))
                self.assertEqual({r['order_within_model'] for r in rows}, {0, 1, 2})


if __name__ == '__main__':
    unittest.main()
