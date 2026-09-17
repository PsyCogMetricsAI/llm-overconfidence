"""Constructed, source-only acceptance examples with analytic expectations."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import unittest
import numpy as np
from numpy.testing import assert_allclose
from metrics_structure import (reference_ease, compute_c, reference_parameters,
    project_reference_items, fit_reference_direction, residual_statistics,
    compute_d_direction, combine_d, compute_structure_one)


class StructureTests(unittest.TestCase):
    def test_ease_organization_equal_and_19_20_boundary(self):
        y = np.zeros((20, 2), bool)
        y[-1] = True
        valid = np.ones_like(y)
        valid[0, 1] = False
        orgs = ['big'] * 19 + ['small']
        got = reference_ease(y, valid, orgs, ['big', 'small'], ['scored'])
        # 19 errors: .5/20; one success: 1.5/2; institutions equally weighted.
        self.assertAlmostEqual(got['w'][0], (.025 + .75) / 2)
        self.assertTrue(np.isnan(got['w'][1]))
        self.assertEqual(got['n_reference'].tolist(), [20, 19])
        self.assertEqual(got['n_reference_orgs'].tolist(), [2, 2])
        with self.assertRaisesRegex(ValueError, 'OVERLAP'):
            reference_ease(y, valid, orgs, ['big'], ['big'])

    def test_c_fixed_denominators_and_quartiles(self):
        c = np.array([.1, .2, .3, .4, .5])
        y, z, valid, w = np.zeros(5), np.arange(5.), np.ones(5, bool), np.full(5, .5)
        # Equal ease, canonical IDs swap first two items; the first quartile has two.
        result = compute_c(c, y, z, valid, ['b', 'a', 'c', 'd', 'e'], w)
        assert_allclose(result['values'], [.5, .15, 1., .019], atol=1e-14)
        r4 = compute_c(c[:4], y[:4], z[:4], valid[:4], ['a', 'b', 'c', 'd'], w[:4])
        self.assertAlmostEqual(r4['values'][3], .0125)
        r3 = compute_c(c[:3], y[:3], z[:3], valid[:3], ['a', 'b', 'c'], w[:3])
        self.assertTrue(np.isnan(r3['values'][3]))
        self.assertEqual(r3['reasons'][3], 'REFERENCE_VALID_ITEMS_LT4')
        w[0] = np.nan
        restricted = compute_c(c, y, z, valid, ['a', 'b', 'c', 'd', 'e'], w)
        self.assertEqual(restricted['n'], 4)
        assert_allclose(restricted['values'][:3], [.5, .175, 1.25], atol=1e-14)
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            compute_c(c, y, z, valid, ['a'] * 5, w)
        empty = compute_c(c, y, z, valid, list('abcde'), np.full(5, np.nan))
        self.assertTrue(np.isnan(empty['values']).all())

    def test_reference_projection_and_gauge(self):
        n = 20
        a = np.linspace(-.4, .6, n)
        s = np.linspace(.3, 2., n)
        b = np.array([-3., .2, 2.])
        ell = a[:, None] - s[:, None] * b
        mask = np.ones_like(ell, bool)
        result = project_reference_items(ell, mask, a, s)
        assert_allclose(result['b'], b, atol=1e-14)
        # Equivalent old gauge raw b = mu + scale * new b.
        mu, scale = 1.7, 2.3
        shifted = project_reference_items(ell, mask, a - s * mu, s * scale)
        assert_allclose(shifted['b'], (b - mu) / scale, atol=1e-14)
        # Deliberately nonzero mean/nonunit variance must survive projection.
        self.assertGreater(abs(result['b'].mean()), .1)
        mask[0, 0] = False
        self.assertTrue(np.isnan(project_reference_items(ell, mask, a, s)['b'][0]))
        zero = project_reference_items(ell, np.ones_like(mask), a, np.zeros(n))
        self.assertTrue(np.isnan(zero['b']).all())

    def test_reference_parameters_masked_exact_line(self):
        b = np.arange(6.)
        a, s = np.array([.2, .8]), np.array([.6, 1.1])
        ell = a[:, None] - s[:, None] * b
        valid = np.ones_like(ell, bool)
        valid[0, 0] = False
        ell[0, 0] = np.nan
        aa, ss = reference_parameters(ell, valid, b)
        assert_allclose(aa, a, atol=1e-14)
        assert_allclose(ss, s, atol=1e-14)

    def test_als_two_directions_on_constructed_rank_one(self):
        n, m = 20, 200
        halves = np.repeat([0, 1], 100)
        rawb = np.r_[np.linspace(-2, 2, 100), np.linspace(-1, 5, 100)]
        a, s = np.linspace(-.5, .5, n), np.linspace(.4, 1.4, n)
        ell = a[:, None] - s[:, None] * rawb
        valid = np.ones((n, m), bool)
        for h in (0, 1):
            result = fit_reference_direction(ell, valid, halves, h)
            fit, evaluate = halves == h, halves == 1 - h
            expected = (rawb - rawb[fit].mean()) / rawb[fit].std()
            assert_allclose(result['b_fit'][fit], expected[fit], atol=1e-12)
            assert_allclose(result['b_eval'][evaluate], expected[evaluate], atol=1e-12)
            self.assertTrue(np.isnan(result['b_fit'][evaluate]).all())
            self.assertTrue(np.isnan(result['b_eval'][fit]).all())
        with self.assertRaisesRegex(ValueError, 'DEGENERATE_REFERENCE_INIT'):
            fit_reference_direction(np.zeros_like(ell), valid, halves, 0)

    def test_residual_analytic_values_and_thresholds(self):
        r, b = np.array([-2., -1., 1., 2.]), np.array([-2., -1., 1., 3.])
        got = residual_statistics(r, b)
        expected = [.5, -1.64, 0., 2.5 / np.sqrt(2.5 * 10.6875)]
        assert_allclose(got['values'], expected, atol=1e-14)
        # M2 > 1e-12 although M2 squared is < 1e-12: D2 remains valid.
        small = residual_statistics(r * 1e-4, b)
        assert_allclose(small['values'], expected, atol=1e-14)
        zero = residual_statistics(np.zeros(4), b)
        self.assertTrue(np.isnan(zero['values']).all())
        self.assertEqual(zero['reasons'][0], 'RESIDUAL_ENERGY_LE1E12')
        constant = residual_statistics(np.ones(4), b)
        self.assertEqual(constant['values'][0], 0.)
        self.assertTrue(np.isnan(constant['values'][1:]).all())
        positive = residual_statistics(r + 10, b)
        self.assertEqual(positive['values'][0], 0.)

    @staticmethod
    def model_example():
        b = np.linspace(-2, 2, 100)
        noise = np.cos(np.arange(100))
        noise -= noise.mean()
        noise -= b * np.mean(noise * b) / np.mean(b * b)
        noise *= .2 / np.sqrt(np.mean(noise ** 2))
        evaluate_b = b + 1.3
        # Unique absolute values avoid a roundoff-sensitive near-tie assertion;
        # exact average-rank ties are tested directly above.
        evaluate_r = np.linspace(-2.03, 2.01, 100)
        ell = np.r_[.3 - 1.2 * b + noise, .3 - 1.2 * evaluate_b + .2 * evaluate_r]
        return ell, np.ones(200, bool), np.repeat([0, 1], 100), np.r_[b, np.full(100, np.nan)], np.r_[np.full(100, np.nan), evaluate_b], evaluate_r

    def test_model_direction_fits_only_training_half(self):
        ell, valid, halves, bfit, beval, expected_r = self.model_example()
        got = compute_d_direction(ell, valid, halves, 0, bfit, beval)
        assert_allclose(got['parameters'][:4], [.3, 1.2, .2, .25], atol=1e-12)
        centered = expected_r - expected_r.mean()
        ranks_abs = np.argsort(np.argsort(abs(expected_r))) + 1
        expected = [np.sum(np.minimum(expected_r, 0.) ** 2) / np.sum(expected_r ** 2),
                    np.mean(centered ** 4) / np.mean(centered ** 2) ** 2 - 3.,
                    np.corrcoef(ranks_abs, np.arange(1, 101))[0, 1],
                    np.corrcoef(expected_r, beval[100:] ** 2)[0, 1]]
        assert_allclose(got['values'], expected, atol=1e-12)
        changed = ell.copy()
        changed[100:] += 10.
        again = compute_d_direction(changed, valid, halves, 0, bfit, beval)
        assert_allclose(again['parameters'], got['parameters'], atol=0.)
        self.assertEqual(again['values'][0], 0.)
        short = valid.copy()
        short[100] = False
        self.assertEqual(compute_d_direction(ell, short, halves, 0, bfit, beval)['reasons'][0], 'EVALUATION_ITEMS_LT100')
        reversed_fit = ell.copy()
        reversed_fit[:100] *= -1
        self.assertIn('S_LE1E8', compute_d_direction(reversed_fit, valid, halves, 0, bfit, beval)['reasons'][0])
        exact = ell.copy()
        exact[:100] = .3 - 1.2 * bfit[:100]
        self.assertIn('SIGMA_LE1E8', compute_d_direction(exact, valid, halves, 0, bfit, beval)['reasons'][0])

    def test_two_direction_requirement_and_single_model_schema(self):
        d0 = dict(values=np.array([.2, 1., 0., .3]), reasons=np.full(4, ''))
        d1 = dict(values=np.array([.4, np.nan, 0., .5]), reasons=np.array(['', 'FAIL', '', '']))
        combined = combine_d(d0, d1)
        assert_allclose(combined['values'][[0, 2, 3]], [.3, 0., .4], atol=1e-14)
        self.assertTrue(np.isnan(combined['values'][1]))
        self.assertEqual(combined['reasons'][1], 'H1:FAIL')
        ell, valid, halves, bf, be, _ = self.model_example()
        ref = dict(w=np.full(200, .5), b_fit=np.array([bf, be]), b_eval=np.array([be, bf]),
                   direction_failures=np.array(['', 'TEST_FAILURE']))
        src = dict(ell=ell, c=np.full(200, .4), y=np.zeros(200), z=np.ones(200), valid=valid,
                   source_half=halves, item_ids=np.array(['i%03d' % i for i in range(200)]))
        got = compute_structure_one(src, ref)
        self.assertEqual(got['values'].shape, (8,))
        self.assertEqual(got['half_values'].shape, (2, 8))
        self.assertTrue(np.isfinite(got['values'][:4]).all())
        self.assertTrue(np.isnan(got['values'][4:]).all())
        self.assertEqual(got['D_parameters'].shape, (2, 5))


if __name__ == '__main__':
    unittest.main(verbosity=2)
