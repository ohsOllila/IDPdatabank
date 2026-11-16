"""Tests for the synthetic ground-truth study; run with the IDPs unittest discovery."""

import unittest

import numpy as np
from run_synthetic_truth_study import (
    RG_TARGET_NM,
    R_MAX,
    R_POINTS,
    bift_style_chi2,
    coil_pair_distance,
    gyration_radius,
    solve_s1,
    true_discrepancy,
)


def grid():
    r = np.linspace(0, R_MAX, R_POINTS)
    weights = np.full(R_POINTS, r[1])
    weights[[0, -1]] *= 0.5
    return r, weights


class TruthModelTests(unittest.TestCase):
    def test_gyration_radius_of_single_gaussian_lobe(self):
        r, weights = grid()
        sigma = 30.0
        density = coil_pair_distance(r, sigma, 2.5 * sigma, 0.0)
        # For P(r) ~ r^2 exp(-r^2/2s^2): Rg = sqrt(3/2) s.
        np.testing.assert_allclose(gyration_radius(r, density, weights), np.sqrt(1.5) * sigma / 10, rtol=1e-10)

    def test_solved_s1_matches_target_rg(self):
        r, weights = grid()
        s1 = solve_s1(r, weights, RG_TARGET_NM, 2.5, 0.30)
        achieved = gyration_radius(r, coil_pair_distance(r, s1, 2.5 * s1, 0.30), weights)
        self.assertAlmostEqual(achieved, RG_TARGET_NM, places=6)

    def test_size_scaling_moves_rg_monotonically(self):
        r, weights = grid()
        s1 = solve_s1(r, weights, RG_TARGET_NM, 2.5, 0.30)
        smaller = gyration_radius(r, coil_pair_distance(r, 0.9 * s1, 0.9 * 2.5 * s1, 0.30), weights)
        larger = gyration_radius(r, coil_pair_distance(r, 1.1 * s1, 1.1 * 2.5 * s1, 0.30), weights)
        self.assertLess(smaller, RG_TARGET_NM)
        self.assertGreater(larger, RG_TARGET_NM)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.q = np.linspace(0.01, 0.4, 60)
        self.truth = np.exp(-30 * self.q**2)
        rng = np.random.default_rng(5)
        self.noisy = self.truth + rng.normal(0, 0.01, len(self.q))
        self.sigma = np.full_like(self.q, 0.01)

    def test_chi2_zero_for_exact_noiseless_curve(self):
        chi2, scale = bift_style_chi2(self.q, self.truth, self.sigma, self.q, 2.5 * self.truth)
        self.assertAlmostEqual(chi2, 0.0, places=10)
        self.assertAlmostEqual(scale, 0.4, places=10)

    def test_chi2_fit_is_sigma_independent(self):
        # The notebook's convention fits scale/offset unweighted; sigma only
        # normalizes residuals, so the fitted scale must not change with sigma.
        _, scale_correct = bift_style_chi2(self.q, self.noisy, self.sigma, self.q, self.truth)
        _, scale_bad = bift_style_chi2(self.q, self.noisy, 0.2 * self.sigma, self.q, self.truth)
        self.assertEqual(scale_correct, scale_bad)

    def test_chi2_near_one_for_calibrated_noise(self):
        chi2, _ = bift_style_chi2(self.q, self.noisy, self.sigma, self.q, self.truth)
        self.assertAlmostEqual(chi2, 1.0, delta=0.5)

    def test_chi2_grows_when_sigma_underestimated(self):
        chi2_correct, _ = bift_style_chi2(self.q, self.noisy, self.sigma, self.q, self.truth)
        chi2_bad, _ = bift_style_chi2(self.q, self.noisy, 0.2 * self.sigma, self.q, self.truth)
        self.assertAlmostEqual(chi2_bad, chi2_correct / 0.04, delta=1.0)

    def test_true_discrepancy_zero_for_affine_equivalent(self):
        sim = self.truth.copy()
        value = true_discrepancy(self.q, 3.1 * sim + 0.07, self.q, self.truth)
        self.assertAlmostEqual(value, 0.0, places=10)

    def test_true_discrepancy_grows_with_perturbation(self):
        base = true_discrepancy(self.q, self.truth, self.q, self.truth)
        slight = true_discrepancy(self.q, self.truth * (1 + 0.02 * self.q), self.q, self.truth)
        strong = true_discrepancy(self.q, self.truth * (1 + 0.3 * self.q), self.q, self.truth)
        self.assertAlmostEqual(base, 0.0, places=12)
        self.assertLess(base, slight)
        self.assertLess(slight, strong)
        self.assertGreater(strong, 0)


if __name__ == "__main__":
    unittest.main()
