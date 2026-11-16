"""Tests for draw-mixture CRPS; run with the IDPs unittest discovery command."""

import unittest

import numpy as np
from gp_quality import gaussian_crps
from gp_quality_draws import empirical_crps, mixture_crps, pairwise_crps_term, score_simulation_draws


class MixtureCrpsTests(unittest.TestCase):
    def test_single_component_matches_gaussian_crps(self):
        observation = np.array([0.3, -1.2, 2.0])
        means = np.array([[0.0, 0.5, 1.0]])
        sds = np.array([[1.0, 0.7, 2.0]])
        pairwise_half = pairwise_crps_term(means, sds)
        np.testing.assert_allclose(
            mixture_crps(observation, means, sds, pairwise_half),
            gaussian_crps(observation, means[0], sds[0]),
            rtol=1e-12,
        )

    def test_identical_components_match_gaussian_crps(self):
        observation = np.array([0.0, 1.5])
        means = np.tile([[1.0, -0.5]], (7, 1))
        sds = np.tile([[0.3, 1.1]], (7, 1))
        pairwise_half = pairwise_crps_term(means, sds)
        np.testing.assert_allclose(
            mixture_crps(observation, means, sds, pairwise_half),
            gaussian_crps(observation, means[0], sds[0]),
            rtol=1e-10,
        )

    def test_mixture_matches_empirical_draws(self):
        rng = np.random.default_rng(7)
        means = np.array([[-1.0, 0.0], [1.0, 0.4], [0.3, -0.8]])
        sds = np.array([[0.5, 0.2], [0.8, 0.6], [0.3, 0.9]])
        pairwise_half = pairwise_crps_term(means, sds)
        observation = np.array([0.2, -0.1])
        samples = rng.normal(means[None, :, :], sds[None, :, :], size=(200000,) + means.shape)
        for point in range(2):
            empirical = empirical_crps(observation[point], samples[:, :, point].ravel())
            analytic = mixture_crps(observation, means, sds, pairwise_half)[point]
            self.assertAlmostEqual(empirical, analytic, delta=0.02 * max(analytic, 1e-3))

    def test_mixture_penalizes_bimodality(self):
        # An observation at the mixture mean of two separated modes sits in a low
        # probability valley and must score worse than the single-Gaussian summary.
        observation = np.array([0.0])
        means = np.array([[-2.0], [2.0]])
        sds = np.array([[0.5], [0.5]])
        pairwise_half = pairwise_crps_term(means, sds)
        mixture = mixture_crps(observation, means, sds, pairwise_half)[0]
        gaussian = gaussian_crps(observation, means.mean(), np.sqrt(0.25 + 4.0))[0]
        self.assertGreater(mixture, gaussian)

    def test_empirical_crps_matches_known_gaussian_value(self):
        rng = np.random.default_rng(11)
        samples = rng.normal(0.0, 1.0, 2000000)
        self.assertAlmostEqual(empirical_crps(0.0, samples), (np.sqrt(2) - 1) / np.sqrt(np.pi), delta=2e-3)


class ScoreDrawsTests(unittest.TestCase):
    def test_degenerate_mixture_reproduces_gaussian_score(self):
        q = np.linspace(0.01, 0.4, 30)
        simulation = np.exp(-10 * q * q)
        mean = 3 * simulation + 0.2
        sd = np.full_like(q, 0.05)
        means = np.tile(mean, (5, 1))
        sds = np.tile(sd, (5, 1))
        score = score_simulation_draws(q, simulation, q, means, sds, 3.0, 0.2)
        self.assertAlmostEqual(score["gp_crps_draws_sd_units"], (np.sqrt(2.0) - 1.0) / np.sqrt(np.pi), places=10)
        self.assertEqual(score["draws"], 5)
        self.assertEqual(score["q_points"], len(q))

    def test_bimodal_mixture_exceeds_gaussian_summary_score(self):
        # Two distant modes: the draw mixture scores a midway observation near
        # E|y - X| - 0.5 E|X - X'| ~ 1 - 0.5, while the single-Gaussian summary
        # (mean 0, wide SD) scores it far lower.
        q = np.linspace(0.01, 0.4, 20)
        simulation = np.full_like(q, 1.5)
        means = np.vstack([np.full_like(q, 0.5), np.full_like(q, 2.5)])
        sds = np.full_like(means, 0.01)
        score = score_simulation_draws(q, simulation, q, means, sds, 1.0, 0.0)
        self.assertGreater(score["gp_crps_draws"], 0.4)
        self.assertLess(score["gp_crps_draws"], 0.6)


if __name__ == "__main__":
    unittest.main()
