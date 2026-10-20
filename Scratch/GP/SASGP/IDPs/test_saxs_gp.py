"""Run with: MPLCONFIGDIR=IDPs/.mplconfig IDPs/.venv/bin/python -m unittest discover -s IDPs -p 'test_*.py'."""
import unittest
import numpy as np
from saxs_gp import gyration_samples
from gp_quality import gaussian_crps, q_weights, score_simulation


class GyrationTests(unittest.TestCase):
    def test_gaussian_distribution_and_units(self):
        r = np.linspace(0, 100, 501)  # Å
        weights = np.full(len(r), r[1]); weights[[0, -1]] *= .5
        sigma = 10.  # Å; P(r) proportional to r² exp(-r²/(2 sigma²))
        mean = r**2*np.exp(-r**2/(2*sigma**2))
        _, rg, valid = gyration_samples(r, weights, mean, np.zeros((len(r), len(r))), samples=10)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(rg, np.sqrt(1.5)*sigma/10, rtol=1e-10)

    def test_correlated_amplitude_cancels_in_ratio(self):
        r = np.array([0., 10., 20.])
        weights = np.ones(3)
        mean = np.array([0., 2., 1.])
        covariance = .001*np.outer(mean, mean)
        moments, rg, valid = gyration_samples(r, weights, mean, covariance)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(rg, 1., atol=1e-8)
        operator = np.vstack((weights, weights*(r/10)**2))
        np.testing.assert_allclose(np.cov(moments.T), operator@covariance@operator.T, rtol=.025)

    def test_invalid_moments_are_reported(self):
        _, rg, valid = gyration_samples(np.array([0., 10.]), np.ones(2),
                                        -np.ones(2), np.zeros((2, 2)), samples=10)
        self.assertFalse(valid.any())
        self.assertTrue(np.isnan(rg).all())


class BayesianTests(unittest.TestCase):
    def test_total_variance_and_mixture_quantiles(self):
        from bayesian_saxs import mixture_summary
        mean, sd, interval, within, between = mixture_summary(
            np.array([[-1., 0.], [1., 0.]]), np.array([[4., 0.], [4., 0.]]))
        np.testing.assert_allclose(mean, [0, 0])
        np.testing.assert_allclose(sd**2, [5, 0])
        np.testing.assert_allclose(within, [4, 0])
        np.testing.assert_allclose(between, [1, 0])
        np.testing.assert_allclose(interval[0], -interval[1], atol=1e-10)

    def test_torch_likelihood_and_gradient_match_numpy(self):
        import torch
        from scipy.linalg import cho_factor, cho_solve
        from bayesian_saxs import SAXSNUTSModel
        r = np.linspace(0, 100, 101)
        q = np.linspace(0, .4, 15)
        weights = np.ones(len(r)); weights[[0,-1]] *= .5
        H = np.sinc(np.outer(q,r)/np.pi)*weights
        y = np.exp(-q*q*100)
        bounds = np.array([[.001,.1], [2,100], [2,25], [4,25], [.001,.1]])
        model = SAXSNUTSModel(r,H,y,bounds,'uniform')
        raw = torch.tensor([-.5,.2,.1,-.3,.1], dtype=torch.float64, requires_grad=True)
        a,ell,r0,R,noise = model.physical(raw).detach().numpy()
        w = a*(-np.expm1(-(r/r0)**2))*np.exp(-(r/R)**2)
        K = np.outer(w,w)*np.exp(-(r[:,None]-r[None,:])**2/(2*ell**2))
        C = H@K@H.T+(noise**2+1e-12)*np.eye(len(q))
        factor = cho_factor(C, lower=True)
        expected = -.5*y@cho_solve(factor,y)-np.log(np.diag(factor[0])).sum()-.5*len(q)*np.log(2*np.pi)
        self.assertAlmostEqual(model.log_likelihood(raw).item(), expected, places=9)
        self.assertTrue(torch.autograd.gradcheck(model.log_likelihood, (raw,)))
        # Uniform physical prior has exactly the sigmoid Jacobian in raw space.
        sigmoid = torch.sigmoid(raw)
        torch.testing.assert_close(model.log_prior(raw), torch.log(sigmoid*(1-sigmoid)).sum())
        model.prior_fn = 'gaussian'
        torch.testing.assert_close(model.log_prior(raw), -.5*(raw/3).square().sum())


class QualityScoreTests(unittest.TestCase):
    def test_q_weights_integrate_linear_function(self):
        q = np.array([0., .1, .4, 1.])
        np.testing.assert_allclose(np.sum(q_weights(q) * q), .5)

    def test_crps_prefers_closer_value_and_penalizes_width(self):
        self.assertLess(gaussian_crps(0., 0., 1.), gaussian_crps(2., 0., 1.))
        self.assertLess(gaussian_crps(0., 0., .5), gaussian_crps(0., 0., 2.))

    def test_affine_equivalent_curve_scores_as_covered(self):
        q = np.linspace(.01, .4, 30)
        simulation = np.exp(-10*q*q)
        mean = 3*simulation + .2
        score, detail = score_simulation(q, simulation, q, mean, np.full_like(q, .05))
        self.assertAlmostEqual(score["scale_sim_to_experiment"], 3., places=10)
        self.assertAlmostEqual(score["background"], .2, places=10)
        self.assertAlmostEqual(score["gp_coverage_95"], 1.)
        self.assertAlmostEqual(score["gp_crps_sd_units"],
                               (np.sqrt(2.) - 1.) / np.sqrt(np.pi), places=10)
        np.testing.assert_allclose(detail["z"], 0., atol=1e-10)


if __name__ == '__main__':
    unittest.main()
