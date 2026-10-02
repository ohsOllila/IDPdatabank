import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch
import gptransform as gt


def setup_model(mean='cosine', prior='uniform'):
    bounds = torch.tensor([[.1, 2.5], [1., 5.], [.1, 4.], [.5, 2.], [.1, 2.], [-1., 1.], [.1, 2.], [.5, 3.]], dtype=torch.float64)
    if mean == 'zero':
        bounds = bounds[:6]
    gp = gt.GP(bounds.mean(1), bounds, 0, 1., 1, noise_model='constant', mean_fn=mean, prior_fn=prior)
    r = torch.linspace(-3, 3, 31, dtype=torch.float64)[:, None]
    q = torch.linspace(.1, 1., 7, dtype=torch.float64)[:, None]
    y = torch.sin(q)
    weights = torch.ones(31, dtype=r.dtype) * (r[1, 0] - r[0, 0])
    weights[[0, -1]] *= .5
    model = gt._NUTSModel(len(bounds), 'cpu', bounds, q[:, 0], r[:, 0], y,
        torch.cos(q @ r.T) * weights, gt.custom_cdist(r, r), gt.custom_cdist(-r, r),
        torch.eye(len(q), dtype=q.dtype), 'constant', 5, mean, 6, prior_fn=prior)
    return gp, model, r, q, y


class RegressionTests(unittest.TestCase):
    def test_installed_api(self):
        import sasgp
        self.assertIs(sasgp.GP, gt.GP)

    def test_nuts_matches_gp_likelihood_and_mean(self):
        gp, model, r, q, y = setup_model()
        expected_mean = -gp.theta[6] * torch.cos(np.pi * r / gp.theta[7]) * (abs(r / gp.theta[7]) <= 1.5)
        torch.testing.assert_close(gp.mean_r(r), expected_mean)
        expected = -(gp.NEG_LMLH_Trapz(r, q, y) + gp.log_prior()).squeeze()
        torch.testing.assert_close(model._log_lik(gp.theta_raw), expected, atol=1e-5, rtol=1e-5)

    @unittest.skipUnless(importlib.util.find_spec('pyro'), 'requires optional nuts extra')
    def test_nuts_respects_both_priors(self):
        import pyro
        for prior in ('uniform', 'gaussian'):
            gp, model, r, q, y = setup_model(prior=prior)
            raw = gp.theta_raw.detach() + .4
            trace = pyro.poutine.trace(pyro.poutine.condition(model, data={'theta_raw': raw})).get_trace()
            actual = trace.log_prob_sum() - model._log_lik(raw)
            if prior == 'uniform':
                expected = (torch.nn.functional.logsigmoid(raw) + torch.nn.functional.logsigmoid(-raw)).sum()
            else:
                expected = torch.distributions.Normal(0., 3.).log_prob(raw).sum()
            torch.testing.assert_close(actual, expected)

    def test_fit_and_predict(self):
        gp, _, r, q, y = setup_model(mean='zero')
        optimizer = torch.optim.Adam(gp.parameters(), lr=.001)
        for _ in range(2):
            optimizer.zero_grad()
            loss = gp.NEG_LMLH_Trapz(r, q, y).squeeze()
            loss.backward()
            self.assertTrue(torch.isfinite(gp.theta_raw.grad).all())
            optimizer.step()
        mean, cov = gp.predict_sq_trapz(r, q, q, y)
        self.assertEqual(mean.shape, y.shape)
        self.assertTrue(torch.isfinite(cov).all())

    def test_quadratic_noise_bounds(self):
        path = Path(__file__).resolve().parents[1] / 'membrane/run.py'
        spec = importlib.util.spec_from_file_location('membrane_run', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = dict(module.DEFAULT_CONFIG['bounds'], sigma_n_base=[.3, 7.])
        bounds = module.build_bounds(cfg, 'quadratic')
        torch.testing.assert_close(bounds[4], torch.tensor([.3, 7.]))


if __name__ == '__main__':
    unittest.main()
