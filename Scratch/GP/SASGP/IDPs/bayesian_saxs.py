"""Hyperparameter MCMC and Gaussian-mixture summaries for the SAXS GP."""
import warnings
import numpy as np
from scipy.special import ndtr
import torch
from scipy.optimize import minimize


def mixture_summary(means, variances):
    """Equal-weight Gaussian mixture: total SD and pointwise quantiles."""
    means, variances = np.asarray(means), np.maximum(variances, 0)
    mean = means.mean(axis=0)
    within = variances.mean(axis=0)
    between = means.var(axis=0)
    sd = np.sqrt(within + between)
    component_sd = np.sqrt(variances)
    quantiles = []
    for probability in (.025, .975):
        lo = np.min(means-10*component_sd, axis=0)
        hi = np.max(means+10*component_sd, axis=0)
        for _ in range(60):
            mid = (lo+hi)/2
            z = (mid-means)/np.maximum(component_sd, np.finfo(float).tiny)
            cdf = np.where(component_sd > 0, ndtr(z), mid >= means).mean(axis=0)
            lo, hi = np.where(cdf < probability, mid, lo), np.where(cdf >= probability, mid, hi)
        quantiles.append((lo+hi)/2)
    return mean, sd, np.array(quantiles), within, between


class SAXSNUTSModel:
    """Membrane _NUTSModel conventions, with the SAXS envelope and sinc operator."""
    def __init__(self, r, H, y, bounds, prior_fn):
        self.r, self.H, self.y, self.bounds = [torch.as_tensor(x, dtype=torch.float64) for x in (r, H, y, bounds)]
        self.distance2 = (self.r[:, None]-self.r[None, :])**2
        self.eye_q = torch.eye(len(y), dtype=torch.float64)
        self.prior_fn = prior_fn

    def physical(self, raw):
        return self.bounds[:, 0] + (self.bounds[:, 1]-self.bounds[:, 0])*torch.sigmoid(raw)

    def log_likelihood(self, raw):
        amplitude, ell, r0, R, noise = self.physical(raw)
        w = amplitude*(-torch.expm1(-(self.r/r0)**2))*torch.exp(-(self.r/R)**2)
        K = w[:, None]*w[None, :]*torch.exp(-self.distance2/(2*ell**2))
        C = self.H @ K @ self.H.T + self.eye_q*(noise**2+1e-12)
        C = (C+C.T)/2
        L = torch.linalg.cholesky(C)
        alpha = torch.cholesky_solve(self.y[:, None], L)[:, 0]
        return -.5*self.y@alpha-torch.log(torch.diag(L)).sum()-.5*len(self.y)*np.log(2*np.pi)

    def log_prior(self, raw):
        if self.prior_fn == 'uniform':
            return (torch.nn.functional.logsigmoid(raw)+torch.nn.functional.logsigmoid(-raw)).sum()
        return -.5*(raw/3).square().sum()

    def __call__(self):
        import pyro
        import pyro.distributions as dist
        raw = pyro.sample('theta_raw', dist.Normal(torch.zeros(5, dtype=torch.float64),
                                                   torch.full((5,), 3., dtype=torch.float64)).to_event(1))
        if self.prior_fn == 'uniform':
            normal = dist.Normal(0., 3.).log_prob(raw).sum()
            pyro.factor('prior_correction', self.log_prior(raw)-normal)
        pyro.factor('log_lik', self.log_likelihood(raw))


def sample_hyperparameters(r, H, y, bounds, initial, output_dir, scale,
                           inference='nuts', prior_fn='uniform', samples=500,
                           warmup=500, chains=2, seed=43):
    """Same raw-logit priors, NUTS or MAP/Hessian Laplace as membrane workflow."""
    if inference not in ('nuts', 'laplace') or prior_fn not in ('uniform', 'gaussian'):
        raise ValueError('Unknown inference or prior')
    if samples < 10 or warmup < 1 or chains < 1:
        raise ValueError('Require samples >= 10, warmup >= 1, chains >= 1')
    torch.set_num_threads(1)
    model = SAXSNUTSModel(r, H, y, bounds, prior_fn)
    fraction = np.clip((initial-bounds[:, 0])/(bounds[:, 1]-bounds[:, 0]), 1e-5, 1-1e-5)
    initial_raw = np.log(fraction/(1-fraction))
    def objective(raw):
        t = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
        loss = -model.log_likelihood(t)-model.log_prior(t)
        gradient, = torch.autograd.grad(loss, t)
        return loss.item(), gradient.detach().numpy()
    fits = [minimize(objective, start, jac=True, method='BFGS', options={'gtol': 1e-5})
            for start in (initial_raw, np.zeros(5))]
    best = min(fits, key=lambda f: f.fun)
    if not np.isfinite(best.fun) or np.max(np.abs(best.jac)) > 1e-3:
        raise RuntimeError(f'Raw-parameter MAP failed: {best.message}')
    info = dict(method=inference, prior_fn=prior_fn, seed=seed,
                physical_bounds=(bounds*np.array([scale, 1, 1, 1, scale])[:, None]).tolist(),
                prior='Uniform within physical bounds' if prior_fn == 'uniform' else 'Normal(0,3) on raw logit parameters',
                raw_map=best.x.tolist(), raw_map_negative_log_posterior=float(best.fun))
    if inference == 'laplace':
        center = torch.tensor(best.x, dtype=torch.float64)
        hessian = torch.autograd.functional.hessian(lambda t: -model.log_likelihood(t)-model.log_prior(t), center)
        eigenvalues = torch.linalg.eigvalsh(hessian)
        if eigenvalues.min() <= 0:
            raise RuntimeError('Laplace MAP Hessian is not positive definite; use NUTS')
        covariance = torch.linalg.inv(hessian).numpy()
        raw_chain = np.random.default_rng(seed).multivariate_normal(best.x, covariance, size=samples)[None, :, :]
        info.update(hessian_eigenvalues=eigenvalues.tolist(), approximation='Gaussian in raw space around MAP')
    else:
        import pyro
        from pyro.infer import MCMC, NUTS
        from pyro.ops.stats import effective_sample_size, split_gelman_rubin
        draws, divergences, acceptance = [], [], []
        for chain in range(chains):
            pyro.set_rng_seed(seed+chain)
            print(f'NUTS chain {chain+1}/{chains}: {warmup} warmup, {samples} samples', flush=True)
            kernel = NUTS(model, adapt_step_size=True, jit_compile=True, target_accept_prob=.9)
            start = torch.tensor(best.x + np.random.default_rng(seed+chain).normal(0, .5, 5), dtype=torch.float64)
            mcmc = MCMC(kernel, num_samples=samples, warmup_steps=warmup,
                        initial_params={'theta_raw': start}, disable_progbar=True)
            mcmc.run()
            draws.append(mcmc.get_samples()['theta_raw'])
            diagnostics = mcmc.diagnostics()
            divergences.append(len(diagnostics['divergences']['chain 0']))
            acceptance.append(float(diagnostics['acceptance rate']['chain 0']))
        stacked = torch.stack(draws)
        raw_chain = stacked.numpy()
        ess = effective_sample_size(stacked).tolist()
        rhat = split_gelman_rubin(stacked).tolist()
        passed = bool(chains >= 2 and np.isfinite(rhat).all() and max(rhat) < 1.01
                      and np.isfinite(ess).all() and min(ess) > 100 and sum(divergences) == 0)
        info.update(warmup=warmup, samples_per_chain=samples, chains=chains,
                    split_rhat=rhat, effective_sample_size=ess, divergences=divergences,
                    acceptance_rate=acceptance, diagnostic_checks_passed=passed)
        if not passed:
            warnings.warn('NUTS diagnostic checks have not passed; posterior summaries are provisional.')
    with torch.no_grad():
        physical = model.physical(torch.as_tensor(raw_chain)).numpy()
    physical_units = physical*np.array([scale, 1, 1, 1, scale])
    np.savez_compressed(output_dir/'saxs_gp_hyperparameters.npz', raw_chain=raw_chain,
                        chain=physical_units, names=np.array(['amplitude', 'ell_A', 'r0_A', 'R_A', 'noise_sigma']))
    info['parameter_quantiles_025_50_975'] = np.quantile(physical_units.reshape(-1,5), [.025,.5,.975], axis=0).tolist()
    return physical.reshape(-1, 5), info, physical_units
