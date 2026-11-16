"""Sinc-transform Gaussian-process fitting of experimental SAXS curves and
export of portable per-experiment fit files for databank quality scoring.

I(q) = background + integral P(r) sinc(qr) dr with a smooth P(r) prior
(kernel envelope w(r) = amplitude * (1 - exp(-(r/r0)^2)) * exp(-(r/R)^2),
squared and exponentially correlated). Four kernel hyperparameters plus a
constant background and a constant Gaussian observation-noise SD are sampled
jointly with NUTS (Pyro) or a MAP/Hessian Laplace approximation; the
hyperparameter-marginalized predictive distribution is summarized per q point.

The fit is stored next to the experiment as gp_fit.dat (see
fairmd.idp.saxs_gp_quality.write_gp_fit): q grid, predictive mean and SD, and
a JSON metadata line with noise/background estimates and MCMC diagnostics.
That file is the only artifact the scoring path
(fairmd.idp.saxs_gp_quality.evaluate_SAXS_quality_GP, called from
calcProperties.py) depends on, keeping scoring independent of the fitting
environment.

torch and pyro are required for fitting only; they are imported lazily so
that this module (and everything that imports it) loads without them.
Reported experimental error bars are not used anywhere; the noise level is
inferred from the data.
"""

import json
import os

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import ndtr

try:
    import torch
except ImportError:  # pragma: no cover - torch is an optional fitting dependency
    torch = None

from fairmd.idp.saxs_gp_quality import GP_FIT_FILENAME, write_gp_fit

TORCH_REQUIRED = "torch and pyro are required for GP SAXS fitting (pip install torch pyro-ppl)"


def read_saxs_dat(path):
    """Read (q, intensity) from an experimental saxs .dat file.

    q is converted to 1/A when qmax indicates 1/nm. Header/non-numeric lines
    are skipped. Exactly the preprocessing of the Scratch pipeline.
    """
    rows = []
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 2:
                continue
            try:
                rows.append((float(fields[0]), float(fields[1])))
            except ValueError:
                continue
    if len(rows) < 3:
        raise ValueError(f"too few numeric points in {path}")
    data = np.asarray(rows)
    q = data[:, 0]
    if q.max() > 2:
        q = q / 10
    order = np.argsort(q)
    return q[order], data[order, 1]


def thin_evenly(q, intensity, maximum):
    """Deterministic row thinning to control the cubic cost of GP inference."""
    if len(q) <= maximum:
        return q, intensity
    indices = np.round(np.linspace(0, len(q) - 1, maximum)).astype(int)
    return q[indices], intensity[indices]


def mixture_summary(means, variances):
    """Equal-weight Gaussian mixture: total SD and pointwise quantiles."""
    means, variances = np.asarray(means), np.maximum(variances, 0)
    mean = means.mean(axis=0)
    within = variances.mean(axis=0)
    between = means.var(axis=0)
    sd = np.sqrt(within + between)
    component_sd = np.sqrt(variances)
    quantiles = []
    for probability in (0.025, 0.975):
        lo = np.min(means - 10 * component_sd, axis=0)
        hi = np.max(means + 10 * component_sd, axis=0)
        for _ in range(60):
            mid = (lo + hi) / 2
            z = (mid - means) / np.maximum(component_sd, np.finfo(float).tiny)
            cdf = np.where(component_sd > 0, ndtr(z), mid >= means).mean(axis=0)
            lo, hi = np.where(cdf < probability, mid, lo), np.where(cdf >= probability, mid, hi)
        quantiles.append((lo + hi) / 2)
    return mean, sd, np.array(quantiles), within, between


class SAXSNUTSModel:
    """NUTS model over raw logit hyperparameters with the SAXS envelope and sinc operator."""

    def __init__(self, r, H, y, bounds, prior_fn):
        if torch is None:
            raise ImportError(TORCH_REQUIRED)
        self.r, self.H, self.y, self.bounds = [torch.as_tensor(x, dtype=torch.float64) for x in (r, H, y, bounds)]
        self.distance2 = (self.r[:, None] - self.r[None, :]) ** 2
        self.eye_q = torch.eye(len(y), dtype=torch.float64)
        self.prior_fn = prior_fn

    def physical(self, raw):
        return self.bounds[:, 0] + (self.bounds[:, 1] - self.bounds[:, 0]) * torch.sigmoid(raw)

    def log_likelihood(self, raw):
        amplitude, ell, r0, R, noise, background = self.physical(raw)
        w = amplitude * (-torch.expm1(-((self.r / r0) ** 2))) * torch.exp(-((self.r / R) ** 2))
        K = w[:, None] * w[None, :] * torch.exp(-self.distance2 / (2 * ell**2))
        C = self.H @ K @ self.H.T + self.eye_q * (noise**2 + 1e-12)
        C = (C + C.T) / 2
        L = torch.linalg.cholesky(C)
        residual = self.y - background
        alpha = torch.cholesky_solve(residual[:, None], L)[:, 0]
        return -0.5 * residual @ alpha - torch.log(torch.diag(L)).sum() - 0.5 * len(self.y) * np.log(2 * np.pi)

    def log_prior(self, raw):
        if self.prior_fn == "uniform":
            return (torch.nn.functional.logsigmoid(raw) + torch.nn.functional.logsigmoid(-raw)).sum()
        return -0.5 * (raw / 3).square().sum()

    def __call__(self):
        import pyro
        import pyro.distributions as dist

        raw = pyro.sample(
            "theta_raw",
            dist.Normal(torch.zeros(6, dtype=torch.float64), torch.full((6,), 3.0, dtype=torch.float64)).to_event(1),
        )
        if self.prior_fn == "uniform":
            normal = dist.Normal(0.0, 3.0).log_prob(raw).sum()
            pyro.factor("prior_correction", self.log_prior(raw) - normal)
        pyro.factor("log_lik", self.log_likelihood(raw))


def sample_hyperparameters(
    r,
    H,
    y,
    bounds,
    initial,
    output_dir,
    scale,
    inference="nuts",
    prior_fn="uniform",
    samples=500,
    warmup=500,
    chains=2,
    seed=43,
):
    """Infer bounded hyperparameters with NUTS or MAP/Hessian Laplace."""
    if inference not in ("nuts", "laplace") or prior_fn not in ("uniform", "gaussian"):
        raise ValueError("Unknown inference or prior")
    if samples < 10 or warmup < 1 or chains < 1:
        raise ValueError("Require samples >= 10, warmup >= 1, chains >= 1")
    if torch is None:
        raise ImportError(TORCH_REQUIRED)
    torch.set_num_threads(1)
    model = SAXSNUTSModel(r, H, y, bounds, prior_fn)
    fraction = np.clip((initial - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]), 1e-5, 1 - 1e-5)
    initial_raw = np.log(fraction / (1 - fraction))

    def objective(raw):
        t = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
        loss = -model.log_likelihood(t) - model.log_prior(t)
        (gradient,) = torch.autograd.grad(loss, t)
        return loss.item(), gradient.detach().numpy()

    fits = [
        minimize(objective, start, jac=True, method="BFGS", options={"gtol": 1e-5})
        for start in (initial_raw, np.zeros(6))
    ]
    best = min(fits, key=lambda f: f.fun)
    if not np.isfinite(best.fun) or np.max(np.abs(best.jac)) > 1e-3:
        raise RuntimeError(f"Raw-parameter MAP failed: {best.message}")
    info = dict(
        method=inference,
        prior_fn=prior_fn,
        seed=seed,
        physical_bounds=(bounds * np.array([scale, 1, 1, 1, scale, scale])[:, None]).tolist(),
        prior="Uniform within physical bounds" if prior_fn == "uniform" else "Normal(0,3) on raw logit parameters",
        raw_map=best.x.tolist(),
        raw_map_negative_log_posterior=float(best.fun),
    )
    if inference == "laplace":
        center = torch.tensor(best.x, dtype=torch.float64)
        hessian = torch.autograd.functional.hessian(lambda t: -model.log_likelihood(t) - model.log_prior(t), center)
        eigenvalues = torch.linalg.eigvalsh(hessian)
        if eigenvalues.min() <= 0:
            raise RuntimeError("Laplace MAP Hessian is not positive definite; use NUTS")
        covariance = torch.linalg.inv(hessian).numpy()
        raw_chain = np.random.default_rng(seed).multivariate_normal(best.x, covariance, size=samples)[None, :, :]
        info.update(hessian_eigenvalues=eigenvalues.tolist(), approximation="Gaussian in raw space around MAP")
    else:
        import pyro
        from pyro.infer import MCMC, NUTS
        from pyro.ops.stats import effective_sample_size, split_gelman_rubin

        draws, divergences, acceptance = [], [], []
        for chain in range(chains):
            pyro.set_rng_seed(seed + chain)
            print(f"NUTS chain {chain + 1}/{chains}: {warmup} warmup, {samples} samples", flush=True)
            kernel = NUTS(model, adapt_step_size=True, jit_compile=True, target_accept_prob=0.9)
            start = torch.tensor(best.x + np.random.default_rng(seed + chain).normal(0, 0.5, 6), dtype=torch.float64)
            mcmc = MCMC(
                kernel,
                num_samples=samples,
                warmup_steps=warmup,
                initial_params={"theta_raw": start},
                disable_progbar=True,
            )
            mcmc.run()
            draws.append(mcmc.get_samples()["theta_raw"])
            diagnostics = mcmc.diagnostics()
            divergences.append(len(diagnostics["divergences"]["chain 0"]))
            acceptance.append(float(diagnostics["acceptance rate"]["chain 0"]))
        stacked = torch.stack(draws)
        raw_chain = stacked.numpy()
        ess = effective_sample_size(stacked).tolist()
        rhat = split_gelman_rubin(stacked).tolist()
        passed = bool(
            chains >= 2
            and np.isfinite(rhat).all()
            and max(rhat) < 1.01
            and np.isfinite(ess).all()
            and min(ess) > 100
            and sum(divergences) == 0
        )
        info.update(
            warmup=warmup,
            samples_per_chain=samples,
            chains=chains,
            split_rhat=rhat,
            effective_sample_size=ess,
            divergences=divergences,
            acceptance_rate=acceptance,
            diagnostic_checks_passed=passed,
        )
        if not passed:
            import warnings

            warnings.warn(
                "NUTS diagnostic checks have not passed; posterior summaries are provisional.", stacklevel=2
            )
    with torch.no_grad():
        physical = model.physical(torch.as_tensor(raw_chain)).numpy()
    physical_units = physical * np.array([scale, 1, 1, 1, scale, scale])
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_dir, "saxs_gp_hyperparameters.npz"),
        raw_chain=raw_chain,
        chain=physical_units,
        names=np.array(["amplitude", "ell_A", "r0_A", "R_A", "noise_sigma", "background"]),
    )
    info["parameter_quantiles_025_50_975"] = np.quantile(
        physical_units.reshape(-1, 6), [0.025, 0.5, 0.975], axis=0
    ).tolist()
    return physical.reshape(-1, 6), info, physical_units


def gyration_samples(r, weights, mean, covariance, samples=50000, seed=42):
    """Propagate the joint GP covariance to M0, M2 and Rg in nm.

    Guinier: I(q)/I(0) = 1 - q^2 Rg^2/3 + O(q^4).
    Expanding sinc gives Rg^2 = M2/(2 M0). Sample the two linear
    moments jointly, exactly equivalent to integrating full GP draws.
    """
    operator = np.vstack((weights, weights * (r / 10) ** 2))
    moment_mean = operator @ mean
    moment_cov = operator @ covariance @ operator.T
    moment_cov = (moment_cov + moment_cov.T) / 2
    eigenvalues, vectors = np.linalg.eigh(moment_cov)
    tolerance = 1e-10 * max(np.max(np.abs(eigenvalues)), np.finfo(float).tiny)
    if eigenvalues.min() < -tolerance:
        raise ValueError("Moment covariance is not positive semidefinite")
    draws = (
        moment_mean
        + np.random.default_rng(seed).standard_normal((samples, 2)) @ (vectors * np.sqrt(np.maximum(eigenvalues, 0))).T
    )
    valid = (draws[:, 0] > 0) & (draws[:, 1] >= 0)
    rg = np.full(samples, np.nan)
    rg[valid] = np.sqrt(draws[valid, 1] / (2 * draws[valid, 0]))
    return draws, rg, valid


def fit_saxs(
    q,
    intensity,
    output_dir,
    r_max=400.0,
    points=401,
    inference="nuts",
    prior_fn="uniform",
    samples=500,
    warmup=500,
    chains=2,
    seed=43,
    reference_rg_nm=None,
    guinier_q_max=None,
    make_plots=True,
):
    """Fit intensities with inferred constant background and observation noise.

    P includes the intensity scale: I(q) = integral P(r) sinc(qr) dr.
    Zero mean, no reflection, no positivity or normalization constraint.
    Intervals marginalize kernel hyperparameters using NUTS or Laplace.
    Writes the full diagnostic artifact set plus saxs_gp_intensity.csv into
    output_dir and returns the fit report. matplotlib is optional (plots are
    skipped with a warning when unavailable or make_plots is False).
    """
    output_dir = str(output_dir)
    q, y = np.asarray(q, float), np.asarray(intensity, float)
    if (
        q.ndim != 1
        or y.shape != q.shape
        or len(q) < 2
        or not np.isfinite(q).all()
        or not np.isfinite(y).all()
        or np.any(q < 0)
        or np.any(np.diff(q) <= 0)
        or not np.isfinite(r_max)
        or r_max <= 0
        or points < 51
    ):
        raise ValueError("Require sorted nonnegative q, finite data, positive r_max and >=51 grid points")
    r = np.linspace(0, r_max, points)
    dr = r[1]
    weights = np.full(points, dr)
    weights[[0, -1]] *= 0.5
    H = np.sinc(np.outer(q, r) / np.pi) * weights
    scale = float(np.max(np.abs(y)))
    if scale <= 0:
        raise ValueError("Intensity scale must be nonzero")
    yn = y / scale
    # Used only for optimizer initialization; noise is inferred from the likelihood.
    sn = np.clip(np.median(np.abs(np.diff(yn) - np.median(np.diff(yn)))) / 0.67449 / np.sqrt(2), 1e-5, 0.5)
    distance2 = (r[:, None] - r[None, :]) ** 2
    # R <= r_max/4 makes the envelope at the integration boundary <= exp(-16).
    bounds = np.log(
        [
            (1e-5 / r_max, 100 / r_max),
            (2 * dr, r_max / 2),
            (2 * dr, r_max / 4),
            (4 * dr, r_max / 4),
            (1e-6, 1.0),
            (1e-8, 0.5),
        ]
    )

    def covariance(logtheta):
        amplitude, ell, r0, R = np.exp(logtheta[:4])
        w = amplitude * (-np.expm1(-((r / r0) ** 2))) * np.exp(-((r / R) ** 2))
        return w[:, None] * w[None, :] * np.exp(-distance2 / (2 * ell**2)), w

    def objective(theta):
        K, _ = covariance(theta)
        C = H @ K @ H.T + (np.exp(theta[4]) ** 2 + 1e-12) * np.eye(len(q))
        factor = cho_factor(C, lower=True)
        residual = yn - np.exp(theta[5])
        return (
            0.5 * residual @ cho_solve(factor, residual)
            + np.log(np.diag(factor[0])).sum()
            + 0.5 * len(q) * np.log(2 * np.pi)
        )

    tail = np.clip(np.median(yn[-max(3, len(yn) // 10) :]), 1e-7, 0.49)
    starts = [[10 / r_max, r_max / 20, r_max / 25, r_max * f, sn, tail] for f in (0.10, 0.18, 0.24)]
    fits = [
        minimize(
            objective,
            np.clip(np.log(start), bounds[:, 0], bounds[:, 1]),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 150},
        )
        for start in starts
    ]
    valid = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not valid:
        raise RuntimeError("GP optimization failed: " + "; ".join(str(f.message) for f in fits))
    best = min(valid, key=lambda fit: fit.fun)
    os.makedirs(output_dir, exist_ok=True)
    theta_draws, inference_info, physical_chain = sample_hyperparameters(
        r,
        H,
        yn,
        np.exp(bounds),
        np.exp(best.x),
        output_dir,
        scale,
        inference=inference,
        prior_fn=prior_fn,
        samples=samples,
        warmup=warmup,
        chains=chains,
        seed=seed,
    )
    r_means, r_variances, q_means, q_variances = [], [], [], []
    moment_draws, rg_draws, valid_draws = [], [], []
    cov_sum = np.zeros((points, points))
    prior_variance = np.zeros(points)
    for index, theta in enumerate(theta_draws):
        K, w = covariance(np.log(theta))
        cross = K @ H.T
        C = H @ cross + (theta[4] ** 2 + 1e-12) * np.eye(len(q))
        factor = cho_factor(C, lower=True)
        mean = cross @ cho_solve(factor, yn - theta[5]) * scale
        posterior = (K - cross @ cho_solve(factor, cross.T)) * scale**2
        posterior = (posterior + posterior.T) / 2
        r_means.append(mean)
        r_variances.append(np.maximum(np.diag(posterior), 0))
        q_means.append(H @ mean + theta[5] * scale)
        q_variances.append(np.maximum(np.einsum("ij,ij->i", H @ posterior, H), 0))
        cov_sum += posterior
        prior_variance += (w * scale) ** 2
        moments, rg, rg_valid = gyration_samples(r, weights, mean, posterior, samples=50, seed=seed + index + 1)
        moment_draws.append(moments)
        rg_draws.append(rg)
        valid_draws.append(rg_valid)
    mean, sd, r_interval, r_within, r_between = mixture_summary(r_means, r_variances)
    predicted, intensity_sd, q_interval, q_within, q_between = mixture_summary(q_means, q_variances)
    r_means = np.asarray(r_means)
    total_covariance = cov_sum / len(theta_draws) + (r_means - mean).T @ (r_means - mean) / len(theta_draws)
    np.savez_compressed(
        os.path.join(output_dir, "saxs_gp_covariance.npz"),
        r_A=r,
        q_per_A=q,
        covariance_r=total_covariance,
        covariance_q=H @ total_covariance @ H.T,
        within_variance_r=r_within,
        between_variance_r=r_between,
        within_variance_q=q_within,
        between_variance_q=q_between,
    )
    np.savetxt(
        os.path.join(output_dir, "saxs_gp_real_space.csv"),
        np.column_stack((r, mean, sd, np.sqrt(prior_variance / len(theta_draws)), r_interval.T)),
        delimiter=",",
        header="r_A,P_mean,P_sd,hyperposterior_envelope_rms,P_lower_95,P_upper_95",
        comments="",
    )
    np.savetxt(
        os.path.join(output_dir, "saxs_gp_intensity.csv"),
        np.column_stack(
            (
                q,
                y,
                predicted,
                intensity_sd,
                np.sqrt(intensity_sd**2 + np.mean((theta_draws[:, 4] * scale) ** 2)),
                q_interval.T,
            )
        ),
        delimiter=",",
        header="q_per_A,noisy_I,I_mean,I_latent_sd,I_predictive_sd,I_lower_95,I_upper_95",
        comments="",
    )
    amplitude, ell, r0, R, noise_sigma, background = np.median(theta_draws, axis=0)
    moments = np.concatenate(moment_draws)
    rg = np.concatenate(rg_draws)
    rg_valid = np.concatenate(valid_draws)
    rg_quantiles = np.quantile(rg[rg_valid], [0.025, 0.5, 0.975]) if rg_valid.any() else None
    np.savetxt(
        os.path.join(output_dir, "saxs_gp_rg_samples.csv"),
        np.column_stack((moments, rg, rg_valid.astype(int), np.repeat(np.arange(len(theta_draws)), 50))),
        delimiter=",",
        header="I0_au,M2_au_nm2,Rg_nm,valid_moments,hyperparameter_draw",
        comments="",
    )
    report = dict(
        amplitude=float(amplitude * scale),
        ell_A=float(ell),
        r0_A=float(r0),
        R_A=float(R),
        r_max_A=float(r_max),
        grid_points=points,
        noise_sigma=float(noise_sigma * scale),
        noise_model="Inferred constant Gaussian observation SD",
        background=float(background * scale),
        background_interval_95=(np.quantile(theta_draws[:, 5], [0.025, 0.975]) * scale).tolist(),
        noise_sigma_interval_95=(np.quantile(theta_draws[:, 4], [0.025, 0.975]) * scale).tolist(),
        negative_log_marginal_likelihood_scaled=float(best.fun),
        optimizer_message=str(best.message),
        successful_starts=len(valid),
        mle_near_bound_parameters=[
            name
            for name, x, (lo, hi) in zip(  # noqa: B905 -- all inputs contain the six model parameters
                ["amplitude", "ell", "r0", "R", "noise_sigma", "background"], best.x, bounds
            )
            if min(x - lo, hi - x) < 0.01
        ],
        inference=inference_info,
        parameter_summary="Posterior medians",
        intervals="Pointwise equal-tailed hyperparameter-mixture intervals; no positivity constraint",
        mean="Inferred constant background",
        transform="I(q) = background + integral P(r) sinc(qr) dr; sinc(x)=sin(x)/x",
    )
    report["radius_of_gyration"] = dict(
        formula="Rg^2 = integral r^2 P(r) dr / (2 integral P(r) dr)",
        median_nm=float(rg_quantiles[1]) if rg_quantiles is not None else None,
        interval_95_nm=rg_quantiles[[0, 2]].tolist() if rg_quantiles is not None else None,
        draws=len(rg),
        hyperparameter_draws=len(theta_draws),
        conditional_draws_per_hyperparameter=50,
        seed=seed,
        invalid_moment_fraction=float(1 - rg_valid.mean()),
        uncertainty=(
            "Hyperparameter-marginalized joint GP moments, conditional on valid moments; "
            "no pointwise positivity constraint"
        ),
    )
    with open(os.path.join(output_dir, "saxs_gp_fit.json"), "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if make_plots:
        try:
            _plot_fit(
                q,
                y,
                predicted,
                q_interval,
                r,
                mean,
                r_interval,
                rg,
                rg_valid,
                rg_quantiles,
                inference,
                inference_info,
                physical_chain,
                reference_rg_nm,
                guinier_q_max,
                total_covariance,
                H,
                output_dir,
                seed,
            )
        except ImportError:
            import warnings

            warnings.warn("matplotlib not available; skipping GP fit plots", stacklevel=2)
    return report


def _plot_fit(
    q,
    y,
    predicted,
    q_interval,
    r,
    mean,
    r_interval,
    rg,
    rg_valid,
    rg_quantiles,
    inference,
    inference_info,
    physical_chain,
    reference_rg_nm,
    guinier_q_max,
    total_covariance,
    H,
    output_dir,
    seed,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    axes[0].plot(q, y, ".", alpha=0.5, label="Observed intensity")
    axes[0].plot(q, predicted, label="FTGP mean")
    axes[0].fill_between(q, q_interval[0], q_interval[1], alpha=0.25, label="95% latent interval")
    axes[0].set(xlabel="q (Å⁻¹)", ylabel="Intensity (a.u.)", title="SAXS fit")
    # Convert the density as well as distance to preserve its integral.
    axes[1].plot(r / 10, mean * 10, label="FTGP mean")
    axes[1].fill_between(r / 10, r_interval[0] * 10, r_interval[1] * 10, alpha=0.25, label="95% interval")
    axes[1].set(
        xlabel="Pair separation r (nm)", ylabel="P(r) (a.u. / nm)", xlim=(0, 25), title="Pair-distance distribution"
    )
    if rg_quantiles is not None:
        axes[2].hist(rg[rg_valid], bins=70, density=True, alpha=0.65, label="P(r) moments: posterior draws")
        axes[2].axvspan(rg_quantiles[0], rg_quantiles[2], alpha=0.15, color="C1", label="P(r) moments: 95% interval")
        axes[2].axvline(rg_quantiles[1], color="C1", label=f"P(r) median {rg_quantiles[1]:.3f} nm")
    else:
        axes[2].text(0.5, 0.5, "No valid Rg draws", ha="center", transform=axes[2].transAxes)
    axes[2].set(
        xlabel="Radius of gyration (nm)",
        ylabel="Probability density (nm⁻¹)",
        title=f"Rg uncertainty ({inference.upper()})",
    )
    if guinier_q_max is not None:
        from guinier_uq import guinier

        guinier_result, *_ = guinier(q, predicted, H @ total_covariance @ H.T, guinier_q_max)
        center, width = guinier_result["Rg_nm"], guinier_result["Rg_sd_nm"]
        gx = np.linspace(center - 4 * width, center + 4 * width, 401)
        density = np.exp(-0.5 * ((gx - center) / width) ** 2) / (width * np.sqrt(2 * np.pi))
        axes[2].plot(gx, density, color="C2", linewidth=2, label=f"Guinier 1st-order UQ: {center:.3f} nm")
        axes[2].axvspan(
            *guinier_result["interval_95_nm"], color="C2", alpha=0.12, label=f"Guinier 95% (q ≤ {guinier_q_max:g} Å⁻¹)"
        )
        axes[2].set_title("Rg: moment propagation vs Guinier")
    if reference_rg_nm is not None:
        axes[2].axvline(
            reference_rg_nm,
            color="C3",
            linestyle="--",
            linewidth=1.8,
            label=f"Guinier reference {reference_rg_nm:g} nm",
        )
    if not rg_valid.all():
        axes[2].text(0.02, 0.95, f"Invalid moments: {1 - rg_valid.mean():.1%}", va="top", transform=axes[2].transAxes)
    if inference == "nuts" and not inference_info["diagnostic_checks_passed"]:
        fig.suptitle("Provisional: NUTS diagnostic checks have not passed")
    for ax in axes:
        ax.axhline(0, color="0.5", lw=0.7)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(os.path.join(output_dir, "saxs_gp_fit.png"), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(6, 2, figsize=(11, 14), constrained_layout=True)
    for j, label in enumerate(
        ["Amplitude (a.u. / Å)", "ell (Å)", "r0 (Å)", "R (Å)", "Noise SD (a.u.)", "Background (a.u.)"]
    ):
        for c, chain_values in enumerate(physical_chain):
            axes[j, 0].plot(chain_values[:, j], alpha=0.7, lw=0.6, label=f"Chain {c + 1}")
        axes[j, 0].set(ylabel=label, xlabel="Posterior draw")
        axes[j, 1].hist(physical_chain[:, :, j].ravel(), bins=40, density=True)
        axes[j, 1].set(xlabel=label, ylabel="Density")
    axes[0, 0].legend()
    fig.suptitle(f"{inference.upper()} hyperparameter posterior")
    fig.savefig(os.path.join(output_dir, "saxs_gp_hyperparameters.png"), dpi=150)
    plt.close(fig)


def export_gp_fit(fit_dir, experiment_dir):
    """Write the portable gp_fit.dat for one experiment from a finished fit.

    fit_dir holds the full artifact set (saxs_gp_intensity.csv,
    saxs_gp_fit.json); experiment_dir is the experiment's databank folder
    (Data/Experiments/saxs/<id>), which receives gp_fit.dat. This file is the
    only thing needed later for quality scoring in calcProperties.py.
    """
    fit = np.genfromtxt(os.path.join(fit_dir, "saxs_gp_intensity.csv"), delimiter=",", names=True)
    with open(os.path.join(fit_dir, "saxs_gp_fit.json")) as handle:
        report = json.load(handle)
    inference = report.get("inference", {}) if isinstance(report, dict) else {}
    metadata = {
        "noise_sigma": report.get("noise_sigma"),
        "background": report.get("background"),
        "noise_sigma_interval_95": report.get("noise_sigma_interval_95"),
        "background_interval_95": report.get("background_interval_95"),
        "r_max_A": report.get("r_max_A"),
        "grid_points": report.get("grid_points"),
        "nuts_diagnostics_passed": inference.get("diagnostic_checks_passed") if isinstance(inference, dict) else None,
        "split_rhat": inference.get("split_rhat") if isinstance(inference, dict) else None,
        "effective_sample_size": inference.get("effective_sample_size") if isinstance(inference, dict) else None,
        "divergences": inference.get("divergences") if isinstance(inference, dict) else None,
        "source_pipeline": "fairmd.idp.saxs_gp_fit",
        "model": report.get("transform"),
    }
    write_gp_fit(
        os.path.join(experiment_dir, GP_FIT_FILENAME),
        fit["q_per_A"],
        fit["I_mean"],
        fit["I_predictive_sd"],
        fit["I_latent_sd"],
        fit["I_lower_95"],
        fit["I_upper_95"],
        metadata,
    )
