"""CRPS against the full hyperparameter-draw mixture instead of a single Gaussian.

The Gaussian pipeline collapses the NUTS hyperparameter posterior to one
predictive mean and one predictive SD per q point before scoring. Here the
predictive marginal is kept as the equal-weight Gaussian mixture implied by
the sampled draws, and the CRPS of the fitted simulation is evaluated against
that mixture analytically:

    CRPS(F, y) = E|y - X| - 0.5 E|X - X'|,   X ~ F = (1/N) sum_i N(m_i, s_i)

with E|y - X_i| and E|X_i - X_j| both closed-form Gaussians. This is the exact
expectation of the empirical draw-based CRPS estimator and captures skew and
multimodality of the hyperparameter posterior that the single-Gaussian
summary discards. Pairwise terms E|X_i - X_j| do not depend on the scored
simulation, so they are computed once per experiment and reused.
"""

import json
import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr

from gp_quality import gaussian_crps, q_weights

PARAMETER_NAMES = ("amplitude", "ell_A", "r0_A", "R_A", "noise_sigma", "background")


def thin_evenly(values, maximum):
    """Evenly spaced rows, deterministically, for mixture and pairwise cost control."""
    if len(values) <= maximum:
        return values
    indices = np.unique(np.round(np.linspace(0, len(values) - 1, maximum)).astype(int))
    return values[indices]


def hyperparameter_draws(experiment_dir):
    """Physical-unit draws (n, 6): amplitude, ell, r0, R, noise_sigma, background.

    Fits saved before the background parameter was introduced have five
    columns; their background is zero.
    """
    data = np.load(experiment_dir / "saxs_gp_hyperparameters.npz")
    chain = np.asarray(data["chain"])
    flat = chain.reshape(-1, chain.shape[-1])
    names = [str(name) for name in data["names"]]
    if set(names) != {name for name in PARAMETER_NAMES if name != "background"} and set(names) != set(PARAMETER_NAMES):
        raise ValueError(f"unexpected hyperparameters {names} in {experiment_dir}")
    columns = [flat[:, names.index(name)] if name in names else np.zeros(len(flat)) for name in PARAMETER_NAMES]
    return np.column_stack(columns)


def mixture_components(experiment_dir, max_draws=1000, recompute=False, mean_tolerance=1e-4):
    """Per-draw predictive means and SDs on the fitted q grid.

    Reconstructs the conditional GP posterior for every saved hyperparameter
    draw, exactly mirroring the fit-time mixture, and validates its mean
    against saxs_gp_intensity.csv. Results are cached in
    saxs_gp_mixture_components.npz per experiment.
    """
    from pathlib import Path

    experiment_dir = Path(experiment_dir)
    cache = experiment_dir / "saxs_gp_mixture_components.npz"
    if not recompute and cache.exists():
        data = np.load(cache)
        return data["q_per_A"], data["component_mean"], data["component_predictive_sd"]

    covariance = np.load(experiment_dir / "saxs_gp_covariance.npz")
    r, q = covariance["r_A"], covariance["q_per_A"]
    report = json.loads((experiment_dir / "saxs_gp_fit.json").read_text())
    if report["grid_points"] != len(r):
        raise ValueError(f"r grid in {experiment_dir} does not match the saved fit report")
    dr = r[1]
    weights = np.full(len(r), dr)
    weights[[0, -1]] *= 0.5
    operator = np.sinc(np.outer(q, r) / np.pi) * weights
    observed = np.genfromtxt(experiment_dir / "saxs_gp_intensity.csv", delimiter=",", names=True)["noisy_I"]
    if len(observed) != len(q):
        raise ValueError(f"q grid in {experiment_dir} does not match the saved intensity table")
    draws = thin_evenly(hyperparameter_draws(experiment_dir), max_draws)
    distance2 = (r[:, None] - r[None, :]) ** 2
    identity = np.eye(len(q))
    means, predictive_sd, latent_variances = [], [], []
    for amplitude, ell, r0, radius, noise, background in draws:
        envelope = amplitude * (-np.expm1(-((r / r0) ** 2))) * np.exp(-((r / radius) ** 2))
        kernel = np.outer(envelope, envelope) * np.exp(-distance2 / (2 * ell**2))
        cross = kernel @ operator.T
        factor = cho_factor(operator @ cross + (noise**2 + 1e-12) * identity, lower=True)
        solve = cho_solve(factor, observed - background)
        means.append(operator @ (cross @ solve) + background)
        posterior = kernel - cross @ cho_solve(factor, cross.T)
        posterior = (posterior + posterior.T) / 2
        latent_variance = np.maximum(np.einsum("ij,ij->i", operator @ posterior, operator), 0)
        latent_variances.append(latent_variance)
        predictive_sd.append(np.sqrt(latent_variance + noise**2))
    means = np.asarray(means)
    predictive_sd = np.asarray(predictive_sd)
    mixture_mean = means.mean(axis=0)
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(predictive_sd)):
        raise ValueError(f"non-finite reconstructed components in {experiment_dir}")
    saved = np.genfromtxt(experiment_dir / "saxs_gp_intensity.csv", delimiter=",", names=True)
    relative_error = np.max(
        np.abs(mixture_mean - saved["I_mean"]) / np.maximum(np.abs(saved["I_mean"]), np.finfo(float).tiny)
    )
    latent_sd = np.sqrt(np.asarray(latent_variances).mean(axis=0) + means.var(axis=0))
    sd_relative_error = np.max(
        np.abs(latent_sd - saved["I_latent_sd"]) / np.maximum(saved["I_latent_sd"], np.finfo(float).tiny)
    )
    if relative_error > mean_tolerance or sd_relative_error > mean_tolerance:
        warnings.warn(
            f"Reconstructed mixture in {experiment_dir} deviates from the saved saxs_gp_intensity.csv "
            f"(max relative mean error {relative_error:.3g}, latent SD error {sd_relative_error:.3g}); "
            "the cached fit may use a different model.",
            stacklevel=2,
        )
    np.savez_compressed(cache, q_per_A=q, component_mean=means, component_predictive_sd=predictive_sd)
    return q, means, predictive_sd


def pairwise_crps_term(means, predictive_sd):
    """Half of E|X - X'| per q point for the equal-weight predictive mixture.

    Precomputable once per experiment because it does not involve the scored
    simulation. Memory is bounded by chunking the leading draw index.
    """
    means = np.asarray(means, float)
    predictive_sd = np.asarray(predictive_sd, float)
    if means.ndim != 2 or means.shape != predictive_sd.shape or len(means) < 1:
        raise ValueError("means and predictive_sd must share a (draws, q points) shape")
    n_points = means.shape[1]
    term = np.empty(n_points)
    n = len(means)
    chunk = max(1, int(2e7 // max(n, 1)))
    for point in range(n_points):
        m, s = means[:, point], predictive_sd[:, point]
        total = 0.0
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            diff = m[start:stop, None] - m[None, :]
            sigma = np.hypot(s[start:stop, None], s[None, :])
            total += np.sum(
                sigma * np.sqrt(2 / np.pi) * np.exp(-0.5 * (diff / sigma) ** 2) + diff * (2 * ndtr(diff / sigma) - 1)
            ) / (n * n)
        term[point] = total / 2
    return term


def mixture_crps(observation, means, predictive_sd, pairwise_half):
    """Pointwise CRPS of observations under the equal-weight Gaussian mixture.

    observation: (q points,), means/predictive_sd: (draws, q points),
    pairwise_half: output of pairwise_crps_term for the same mixture.
    """
    observation = np.asarray(observation, float)
    if observation.shape != means.shape[1:] or observation.shape != predictive_sd.shape[1:]:
        raise ValueError("observation must match the mixture q grid")
    component = gaussian_crps(observation[None, :], means, predictive_sd).mean(axis=0)
    return component + predictive_sd.mean(axis=0) / np.sqrt(np.pi) - pairwise_half


def empirical_crps(observation, samples):
    """Empirical CRPS of a scalar against posterior samples; used by tests."""
    observation = np.asarray(observation, float)
    samples = np.asarray(samples, float).ravel()
    if samples.ndim != 1 or len(samples) < 2:
        raise ValueError("samples must be one-dimensional with at least two entries")
    n = len(samples)
    # (1/n^2) sum over ordered pairs = 2 sum_{i<j} |x_i - x_j| / n^2, and the
    # sorted-weight identity gives sum_{i<j} |x_i - x_j| directly.
    first = np.mean(np.abs(observation - samples))
    ordered = np.sort(samples)
    summed_pairs = np.sum((2 * np.arange(1, n + 1) - n - 1) * ordered) / (n * n)
    return first - summed_pairs


def score_simulation_draws(
    q_sim, intensity_sim, q_gp, component_means, component_predictive_sd, scale, background, sd_floor_fraction=1e-6
):
    """Score a simulation against the draw mixture, reusing the Gaussian-pipeline fit.

    The scale and background fitted by score_simulation against the mixture
    mean remain optimal here because the mixture mean is unchanged. Aggregation
    follows gp_quality: trapezoid q weights, CRPS in intensity, normalized, and
    predictive-SD units.
    """
    q_sim, intensity_sim, q_gp = (np.asarray(value, float) for value in (q_sim, intensity_sim, q_gp))
    component_means = np.asarray(component_means, float)
    component_predictive_sd = np.asarray(component_predictive_sd, float)
    for name, values in (("q_sim", q_sim), ("intensity_sim", intensity_sim), ("q_gp", q_gp)):
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite and one-dimensional")
    if component_means.shape != component_predictive_sd.shape or component_means.ndim != 2:
        raise ValueError("component arrays must share a (draws, q points) shape")
    if len(q_gp) != component_means.shape[1]:
        raise ValueError("q_gp must match the mixture q grid")
    if np.any(np.diff(q_sim) <= 0) or np.any(np.diff(q_gp) <= 0):
        raise ValueError("q arrays must be strictly increasing")
    overlap = (q_gp >= q_sim[0]) & (q_gp <= q_sim[-1])
    if overlap.sum() < 3:
        raise ValueError("simulation and GP need at least three overlapping q points")
    q = q_gp[overlap]
    means = component_means[:, overlap]
    predictive_sd = component_predictive_sd[:, overlap]
    simulation = np.interp(q, q_sim, intensity_sim)
    fitted = scale * simulation + background
    scale_reference = max(float(np.max(np.abs(means.mean(axis=0)))), np.finfo(float).tiny)
    predictive_sd = np.maximum(predictive_sd, sd_floor_fraction * scale_reference)
    weights = q_weights(q)
    mixture_mean = means.mean(axis=0)
    normalization = float(np.sum(weights * np.abs(mixture_mean)))
    if normalization <= np.finfo(float).tiny:
        raise ValueError("GP mean has zero intensity scale")
    pairwise_half = pairwise_crps_term(means, predictive_sd)
    crps = mixture_crps(fitted, means, predictive_sd, pairwise_half)
    result = {
        "gp_crps_draws": float(np.sum(weights * crps)),
        "gp_crps_draws_normalized": float(np.sum(weights * crps) / normalization),
        "gp_crps_draws_sd_units": float(np.sum(weights * crps) / np.sum(weights * predictive_sd.mean(axis=0))),
        "draws": int(len(means)),
        "q_min_per_A": float(q[0]),
        "q_max_per_A": float(q[-1]),
        "q_points": int(len(q)),
    }
    return result
