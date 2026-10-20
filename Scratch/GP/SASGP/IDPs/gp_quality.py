"""Scores for comparing a simulated SAXS curve with a fitted GP posterior."""
import numpy as np
from scipy.optimize import lsq_linear
from scipy.special import ndtr


def q_weights(q):
    """Normalized trapezoid weights, making summaries insensitive to q sampling density."""
    q = np.asarray(q, float)
    if q.ndim != 1 or len(q) < 2 or np.any(np.diff(q) <= 0):
        raise ValueError("q must be a strictly increasing one-dimensional array")
    weights = np.empty_like(q)
    weights[0] = (q[1] - q[0]) / 2
    weights[-1] = (q[-1] - q[-2]) / 2
    weights[1:-1] = (q[2:] - q[:-2]) / 2
    return weights / weights.sum()


def gaussian_crps(observation, mean, sd):
    """Continuous ranked probability score for N(mean, sd). Lower is better."""
    observation, mean, sd = np.broadcast_arrays(observation, mean, sd)
    if np.any(sd <= 0) or not np.isfinite(sd).all():
        raise ValueError("sd must be finite and positive")
    z = (observation - mean) / sd
    density = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    return sd * (z * (2 * ndtr(z) - 1) + 2 * density - 1 / np.sqrt(np.pi))


def score_simulation(q_sim, intensity_sim, q_gp, mean_gp, sd_gp,
                     fit_background=True, sd_floor_fraction=1e-6):
    """Fit simulation scale/background and score it against a GP predictive marginal.

    The GP is treated pointwise. This deliberately avoids inversion of its strongly
    correlated, often nearly singular covariance. Aggregates use q-interval weights.
    The primary score is CRPS normalized by the q-weighted mean absolute GP intensity.
    """
    q_sim, intensity_sim, q_gp, mean_gp, sd_gp = map(
        lambda value: np.asarray(value, float),
        (q_sim, intensity_sim, q_gp, mean_gp, sd_gp),
    )
    for name, values in (("q_sim", q_sim), ("intensity_sim", intensity_sim),
                         ("q_gp", q_gp), ("mean_gp", mean_gp), ("sd_gp", sd_gp)):
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite and one-dimensional")
    if len(q_sim) != len(intensity_sim) or len(q_gp) != len(mean_gp) or len(q_gp) != len(sd_gp):
        raise ValueError("q and intensity arrays must have matching lengths")
    if np.any(np.diff(q_sim) <= 0) or np.any(np.diff(q_gp) <= 0):
        raise ValueError("q arrays must be strictly increasing")

    overlap = (q_gp >= q_sim[0]) & (q_gp <= q_sim[-1])
    if overlap.sum() < 3:
        raise ValueError("simulation and GP need at least three overlapping q points")
    q = q_gp[overlap]
    mean = mean_gp[overlap]
    simulation = np.interp(q, q_sim, intensity_sim)
    scale_reference = max(float(np.max(np.abs(mean))), np.finfo(float).tiny)
    sd = np.maximum(sd_gp[overlap], sd_floor_fraction * scale_reference)
    weights = q_weights(q)

    columns = [simulation]
    lower = [0.0]
    upper = [np.inf]
    if fit_background:
        columns.append(np.ones_like(simulation))
        lower.append(-np.inf)
        upper.append(np.inf)
    design = np.column_stack(columns)
    whiten = np.sqrt(weights) / sd
    fit = lsq_linear(design * whiten[:, None], mean * whiten,
                     bounds=(lower, upper), method="bvls")
    fitted = design @ fit.x
    residual = fitted - mean
    z = residual / sd
    normalization = np.sum(weights * np.abs(mean))
    if normalization <= np.finfo(float).tiny:
        raise ValueError("GP mean has zero intensity scale")
    crps = gaussian_crps(fitted, mean, sd)

    result = {
        "gp_crps": float(np.sum(weights * crps)),
        "gp_crps_normalized": float(np.sum(weights * crps) / normalization),
        "gp_crps_sd_units": float(np.sum(weights * crps) / np.sum(weights * sd)),
        "gp_z_rmse": float(np.sqrt(np.sum(weights * z**2))),
        "gp_mean_abs_z": float(np.sum(weights * np.abs(z))),
        "gp_coverage_68": float(np.sum(weights * (np.abs(z) <= 1.0))),
        "gp_coverage_95": float(np.sum(weights * (np.abs(z) <= 1.959963984540054))),
        "gp_relative_rmse": float(np.sqrt(np.sum(weights * residual**2)) / normalization),
        "scale_sim_to_experiment": float(fit.x[0]),
        "background": float(fit.x[1]) if fit_background else 0.0,
        "q_min_per_A": float(q[0]),
        "q_max_per_A": float(q[-1]),
        "q_points": int(len(q)),
        "fit_background": bool(fit_background),
    }
    return result, {"q": q, "gp_mean": mean, "gp_sd": sd,
                    "simulation_fitted": fitted, "z": z, "q_weight": weights}
