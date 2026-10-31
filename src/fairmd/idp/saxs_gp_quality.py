"""GP SAXS quality scoring for IDPdatabank simulations.

Standalone scoring of simulated SAXS profiles against precomputed
sinc-transform Gaussian-process fits of experimental curves:

- The fit of each experimental dataset is produced by the Scratch pipeline
  (Scratch/GP/SASGP/IDPs, run_gp_quality.py; NUTS over hyperparameters) and
  stored with the experiment itself as Data/Experiments/saxs/<experiment>/
  gp_fit.dat, a portable text file with a JSON metadata header. The GP
  infers the noise level and background from the data; reported experimental
  error bars are not used.
- evaluate_SAXS_quality_GP(system, databankPath) mirrors
  evaluate_SAXS_quality: it scores the system's SAXS.yaml against the stored
  fit of every experiment linked in its README.yaml and returns one row per
  experiment, keyed with the SAXS_ prefix so results sit alongside
  SAXS_chi2_bift in the simulation folder.

This module needs only numpy, scipy, and pyyaml; it never imports torch,
pyro, or MDAnalysis. The primary score SAXS_gp_crps_sd_units is the
trapezoid-q-weighted pointwise Gaussian CRPS divided by the mean GP
predictive SD: a curve equal to the GP mean scores 0.234, offsets of one and
two predictive SDs score 0.602 and 1.453. Lower is better.
"""

import json
import os

import numpy as np
import yaml
from scipy.optimize import lsq_linear
from scipy.special import ndtr

GP_FIT_FILENAME = "gp_fit.dat"
METADATA_PREFIX = "# metadata: "
COLUMNS = ("q_per_A", "I_mean", "I_predictive_sd", "I_latent_sd", "I_lower_95", "I_upper_95")

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


def score_simulation(q_sim, intensity_sim, q_gp, mean_gp, sd_gp, fit_background=True, sd_floor_fraction=1e-6):
    """Fit simulation scale/background and score it against a GP predictive marginal.

    The GP is treated pointwise. This deliberately avoids inversion of its strongly
    correlated, often nearly singular covariance. Aggregates use q-interval weights.
    The primary score is CRPS normalized by the q-weighted mean absolute GP intensity.
    """
    q_sim, intensity_sim, q_gp, mean_gp, sd_gp = map(
        lambda value: np.asarray(value, float),
        (q_sim, intensity_sim, q_gp, mean_gp, sd_gp),
    )
    for name, values in (
        ("q_sim", q_sim),
        ("intensity_sim", intensity_sim),
        ("q_gp", q_gp),
        ("mean_gp", mean_gp),
        ("sd_gp", sd_gp),
    ):
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
    fit = lsq_linear(design * whiten[:, None], mean * whiten, bounds=(lower, upper), method="bvls")
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
    detail = {"q": q, "gp_mean": mean, "gp_sd": sd, "simulation_fitted": fitted, "z": z, "q_weight": weights}
    return result, detail


def write_gp_fit(path, q_per_A, I_mean, I_predictive_sd, I_latent_sd, I_lower_95, I_upper_95, metadata):
    """Write the portable per-experiment GP fit: JSON header line + fixed columns.

    Written by the fitting pipeline (fairmd.idp.saxs_gp_fit, run by
    Scripts/AnalyzeDatabank/calcSAXSGPFits.py) next to the experiment's
    saxs.dat so scoring never depends on the fitting environment.
    """
    arrays = np.broadcast_arrays(
        np.asarray(q_per_A, float), np.asarray(I_mean, float), np.asarray(I_predictive_sd, float),
        np.asarray(I_latent_sd, float), np.asarray(I_lower_95, float), np.asarray(I_upper_95, float),
    )
    if arrays[0].ndim != 1 or len(arrays[0]) < 3:
        raise ValueError("gp fit columns must be one-dimensional with at least three points")
    if not np.isfinite(np.column_stack(arrays)).all():
        raise ValueError("gp fit columns must be finite")
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(f"# GP sinc-transform SAXS fit; columns: {' '.join(COLUMNS)}\n")
        handle.write(f"{METADATA_PREFIX}{json.dumps(metadata, sort_keys=True)}\n")
        np.savetxt(handle, np.column_stack(arrays), fmt="%.10e")


def read_gp_fit(path):
    """Read a gp_fit.dat: returns (columns dict, metadata dict)."""
    metadata = {}
    data_lines = []
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(METADATA_PREFIX):
                metadata = json.loads(stripped[len(METADATA_PREFIX):])
                continue
            if stripped.startswith("#"):
                continue
            data_lines.append(line)
    if not data_lines:
        raise ValueError(f"no data columns found in {path}")
    if len(metadata) == 0:
        raise ValueError(f"no metadata header found in {path}")
    data = np.loadtxt(data_lines)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != len(COLUMNS):
        raise ValueError(f"{path} has {data.shape[1]} columns; expected {len(COLUMNS)}")
    return {name: data[:, index] for index, name in enumerate(COLUMNS)}, metadata


def read_saxs_yaml(filename):
    """Read q[1/A] and mean I(q) from a SAXS.yaml / SAXS_MAICoS.yaml file.

    Same key conventions as fairmd.idp.protein_functions.read_saxs_yaml, kept
    local so this module never triggers that module's heavy imports.
    """
    with open(filename) as handle:
        data = yaml.safe_load(handle)
    intensity_keys = ("mean_I(q)[a.u.]", "mean_Inten[a.u.]", "I(q)[a.u.]")
    q, intensity = [], []
    for record in data:
        q_value = record.get("q[1/A]", record.get("q"))
        value = next((record[key] for key in intensity_keys if key in record), None)
        if q_value is not None and value is not None:
            q.append(q_value)
            intensity.append(value)
    return np.asarray(q, float), np.asarray(intensity, float)


def _evaluation_plot(q, gp_mean, gp_sd, fitted, z, intensity_sim_fitted, output_file, experiment_id, score):
    """Per-simulation quality figure, mirroring the saxs_evaluation_*.pdf convention.

    Top panel: GP predictive band with the fitted simulation curve;
    bottom panel: standardized residuals in predictive-SD units.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (curve_ax, residual_ax) = plt.subplots(
        2, 1, figsize=(7, 8), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": (2, 1)}
    )
    curve_ax.fill_between(
        q, gp_mean - gp_sd, gp_mean + gp_sd, alpha=0.25, color="C0", label=r"GP mean $\pm$ 1 predictive SD"
    )
    curve_ax.plot(q, gp_mean, color="C0", lw=1.5, label="GP mean (experiment)")
    curve_ax.plot(q, fitted, color="C3", lw=1.2, label="Fitted simulation")
    curve_ax.set(yscale="log", ylabel=r"$I(q)$ (a.u.)", title=f"SAXS GP quality: {experiment_id}")
    curve_ax.grid(alpha=0.2)
    curve_ax.legend(frameon=False, fontsize=8)
    residual_ax.axhline(0, color="0.4", lw=0.8)
    residual_ax.fill_between([q[0], q[-1]], -1, 1, color="0.85", zorder=0)
    residual_ax.plot(q, z, ".-", ms=3, lw=0.6, color="C3")
    residual_ax.set(xlabel=r"$q$ (Å$^{-1}$)", ylabel="z = (sim - GP) / SD", xlim=(q[0], q[-1]))
    residual_ax.set_title(f"CRPS/SD = {score:.3f} (exact match 0.234, 1 SD 0.602)", fontsize=9)
    residual_ax.grid(alpha=0.2)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def evaluate_SAXS_quality_GP(system, databankPath, plot=False):
    """Score a system's SAXS profile against stored GP fits of its linked experiments.

    Returns one row per linked experimental dataset that has a gp_fit.dat
    (empty list if the system links no SAXS experiment or lacks SAXS.yaml;
    experiments without a stored fit are skipped with a message so the rest
    of the property calculation continues). Row keys carry the SAXS_ prefix
    like evaluate_SAXS_quality, with SAXS_gp_crps_sd_units the primary score.

    With plot=True, also writes saxs_gp_evaluation_<experiment>.png into the
    simulation folder for each scored experiment, mirroring the
    saxs_evaluation_<experiment>.pdf convention of evaluate_SAXS_quality.
    """
    experiment_ids = system.get("EXPERIMENT", {}).get("saxs", {}).get("path", []) or []
    if not experiment_ids:
        return []

    sim_dir = os.path.join(databankPath, "Data", "Simulations", system["path"])
    saxs_file = os.path.join(sim_dir, "SAXS.yaml")
    if not os.path.exists(saxs_file):
        print("  SAXS.yaml not found; run the CRYSOL profile step first")
        return []

    q_sim, intensity_sim = read_saxs_yaml(saxs_file)

    rows = []
    for experiment_id in experiment_ids:
        fit_file = os.path.join(databankPath, "Data", "Experiments", "saxs", experiment_id, GP_FIT_FILENAME)
        if not os.path.exists(fit_file):
            print(
                f"  GP fit for {experiment_id} not found at {fit_file}; "
                "run Scratch/GP/SASGP/IDPs/run_gp_quality.py to create it"
            )
            continue
        fit, metadata = read_gp_fit(fit_file)
        result, detail = score_simulation(
            q_sim, intensity_sim, fit["q_per_A"], fit["I_mean"], fit["I_predictive_sd"], fit_background=True
        )
        row = {f"SAXS_{key}": value for key, value in result.items()}
        row["experiment"] = experiment_id
        row["gp_nuts_diagnostics_passed"] = metadata.get("nuts_diagnostics_passed")
        row["gp_inferred_noise_sigma"] = metadata.get("noise_sigma")
        row["gp_fitted_background"] = metadata.get("background")
        row["SAXS_gp_crps_interpretation"] = (
            "gp_crps_sd_units: exact match to GP mean = 0.234; one and two predictive SDs = 0.602 and 1.453"
        )
        rows.append(row)
        if plot:
            try:
                _evaluation_plot(
                    detail["q"],
                    detail["gp_mean"],
                    detail["gp_sd"],
                    detail["simulation_fitted"],
                    detail["z"],
                    intensity_sim,
                    os.path.join(sim_dir, f"saxs_gp_evaluation_{str(experiment_id).replace('/', '_')}.png"),
                    experiment_id,
                    row["SAXS_gp_crps_sd_units"],
                )
            except ImportError:
                print("  matplotlib not available; skipping GP evaluation plot")
    return rows
