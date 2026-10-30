"""Synthetic ground-truth study: does BIFT chi-squared misrank because of error bars?

Builds a known truth curve mimicking SASDQK7 (α-synuclein): a two-Gaussian
coil P(r) matched to the experiment's fitted Rg and intensity scale, on the
pipeline's thinned q grid. Twenty synthetic simulations with graded, physical
perturbations (size scaling, shoulder weight, tail length, compact-subpopulation
bumps, broadening, tail stretching, large-distance truncation) are scored
against the truth, giving an exact noise-free discrepancy that defines the
ground-truth ranking after the same scale+background freedom the pipeline uses.

Each error-bar scenario produces one synthetic experiment (truth + seeded
noise); the GP is refit and scores all simulations by CRPS/SD, while the
notebook's reduced chi-squared (unweighted affine scale+offset fit; sigma only
normalizes residuals; reduced by n - 1) scores them with the scenario's
reported error bars. The BIFT Fortran binary is not recompiled here; its
global error-correction factor is rank-invariant, and its rebinned q grid is
the only unmodeled difference.

Scenarios:
- calibrated: constant sigma = SASDQK7's GP-inferred noise for both noise and
  reported bars.
- lowq_overestimated: same noise, but reported bars inflated 4x at the lowest
  q, decaying with q.
- counting_statistics: same constant noise, but reported bars proportional to
  intensity (relative errors), a common misassignment for constant noise.
- noisy_lowq: noise inflated 4x at low q (structure/aggregation), but reported
  bars constant everywhere -- SASDQK7's actual situation (its sigma column is
  -1 placeholders interpolated to a single constant).
- systematic_lowq: calibrated constant noise and bars, but a smooth correlated
  systematic error (5%-of-I(0) low-q additive excess, exp(-(q/0.08)^2)) is
  added to the experiment, mimicking imperfect buffer subtraction or slight
  interparticle effects. This violates both methods' iid-residual assumption
  and tests whether inflating predictive uncertainty (GP) or residual
  penalization (chi_r^2) handles a correlated artifact more gracefully.
"""

import argparse
import csv
import json
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gp_quality import score_simulation
from run_gp_quality import safe_name, thin_evenly
from saxs_gp import fit_saxs
from scipy.stats import spearmanr

RG_TARGET_NM = 4.6927
NOISE_SIGMA = 291.01
I_ZERO = 10000.0
R_MAX, R_POINTS = 400.0, 401


def coil_pair_distance(r, s1, s2, weight):
    """Two-Gaussian coil P(r) = r^2 [(1-w) N(r; s1) + w N(r; s2)], normalized to peak 1."""
    def lobe(scale):
        return (r**2) * np.exp(-(r**2) / (2 * scale**2))
    density = (1 - weight) * lobe(s1) + weight * lobe(s2)
    return density / density.max()


def gyration_radius(r, density, weights):
    m0 = np.sum(weights * density)
    m2 = np.sum(weights * density * r**2)
    return np.sqrt(m2 / (2 * m0)) / 10


def solve_s1(r, weights, target_rg_nm, s2_factor, weight):
    """Bisection on the short lobe so the pair model matches the target Rg."""
    low, high = 10.0, 100.0
    for _ in range(80):
        mid = (low + high) / 2
        rg = gyration_radius(r, coil_pair_distance(r, mid, s2_factor * mid, weight), weights)
        if rg < target_rg_nm:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def simulation_set(r, weights, operator, s1, s2_factor, weight):
    """Truth plus 20 named perturbations; returns {name: (label, P(r))}."""
    truth = coil_pair_distance(r, s1, s2_factor * s1, weight)
    sims = {"truth_000": ("exact truth", truth.copy())}

    def scaled(scale_factor):
        return coil_pair_distance(r, scale_factor * s1, scale_factor * s2_factor * s1, weight)

    for i, a in enumerate((0.90, 0.95, 0.98, 1.03, 1.10), start=1):
        sims[f"size_{i:02d}"] = (f"size x{a}", scaled(a))
    for i, w in enumerate((0.15, 0.22, 0.40), start=1):
        sims[f"shoulder_{i:02d}"] = (f"shoulder w={w}", coil_pair_distance(r, s1, s2_factor * s1, w))
    for i, factor in enumerate((2.0, 3.25), start=1):
        sims[f"tail_{i:02d}"] = (f"tail scale x{factor}", coil_pair_distance(r, s1, factor * s1, weight))
    for i, amplitude in enumerate((0.05, 0.12, 0.25), start=1):
        bump = truth.max() * amplitude * np.exp(-((r - 70.0) ** 2) / (2 * 12.0**2))
        sims[f"bump_{i:02d}"] = (f"compact r=70A a={amplitude}", truth + bump)
    for i, width in enumerate((12.0, 30.0), start=1):
        kernel = np.exp(-((r - r[len(r) // 2]) ** 2) / (2 * width**2))
        kernel /= kernel.sum()
        smoothed = np.convolve(truth, kernel, mode="same") * (np.convolve(np.ones_like(truth), kernel, mode="same") > 0)
        sims[f"broaden_{i:02d}"] = (f"broadened sd={width}A", smoothed / smoothed.max())
    for i, c in enumerate((-0.15, 0.15), start=1):
        stretched = truth * np.exp(c * r / 100.0)
        sims[f"tilt_{i:02d}"] = (f"tail tilt c={c:+}", stretched / stretched.max())
    for i, cutoff in enumerate((150.0, 100.0), start=1):
        truncated = truth.copy()
        truncated[r > cutoff] = 0
        sims[f"truncate_{i:02d}"] = (f"P(r)=0 beyond {cutoff:.0f}A", truncated)
    return sims


def q_intensity(density, operator):
    return operator @ density


def bift_style_chi2(q_exp, intensity_exp, sigma, q_sim, intensity_sim):
    """saxs_bift_comparison_SO.ipynb compute_chi2 convention, replicated exactly.

    Unweighted affine (scale AND offset) lstsq fit of the simulation onto the
    experimental curve; sigma enters only through the residual normalization;
    reduced by (n_points - 1). The notebook interpolates cubic; linear
    interpolation is used here because the synthetic q grid is dense relative
    to the experimental grid.
    """
    simulation = np.interp(q_exp, q_sim, intensity_sim)
    design = np.column_stack((simulation, np.ones_like(simulation)))
    scale, offset = np.linalg.lstsq(design, intensity_exp, rcond=None)[0]
    residual = (intensity_exp - (simulation * scale + offset)) / sigma
    return float(np.sum(residual**2) / (len(residual) - 1)), float(scale)


def true_discrepancy(q, intensity_sim, q_grid, truth):
    """Noise-free relative q-weighted RMSE after the pipeline's scale+background fit.

    Uses score_simulation with unit predictive SD so the fit is the pipeline's
    weighted affine least squares and the reported relative RMSE is exact.
    """
    result, _ = score_simulation(
        q, intensity_sim, q_grid, truth, np.ones_like(truth), fit_background=True
    )
    return result["gp_relative_rmse"]


def scenario_definitions(q, intensity_truth, noise_sigma):
    """name -> dict(noise, sigma, systematic): per-point actual noise SD,
    reported error bars, and a smooth q-dependent systematic error added to the
    synthetic experiment. All entries keep the RNG realization order of earlier
    versions, so cached fits for the original scenarios stay valid."""
    lowq = np.exp(-(q / 0.03) ** 2)
    relative = noise_sigma / np.mean(intensity_truth)
    zero = np.zeros_like(q)
    constant = np.full_like(q, noise_sigma)
    return {
        "calibrated": {"noise": constant, "sigma": constant, "systematic": zero},
        "lowq_overestimated": {"noise": constant, "sigma": noise_sigma * (1 + 3 * lowq), "systematic": zero},
        "counting_statistics": {"noise": constant, "sigma": relative * intensity_truth, "systematic": zero},
        "noisy_lowq": {"noise": noise_sigma * (1 + 3 * lowq), "sigma": constant, "systematic": zero},
        # Smooth, q-dependent systematic error: a 5%-of-I(0) low-q excess with a
        # physical width like imperfect buffer subtraction or a slight
        # interparticle/repulsion contribution. Calibrated constant noise and
        # bars in every other respect; the systematic is correlated across
        # points, violating both methods' iid assumptions.
        "systematic_lowq": {
            "noise": constant,
            "sigma": constant,
            "systematic": 0.05 * intensity_truth.max() * np.exp(-(q / 0.08) ** 2),
        },
    }


def main():
    here = Path(__file__).resolve().parent
    experiment_file = here / "outputs/databank_quality/SASDQK7/saxs_gp_intensity.csv"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=here / "outputs/synthetic_truth")
    parser.add_argument("--experiment-file", type=Path, default=experiment_file)
    parser.add_argument("--inference", choices=("nuts", "laplace"), default="nuts")
    parser.add_argument("--nuts-warmup", type=int, default=500)
    parser.add_argument("--nuts-samples", type=int, default=500)
    parser.add_argument("--nuts-chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scenarios", action="append", help="Run only these scenarios; may be repeated")
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Independent noise realizations per scenario; with >1 the summary reports "
        "the median and range of each Spearman rho across realizations",
    )
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    template = np.genfromtxt(args.experiment_file, delimiter=",", names=True)
    q_grid = np.sort(np.unique(np.asarray(template["q_per_A"], float)))
    r = np.linspace(0, R_MAX, R_POINTS)
    weights = np.full(R_POINTS, r[1])
    weights[[0, -1]] *= 0.5
    operator = np.sinc(np.outer(q_grid, r) / np.pi) * weights

    s2_factor, shoulder = 2.5, 0.30
    s1 = solve_s1(r, weights, RG_TARGET_NM, s2_factor, shoulder)
    sims_real = simulation_set(r, weights, operator, s1, s2_factor, shoulder)
    truth = q_intensity(sims_real["truth_000"][1], operator)
    truth *= I_ZERO / truth[0]
    for name in sims_real:
        sims_real[name] = (sims_real[name][0], sims_real[name][1] * I_ZERO / q_intensity(sims_real[name][1], operator)[0])

    q_sim = np.linspace(q_grid[0], q_grid[-1], 400)
    sim_curves = {}
    for name, (label, density) in sims_real.items():
        full = q_intensity(density, operator)
        sim_curves[name] = np.interp(q_sim, q_grid, full)

    discrepancies = {
        name: true_discrepancy(q_sim, curve, q_grid, truth) for name, curve in sim_curves.items()
    }
    truth_rank = {}
    for rank, name in enumerate(sorted(discrepancies, key=lambda item: discrepancies[item]), start=1):
        truth_rank[name] = rank
    with (args.output_dir / "simulation_truth_table.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["simulation", "label", "true_discrepancy", "true_rank"])
        for name, _ in sorted(discrepancies.items(), key=lambda item: item[1]):
            writer.writerow([name, sims_real[name][0], discrepancies[name], truth_rank[name]])

    scenarios = scenario_definitions(q_grid, truth, NOISE_SIGMA)
    if args.scenarios:
        scenarios = {name: scenarios[name] for name in args.scenarios}

    summary, per_scenario_rows, replicate_stats = [], [], {}
    for replicate in range(args.replicates):
        seed_replicate = args.seed + 1000 * replicate
        for name, definition in scenarios.items():
            scenario_rng = np.random.default_rng([seed_replicate, zlib.crc32(name.encode("utf-8"))])
            realization = (
                truth + definition["systematic"] + scenario_rng.normal(0, 1, len(truth)) * definition["noise"]
            )
            label = safe_name(name) if args.replicates == 1 else f"{safe_name(name)}_rep{replicate}"
            destination = args.output_dir / label
            intensity_file = destination / "saxs_gp_intensity.csv"
            if args.refit or not intensity_file.exists():
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "scenario.json").write_text(
                    json.dumps(
                        {
                            "noise_model": "constant"
                            if np.ptp(definition["noise"]) < 1e-12
                            else "low-q inflated 4x, exp(-(q/0.03)^2)",
                            "reported_sigma_model": {
                                "calibrated": "constant = GP-inferred SASDQK7 noise",
                                "lowq_overestimated": "4x inflated at low q",
                                "counting_statistics": "proportional to intensity",
                                "noisy_lowq": "constant (SASDQK7's actual case)",
                                "systematic_lowq": "constant, honest",
                            }[name],
                            "systematic_error": {
                                "none": None,
                                "systematic_lowq": "additive 5%-of-I(0) low-q excess, exp(-(q/0.08)^2), "
                                "mimicking imperfect buffer subtraction or slight interparticle effects",
                            }[name if name == "systematic_lowq" else "none"],
                            "seed": seed_replicate,
                            "replicate": replicate,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                fit_saxs(
                    q_grid,
                    realization,
                    destination,
                    inference=args.inference,
                    warmup=args.nuts_warmup,
                    samples=args.nuts_samples,
                    chains=args.nuts_chains,
                    seed=seed_replicate + 17,
                )
            gp = np.genfromtxt(intensity_file, delimiter=",", names=True)
            rows = []
            for sim_name, curve in sim_curves.items():
                gp_score, _ = score_simulation(
                    q_sim, curve, gp["q_per_A"], gp["I_mean"], gp["I_predictive_sd"], fit_background=True
                )
                chi2, _ = bift_style_chi2(q_grid, realization, definition["sigma"], q_sim, curve)
                rows.append(
                    {
                        "scenario": label,
                        "simulation": sim_name,
                        "label": sims_real[sim_name][0],
                        "true_discrepancy": discrepancies[sim_name],
                        "true_rank": truth_rank[sim_name],
                        "gp_crps_sd_units": gp_score["gp_crps_sd_units"],
                        "chi2_reduced": chi2,
                    }
                )
            for key, column in (("gp", "gp_crps_sd_units"), ("chi2", "chi2_reduced")):
                ordered = sorted(rows, key=lambda row: row[column])
                for rank, row in enumerate(ordered, start=1):
                    row[f"{key}_rank"] = rank
            per_scenario_rows.extend(rows)
            with (destination / "simulation_scores.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(sorted(rows, key=lambda row: row["true_rank"]))
            truth_order = [row["simulation"] for row in sorted(rows, key=lambda row: row["true_rank"])]
            rho, top5 = {}, {}
            for key, column in (("gp", "gp_crps_sd_units"), ("chi2", "chi2_reduced")):
                rho[key] = spearmanr([row["true_discrepancy"] for row in rows], [row[column] for row in rows])
                method_order = [row["simulation"] for row in sorted(rows, key=lambda row: row[f"{key}_rank"])]
                top5[key] = len(set(truth_order[:5]) & set(method_order[:5])) / 5
            entry = {
                "gp_spearman_rho": rho["gp"][0],
                "chi2_spearman_rho": rho["chi2"][0],
                "gp_top5_overlap": top5["gp"],
                "chi2_top5_overlap": top5["chi2"],
            }
            replicate_stats.setdefault(name, []).append(entry)
            if args.replicates > 1:
                print(f"{label}: GP rho={rho['gp'][0]:.3f}, chi2 rho={rho['chi2'][0]:.3f}", flush=True)
            if replicate > 0:
                continue

            summary.append(
                {
                    "scenario": name,
                    "gp_spearman_rho": rho["gp"][0],
                    "gp_spearman_p": rho["gp"][1],
                    "chi2_spearman_rho": rho["chi2"][0],
                    "chi2_spearman_p": rho["chi2"][1],
                    "gp_top5_overlap": top5["gp"],
                    "chi2_top5_overlap": top5["chi2"],
                    "chi2_dynamic_range": max(row["chi2_reduced"] for row in rows)
                    - min(row["chi2_reduced"] for row in rows),
                    "replicates": args.replicates,
                }
            )
            chi2_values = np.array([row["chi2_reduced"] for row in sorted(rows, key=lambda row: row["true_rank"])])
            gp_values = np.array([row["gp_crps_sd_units"] for row in sorted(rows, key=lambda row: row["true_rank"])])
            d_values = np.array([row["true_discrepancy"] for row in sorted(rows, key=lambda row: row["true_rank"])])
            labels = [row["simulation"] for row in sorted(rows, key=lambda row: row["true_rank"])]
            fig, (scatter_ax, heat_ax) = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
            scatter_ax.plot(d_values, gp_values, "o", color="C0", label="GP CRPS/SD")
            scatter_ax.plot(d_values, chi2_values / chi2_values.max() * gp_values.max(), "s", color="C3",
                            label=r"BIFT $\chi_r^2$ (rescaled)")
            for d, g, sim_label in zip(d_values, gp_values, labels):  # noqa: B905 -- parallel arrays
                scatter_ax.annotate(sim_label, (d, g), xytext=(3, 3), textcoords="offset points", fontsize=6)
            scatter_ax.set(
                xlabel="True discrepancy (relative q-weighted RMSE)",
                ylabel="Score (GP CRPS/SD; chi2 rescaled)",
                title=f"{name}: score vs truth (GP rho={rho['gp'][0]:.3f}, chi2 rho={rho['chi2'][0]:.3f})",
            )
            scatter_ax.grid(alpha=0.2)
            scatter_ax.legend(frameon=False, fontsize=8)
            stacked = np.column_stack([d_values / d_values.max(), gp_values / gp_values.max(), chi2_values / chi2_values.max()])
            image = heat_ax.imshow(stacked, cmap="RdYlGn_r", aspect="auto", vmin=0)
            heat_ax.set_yticks(np.arange(len(labels)), labels, fontsize=6)
            heat_ax.set_xticks([0, 1, 2], ["truth\n(max)", "GP\n(max)", r"$\chi_r^2$" + "\n(max)"])
            heat_ax.set_title("Scores sorted by true discrepancy (each col. own scale)")
            fig.colorbar(image, ax=heat_ax, shrink=0.8)
            fig.savefig(args.output_dir / f"{safe_name(name)}.png", dpi=220)
            plt.close(fig)
            print(f"{name}: GP rho={rho['gp'][0]:.3f}, chi2 rho={rho['chi2'][0]:.3f}", flush=True)

    if args.replicates > 1:
        for row in summary:
            stats = replicate_stats[row["scenario"]]
            for key in ("gp_spearman_rho", "chi2_spearman_rho", "gp_top5_overlap", "chi2_top5_overlap"):
                values = [float(item[key]) for item in stats]
                row[f"{key}_median"] = float(np.median(values))
                row[f"{key}_min"] = min(values)
                row[f"{key}_max"] = max(values)

    with (args.output_dir / "scenario_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (args.output_dir / "all_scenario_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario"] + [k for k in per_scenario_rows[0] if k != "scenario"])
        writer.writeheader()
        writer.writerows(per_scenario_rows)
    print(f"Wrote summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
