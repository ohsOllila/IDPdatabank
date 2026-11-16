"""Diagnose where GP CRPS and BIFT reduced-chi-squared rankings disagree and why.

For the simulations with the largest absolute rank differences in the matched
tables (outputs/databank_quality/crps_vs_bift), this script standardizes each
fitted simulation residual two ways:

- z_gp(q) = (fitted - gp_mean) / gp_predictive_sd, exactly as the GP pipeline
  computes it (residuals/ CSV), and
- z_chi2(q) = (fitted - experiment) / reported_experimental_sigma, the quantity
  BIFT reduced chi-squared aggregates, using the notebook's sigma convention
  (missing/report values interpolated; SASDQJ7/SASDQK7 collapse to a constant).

Both use the same fitted simulation curve (GP-fitted scale and background), so
differences between the panels isolate the uncertainty model: the GP's inferred
noise versus reported (often constant or miscalibrated) error bars. A sigma
ratio panel shows where the two uncertainty models disagree, and a residual
anatomy panel localizes disagreements to low-q (size) or high-q (shape) error.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot_crps_quality import read_experiment
from run_gp_quality import safe_name


def read_residuals(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.asarray(data[name]) for name in data.dtype.names}


def experimental_standardization(root, experiment, q):
    """Reported experimental intensity and sigma interpolated onto the GP q grid.

    Follows the saxs_bift_comparison_SO.ipynb reader: negative and missing sigma
    values are interpolated from the valid ones before interpolation in q.
    """
    q_exp, intensity_exp, sigma_exp = read_experiment(root / "Data/Experiments/saxs" / experiment / "saxs.dat")
    valid = np.isfinite(sigma_exp) & (sigma_exp > 0)
    if valid.any() and not valid.all():
        sigma_exp = np.interp(np.arange(len(sigma_exp)), np.flatnonzero(valid), sigma_exp[valid])
    if not valid.any():
        raise ValueError(f"{experiment} has no usable sigma column")
    return (
        np.interp(q, q_exp, intensity_exp),
        np.interp(q, q_exp, sigma_exp),
        len(q_exp),
        float(np.median(sigma_exp[valid])),
    )


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, default=here / "outputs/databank_quality/crps_vs_bift")
    parser.add_argument("--residuals-dir", type=Path, default=here / "outputs/databank_quality/residuals")
    parser.add_argument("--output-dir", type=Path, default=here / "outputs/databank_quality/method_disagreement")
    parser.add_argument("--cases", type=int, default=4, help="Largest-rank-difference cases per experiment")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for table in sorted(args.comparison_dir.glob("*.csv")):
        if table.name == "rank_agreement_summary.csv":
            continue
        with table.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows.sort(key=lambda row: -float(row["absolute_rank_difference"]))
        selected = rows[: args.cases]
        if not selected:
            continue
        experiment = selected[0]["experiment"]

        q_exp, intensity_exp, sigma_exp = read_experiment(root / "Data/Experiments/saxs" / experiment / "saxs.dat")
        gp = np.genfromtxt(
            here / "outputs/databank_quality" / safe_name(experiment) / "saxs_gp_intensity.csv",
            delimiter=",",
            names=True,
        )

        for case, row in enumerate(selected, start=1):
            label = f"{row['system']}:{row['simulation_id']}"
            residual = read_residuals(
                args.residuals_dir / f"{row['simulation_id']}_{safe_name(experiment)}.csv"
            )
            q = residual["q_per_A"]
            gp_mean, gp_sd = residual["gp_mean"], residual["gp_predictive_sd"]
            z_gp = residual["z"]
            experiment_on_grid, sigma_on_grid, n_exp, sigma_median = experimental_standardization(root, experiment, q)
            z_chi2 = (residual["simulation_fitted"] - experiment_on_grid) / sigma_on_grid
            q_low = q <= 0.1
            summary.append(
                {
                    "experiment": experiment,
                    "system": row["system"],
                    "simulation_id": row["simulation_id"],
                    "gp_rank": row["gp_rank"],
                    "bift_rank": row["bift_rank"],
                    "rank_difference": row["absolute_rank_difference"],
                    "gp_crps_sd_units": row["gp_crps_sd_units"],
                    "bift_chi2_reduced": row["bift_chi2_reduced"],
                    "gp_z_rmse": float(np.sqrt(np.mean(z_gp**2))),
                    "chi2_z_rmse": float(np.sqrt(np.mean(z_chi2**2))),
                    "gp_mean_abs_z_lowq": float(np.mean(np.abs(z_gp[q_low]))),
                    "chi2_mean_abs_z_lowq": float(np.mean(np.abs(z_chi2[q_low]))),
                    "gp_mean_abs_z_highq": float(np.mean(np.abs(z_gp[~q_low]))),
                    "chi2_mean_abs_z_highq": float(np.mean(np.abs(z_chi2[~q_low]))),
                    "median_sigma_ratio": float(np.median(sigma_on_grid / gp_sd)),
                    "experimental_points": n_exp,
                    "experimental_sigma_median": sigma_median,
                }
            )

            fig, axes = plt.subplots(
                2, 2, figsize=(14, 8.5), constrained_layout=True, gridspec_kw={"width_ratios": (1.6, 1)}
            )
            curve_ax, ratio_ax = axes[0]
            gp_ax, chi2_ax = axes[1]
            use_exp = np.isfinite(intensity_exp) & (intensity_exp > 0) & (q_exp <= q[-1])
            curve_ax.errorbar(
                q_exp[use_exp],
                intensity_exp[use_exp],
                yerr=np.where(np.isfinite(sigma_exp[use_exp]) & (sigma_exp[use_exp] > 0), sigma_exp[use_exp], 0),
                fmt=".",
                markersize=2.5,
                linewidth=0.5,
                color="black",
                alpha=0.55,
                zorder=5,
                label="Experiment",
            )
            curve_ax.plot(q, gp_mean, color="C0", lw=2, label="GP mean")
            curve_ax.fill_between(q, gp_mean - gp_sd, gp_mean + gp_sd, color="C0", alpha=0.2, label=r"GP $\pm$1 SD")
            curve_ax.plot(q, residual["simulation_fitted"], color="C3", lw=1.5, label="Fitted simulation")
            curve_ax.set(yscale="log", xlabel=r"$q$ (Å$^{-1}$)", ylabel=r"$I(q)$ (a.u.)")
            curve_ax.set_title(f"{label}: GP rank {float(row['gp_rank']):.0f}, BIFT rank {float(row['bift_rank']):.0f}")
            curve_ax.grid(alpha=0.2)
            curve_ax.legend(frameon=False, fontsize=8)

            ratio_ax.semilogx(q, sigma_on_grid / gp_sd, ".", ms=3, color="0.35")
            ratio_ax.axhline(1.0, color="C0", lw=1, label="Equal uncertainty")
            ratio_ax.set(
                xlabel=r"$q$ (Å$^{-1}$)",
                ylabel=r"$\sigma_{\rm exp} / \sigma_{\rm GP}$",
                title="Uncertainty models",
            )
            ratio_ax.grid(alpha=0.2)
            ratio_ax.legend(frameon=False, fontsize=8)

            for ax, values, name in ((gp_ax, z_gp, "GP: z = (sim - GP mean) / GP predictive SD"),
                                     (chi2_ax, z_chi2, r"$\chi^2$: z = (sim - exp) / reported $\sigma$")):
                ax.axhline(0, color="0.4", lw=0.8)
                ax.fill_between([-2, 2], -1, 1, color="0.85", zorder=0)
                ax.plot(q, values, ".-", ms=3, lw=0.6, color="C3")
                ax.set(xlabel=r"$q$ (Å$^{-1}$)", ylabel="Standardized residual", title=name, xlim=(0, q[-1]))
                ax.grid(alpha=0.2)
            gp_ax.set_ylim(*np.percentile(np.r_[z_gp, z_chi2], [1, 99]) + np.array([-1, 1]))
            chi2_ax.set_ylim(gp_ax.get_ylim())

            fig.savefig(args.output_dir / f"{safe_name(experiment)}_case{case}_{row['simulation_id']}.png", dpi=250)
            plt.close(fig)

    if summary:
        summary_file = args.output_dir / "disagreement_anatomy.csv"
        with summary_file.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        print(f"Wrote {len(summary)} disagreement cases to {summary_file}")


if __name__ == "__main__":
    main()
