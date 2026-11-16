"""Compare single-Gaussian CRPS with the hyperparameter-draw mixture CRPS.

Mirrors plot_crps_bift_comparison.py: per-experiment ranked heatmaps of both
scores, fitted curves colored by the draw-mixture score, and a rank-agreement
scatter with Spearman correlation and disagreement annotations. The Gaussian
summary uses the saved predictive mean and SD; the draw mixture keeps every
NUTS hyperparameter draw as an equal-weight Gaussian predictive component.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gp_quality_draws import mixture_components, score_simulation_draws
from plot_crps_bift_comparison import finite_lognorm
from plot_crps_quality import read_experiment
from run_gp_quality import read_simulation, safe_name
from scipy.stats import rankdata, spearmanr


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=here / "outputs/databank_quality/gp_quality_scores.csv")
    parser.add_argument("--output-dir", type=Path, default=here / "outputs/databank_quality/crps_gaussian_vs_draws")
    parser.add_argument("--experiments-dir", type=Path, default=here / "outputs/databank_quality")
    parser.add_argument("--max-draws", type=int, default=1000, help="Evenly thinned draw limit for the mixture")
    parser.add_argument("--recompute-components", action="store_true", help="Ignore cached mixture components")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.scores.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    experiments = sorted({row["experiment"] for row in rows})

    merged = []
    for experiment in experiments:
        selected = [row for row in rows if row["experiment"] == experiment]
        q_gp, means, predictive_sd = mixture_components(
            args.experiments_dir / safe_name(experiment),
            max_draws=args.max_draws,
            recompute=args.recompute_components,
        )
        print(f"{experiment}: reconstructed {len(means)} mixture components", flush=True)
        for row in selected:
            q_sim, intensity_sim = read_simulation(root / row["simulation_path"] / "SAXS.yaml")
            draws_score = score_simulation_draws(
                q_sim,
                intensity_sim,
                q_gp,
                means,
                predictive_sd,
                float(row["scale_sim_to_experiment"]),
                float(row["background"]),
            )
            row.update(
                gp_crps_draws=draws_score["gp_crps_draws"],
                gp_crps_draws_normalized=draws_score["gp_crps_draws_normalized"],
                gp_crps_draws_sd_units=draws_score["gp_crps_draws_sd_units"],
                draws=draws_score["draws"],
            )
            merged.append(row)

    fields = list(merged[0])
    with (args.output_dir / "gp_scores_gaussian_vs_draws.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)

    comparison_fields = [
        "experiment",
        "system",
        "simulation_id",
        "gp_crps_sd_units",
        "gp_crps_draws_sd_units",
        "gaussian_rank",
        "draws_rank",
        "absolute_rank_difference",
    ]
    summaries = []
    for experiment in experiments:
        rows_e = [row for row in merged if row["experiment"] == experiment]
        gaussian = np.array([float(row["gp_crps_sd_units"]) for row in rows_e])
        draws = np.array([float(row["gp_crps_draws_sd_units"]) for row in rows_e])
        gaussian_rank, draws_rank = rankdata(gaussian), rankdata(draws)
        for row, a, b in zip(rows_e, gaussian_rank, draws_rank):  # noqa: B905 -- derived from rows_e
            row.update(gaussian_rank=float(a), draws_rank=float(b), absolute_rank_difference=float(abs(a - b)))
        order = np.argsort(gaussian_rank)
        rows_e = [rows_e[index] for index in order]
        gaussian, draws = gaussian[order], draws[order]
        gaussian_rank, draws_rank = gaussian_rank[order], draws_rank[order]
        rho, p_value = spearmanr(gaussian, draws)
        summaries.append(
            {
                "experiment": experiment,
                "matched_simulations": len(rows_e),
                "draws": int(rows_e[0]["draws"]),
                "spearman_rho": rho,
                "spearman_p": p_value,
                "median_relative_score_change": float(np.median(np.abs(draws / gaussian - 1))),
                "maximum_relative_score_change": float(np.max(np.abs(draws / gaussian - 1))),
                "median_absolute_rank_difference": float(np.median(abs(gaussian_rank - draws_rank))),
                "maximum_absolute_rank_difference": float(np.max(abs(gaussian_rank - draws_rank))),
            }
        )

        stem = safe_name(experiment)
        with (args.output_dir / f"{stem}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=comparison_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows_e)

        labels = [f"{row['system']}:{row['simulation_id']}" for row in rows_e]
        height = max(6, min(14, 2.4 + 0.22 * len(rows_e)))
        fig = plt.figure(figsize=(17, height), constrained_layout=True)
        grid = fig.add_gridspec(2, 4, width_ratios=(0.75, 0.75, 2.7, 0.12), height_ratios=(2.1, 1))
        gaussian_ax = fig.add_subplot(grid[:, 0])
        draws_ax = fig.add_subplot(grid[:, 1])
        curve_ax = fig.add_subplot(grid[0, 2])
        rank_ax = fig.add_subplot(grid[1, 2])
        cb_grid = grid[:, 3].subgridspec(2, 1, hspace=0.35)
        gaussian_cax = fig.add_subplot(cb_grid[0])
        draws_cax = fig.add_subplot(cb_grid[1])
        cmap = plt.get_cmap("RdYlGn_r")
        gaussian_norm, draws_norm = finite_lognorm(gaussian), finite_lognorm(draws)
        gaussian_image = gaussian_ax.imshow(gaussian[:, None], cmap=cmap, norm=gaussian_norm, aspect="auto", origin="upper")
        draws_image = draws_ax.imshow(draws[:, None], cmap=cmap, norm=draws_norm, aspect="auto", origin="upper")
        fontsize = 6 if len(rows_e) > 25 else 8
        gaussian_ax.set_yticks(np.arange(len(rows_e)), labels, fontsize=fontsize)
        draws_ax.set_yticks([])
        gaussian_ax.set_xticks([0], ["Gaussian\nCRPS/SD"])
        draws_ax.set_xticks([0], ["Draw-mixture\nCRPS/SD"])
        gaussian_ax.set_ylabel("Simulation (ordered by Gaussian CRPS)")
        for ax, values, norm in ((gaussian_ax, gaussian, gaussian_norm), (draws_ax, draws, draws_norm)):
            for index, value in enumerate(values):
                color = "white" if norm(value) > 0.62 else "black"
                ax.text(0, index, f"{value:.3g}", ha="center", va="center", fontsize=fontsize, color=color)
        fig.colorbar(gaussian_image, cax=gaussian_cax).set_label("Single-Gaussian CRPS / mean predictive SD")
        fig.colorbar(draws_image, cax=draws_cax).set_label("Draw-mixture CRPS / mean predictive SD")

        experiment_path = root / "Data/Experiments/saxs" / experiment / "saxs.dat"
        q_exp, intensity_exp, sigma_exp = read_experiment(experiment_path)
        use_exp = np.isfinite(intensity_exp) & (intensity_exp > 0)
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
        for row, score in zip(rows_e, draws):  # noqa: B905 -- draws is derived from rows_e
            q_sim, intensity_sim = read_simulation(root / row["simulation_path"] / "SAXS.yaml")
            fitted = float(row["scale_sim_to_experiment"]) * intensity_sim + float(row["background"])
            use = (q_sim >= float(row["q_min_per_A"])) & (q_sim <= float(row["q_max_per_A"])) & (fitted > 0)
            curve_ax.plot(q_sim[use], fitted[use], color=cmap(draws_norm(score)), linewidth=1, alpha=0.72)
        curve_ax.set(
            yscale="log", xlabel=r"$q$ (Å$^{-1}$)", ylabel=r"$I(q)$ (a.u.)", title=f"{experiment}: matched simulations"
        )
        curve_ax.grid(alpha=0.18)
        curve_ax.legend(frameon=False)

        rank_ax.scatter(gaussian_rank, draws_rank, c=draws, cmap=cmap, norm=draws_norm, edgecolor="black", linewidth=0.25)
        limit = len(rows_e) + 1
        rank_ax.plot([0, limit], [0, limit], "--", color=".4", linewidth=1)
        disagreement = np.argsort(abs(gaussian_rank - draws_rank))[-min(5, len(rows_e)) :]
        for index in disagreement:
            rank_ax.annotate(
                labels[index],
                (gaussian_rank[index], draws_rank[index]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6,
            )
        rank_ax.set(
            xlim=(0, limit),
            ylim=(0, limit),
            xlabel="Single-Gaussian CRPS rank",
            ylabel="Draw-mixture CRPS rank",
            title=rf"Rank agreement: Spearman $\rho$={rho:.3f}, $p$={p_value:.2g}",
        )
        rank_ax.grid(alpha=0.2)
        fig.savefig(args.output_dir / f"{stem}.png", dpi=450)
        plt.close(fig)

    with (args.output_dir / "rank_agreement_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote {len(merged)} comparisons across {len(summaries)} experiments")


if __name__ == "__main__":
    main()
