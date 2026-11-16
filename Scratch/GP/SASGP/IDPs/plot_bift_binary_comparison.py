"""Figures comparing the real-BIFT binary and GP scoring of the synthetic study.

Reads bift_scores.csv from run_bift_binary_study.py and produces one figure
per scenario:

- score vs true discrepancy (GP, BIFT constant mode, BIFT non-constant mode;
  points are median scores across realizations, with min-max bars),
- a ranked heatmap of the three scores sorted by the exact truth ranking,
- GP-vs-BIFT rank agreement across realizations.

Outputs to outputs/synthetic_truth_bift_binary/plots/.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def median_scores(rows, column):
    scores = defaultdict(list)
    for row in rows:
        scores[row["simulation"]].append(float(row[column]))
    return {name: float(np.median(values)) for name, values in scores.items()}


def spread(rows, column):
    scores = defaultdict(list)
    for row in rows:
        scores[row["simulation"]].append(float(row[column]))
    return {
        name: (float(np.min(values)), float(np.max(values))) for name, values in scores.items()
    }


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=here.parent / "outputs" / "synthetic_truth_bift_binary" / "bift_scores.csv")
    parser.add_argument("--output-dir", type=Path, default=here.parent / "outputs" / "synthetic_truth_bift_binary" / "plots")
    parser.add_argument("--summary", type=Path, default=here.parent / "outputs" / "synthetic_truth_bift_binary" / "bift_summary.csv")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.scores.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with args.summary.open(newline="") as handle:
        summary = {row["scenario"]: row for row in csv.DictReader(handle)}
    scenarios = sorted({row["scenario"] for row in rows})
    methods = (
        ("gp_crps_sd_units", "GP CRPS/SD", "C0"),
        ("bift_binary_chi2", r"real BIFT $\chi_r^2$ (const)", "C3"),
    )

    for scenario in scenarios:
        selected = [row for row in rows if row["scenario"] == scenario and row["rescale_mode"] == "C"]
        selected_n = [row for row in rows if row["scenario"] == scenario and row["rescale_mode"] == "N"]
        truth = {}
        for row in selected:
            truth[row["simulation"]] = float(row["true_discrepancy"])
        labels = sorted(truth, key=lambda name: truth[name])

        fig, (scatter_ax, heat_ax, rank_ax) = plt.subplots(
            1, 3, figsize=(19, 6.2), constrained_layout=True, gridspec_kw={"width_ratios": (1.15, 1, 1)}
        )
        medians, spreads = {}, {}
        medians["gp"] = median_scores(selected, "gp_crps_sd_units")
        spreads["gp"] = spread(selected, "gp_crps_sd_units")
        medians["bift_c"] = median_scores(selected, "bift_binary_chi2")
        spreads["bift_c"] = spread(selected, "bift_binary_chi2")
        medians["bift_n"] = median_scores(selected_n, "bift_binary_chi2")
        spreads["bift_n"] = spread(selected_n, "bift_binary_chi2")

        plot_spec = (
            ("gp", medians["gp"], spreads["gp"], "GP CRPS/SD", "C0"),
            ("bift_c", medians["bift_c"], spreads["bift_c"], r"real BIFT $\chi_r^2$ (const)", "C3"),
            ("bift_n", medians["bift_n"], spreads["bift_n"], r"real BIFT $\chi_r^2$ (non-const)", "C2"),
        )
        for _, med, spr, label, color in plot_spec:
            x = np.array([truth[name] for name in labels])
            y = np.array([med[name] for name in labels])
            lo = np.array([spr[name][0] for name in labels])
            hi = np.array([spr[name][1] for name in labels])
            scatter_ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", ms=4, color=color, label=label, capsize=2, lw=0.8)
        for name in labels:
            scatter_ax.annotate(name, (truth[name], medians["gp"][name]), xytext=(3, 3), textcoords="offset points", fontsize=6)
        scatter_ax.set(
            xlabel="True discrepancy (relative q-weighted RMSE)",
            ylabel="Score (median over realizations; bars = min/max)",
            title=f"{scenario}: score vs truth",
        )
        scatter_ax.grid(alpha=0.2)
        scatter_ax.legend(frameon=False, fontsize=8)

        columns = [
            np.array([truth[name] / max(truth.values()) for name in labels]),
            np.array([medians["gp"][name] / max(medians["gp"].values()) for name in labels]),
            np.array([medians["bift_c"][name] / max(medians["bift_c"].values()) for name in labels]),
            np.array([medians["bift_n"][name] / max(medians["bift_n"].values()) for name in labels]),
        ]
        image = heat_ax.imshow(np.column_stack(columns), cmap="RdYlGn_r", aspect="auto", vmin=0)
        heat_ax.set_yticks(np.arange(len(labels)), labels, fontsize=6)
        heat_ax.set_xticks([0, 1, 2, 3], ["truth\n(max)", "GP\n(max)", "BIFT-C\n(max)", "BIFT-N\n(max)"])
        heat_ax.set_title("Median scores sorted by true discrepancy (each col. own scale)")
        fig.colorbar(image, ax=heat_ax, shrink=0.8)

        replicates = sorted({int(row["replicate"]) for row in selected})
        gp_ranks, bift_ranks = [], []
        for replicate in replicates:
            block = [row for row in selected if int(row["replicate"]) == replicate]
            order_gp = np.argsort([float(row["gp_crps_sd_units"]) for row in block])
            order_bift = np.argsort([float(row["bift_binary_chi2"]) for row in block])
            rank_gp = np.empty(len(block))
            rank_bift = np.empty(len(block))
            rank_gp[order_gp] = np.arange(1, len(block) + 1)
            rank_bift[order_bift] = np.arange(1, len(block) + 1)
            truth_ranks = np.array([float(row["true_rank"]) for row in block])
            gp_ranks.append(np.corrcoef(truth_ranks, rank_gp)[0, 1] ** 2 * np.sign(np.corrcoef(truth_ranks, rank_gp)[0, 1]))
            bift_ranks.append(
                np.corrcoef(truth_ranks, rank_bift)[0, 1] ** 2 * np.sign(np.corrcoef(truth_ranks, rank_bift)[0, 1])
            )
            rank_ax.scatter(rank_gp, rank_bift, s=14, alpha=0.45, color="0.4", edgecolor="none")
        limit = len(labels) + 1
        rank_ax.plot([0, limit], [0, limit], "--", color=".4", lw=1)
        rho_method = spearmanr(gp_ranks, bift_ranks)[0]
        s = summary[scenario]
        rank_ax.set(
            xlim=(0, limit),
            ylim=(0, limit),
            xlabel="GP rank",
            ylabel=r"real BIFT $\chi_r^2$ rank (const)",
            title=(
                f"Per-realization ranks; method Spearman {rho_method:.3f}\n"
                f"vs truth: GP {float(s['gp_rho_median']):.3f}, "
                f"BIFT {float(s['bift_C_rho_median']):.3f}, "
                f"BIFT-N {float(s['bift_N_rho_median']):.3f}"
            ),
        )
        rank_ax.grid(alpha=0.2)
        fig.savefig(args.output_dir / f"{scenario}.png", dpi=220)
        plt.close(fig)
        print(f"{scenario}: wrote figure")


if __name__ == "__main__":
    main()
