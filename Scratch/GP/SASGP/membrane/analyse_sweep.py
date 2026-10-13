#!/usr/bin/env python3
"""Load all sweep results, compute and print average metrics, and plot comparisons."""

import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SWEEP_DIR = Path(__file__).parent / "results" / "sweep"
METRICS = ["rmse", "chi2", "coverage", "nlpd"]
METRIC_LABEL = {"chi2": "χ²"}
SPACES  = ["r"]


def mlabel(m):
    return METRIC_LABEL.get(m, m.upper())


def load_all():
    rows = []
    for combo_dir in sorted(SWEEP_DIR.iterdir()):
        m = re.match(
            r"noise(\d+)_cutoff(\d+)_(\w+)_(\w+)", combo_dir.name
        )
        if not m:
            continue
        noise, cutoff, inference, noise_model = m.groups()
        # Take the single (or latest) timestamped subdir
        ts_dirs = sorted(combo_dir.iterdir())
        if not ts_dirs:
            continue
        metrics_file = ts_dirs[-1] / "all_metrics.json"
        if not metrics_file.exists():
            continue
        with open(metrics_file) as f:
            data = json.load(f)

        for idx, mdict in data.items():
            row = {
                "noise": int(noise),
                "cutoff": int(cutoff),
                "inference": inference,
                "noise_model": noise_model,
                "idx": int(idx),
            }
            for space in SPACES:
                for metric in METRICS:
                    val = mdict.get(space, {}).get(metric, np.nan)
                    row[f"{space}_{metric}"] = val
            rows.append(row)
    return pd.DataFrame(rows)


def print_summary(df):
    group_cols = ["noise", "cutoff", "inference", "noise_model"]
    agg = df.groupby(group_cols)[[f"{s}_{m}" for s in SPACES for m in METRICS]].mean()
    agg = agg.reset_index().sort_values(group_cols)

    print("\n" + "=" * 100)
    print("AVERAGE METRICS PER HYPERPARAMETER COMBINATION")
    print("=" * 100)
    header = f"{'Noise':>6} {'Cutoff':>7} {'Inference':>10} {'NsModel':>10}"
    for s in SPACES:
        for metric in METRICS:
            header += f"  {s.upper()}_{mlabel(metric):>8}"
    print(header)
    print("-" * len(header))
    for _, row in agg.iterrows():
        line = f"{row['noise']:>6} {row['cutoff']:>7} {row['inference']:>10} {row['noise_model']:>10}"
        for s in SPACES:
            for metric in METRICS:
                v = row[f"{s}_{metric}"]
                line += f"  {v:>12.4f}"
        print(line)
    print("=" * 100)
    return agg


def plot_metrics_per_noise(agg):
    """One comparison scatter figure per noise level."""
    from matplotlib.lines import Line2D

    color_by_inference = {"laplace": "#4c78a8", "nuts": "#f58518"}
    marker_by_nm       = {"constant": "o", "quadratic": "s"}

    for noise_val in sorted(agg["noise"].unique()):
        sub = agg[agg["noise"] == noise_val].copy()
        sub["label"] = sub.apply(
            lambda r: f"c{r['cutoff']}\n{r['inference']}\n{r['noise_model']}", axis=1
        )

        fig, axes = plt.subplots(
            len(SPACES), len(METRICS),
            figsize=(4.5 * len(METRICS), 3.5 * len(SPACES)),
            constrained_layout=True,
            squeeze=False,
        )
        fig.suptitle(f"Average metrics — noise level {noise_val}%", fontsize=13)

        for si, space in enumerate(SPACES):
            for mi, metric in enumerate(METRICS):
                ax = axes[si][mi]
                col = f"{space}_{metric}"
                sorted_sub = sub.sort_values(col)
                x = np.arange(len(sorted_sub))
                for xi, (_, row) in zip(x, sorted_sub.iterrows()):
                    ax.scatter(
                        xi, row[col],
                        c=color_by_inference[row["inference"]],
                        marker=marker_by_nm[row["noise_model"]],
                        s=80, zorder=3,
                    )
                ax.set_title(f"{space.upper()} — {mlabel(metric)}", fontsize=10)
                ax.set_xticks(x)
                ax.set_xticklabels(sorted_sub["label"], fontsize=7, rotation=90)
                ax.grid(axis="y", alpha=0.3)

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#4c78a8", markersize=8, label="laplace"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#f58518", markersize=8, label="nuts"),
            Line2D([0], [0], marker="o", color="grey", markersize=8, label="constant noise"),
            Line2D([0], [0], marker="s", color="grey", markersize=8, label="quadratic noise"),
        ]
        fig.legend(handles=legend_elements, loc="lower right", fontsize=8, ncol=4)

        out = Path(__file__).parent / "results" / f"sweep_metrics_noise{noise_val}.png"
        fig.savefig(out, dpi=150)
        print(f"Plot saved → {out}")
        plt.close(fig)


def plot_heatmaps_per_noise(agg):
    """One heatmap figure per noise level: rows = inference×noise_model, cols = cutoff."""
    inferences   = sorted(agg["inference"].unique())
    noise_models = sorted(agg["noise_model"].unique())
    cutoffs      = sorted(agg["cutoff"].unique())

    row_keys = [(inf, nm) for inf in inferences for nm in noise_models]

    for noise_val in sorted(agg["noise"].unique()):
        sub = agg[agg["noise"] == noise_val]

        fig, axes = plt.subplots(
            len(SPACES), len(METRICS),
            figsize=(3.0 * len(METRICS), 2.8 * len(SPACES)),
            constrained_layout=True,
            squeeze=False,
        )
        fig.suptitle(
            f"Metric heatmaps — noise level {noise_val}%\n",
            fontsize=11,
        )

        for si, space in enumerate(SPACES):
            for mi, metric in enumerate(METRICS):
                ax = axes[si][mi]
                col = f"{space}_{metric}"
                mat = np.full((len(row_keys), len(cutoffs)), np.nan)
                for ri, (inf, nm) in enumerate(row_keys):
                    for ci, c in enumerate(cutoffs):
                        mask = (
                            (sub["inference"] == inf) &
                            (sub["noise_model"] == nm) &
                            (sub["cutoff"] == c)
                        )
                        vals = sub.loc[mask, col]
                        if len(vals):
                            mat[ri, ci] = vals.iloc[0]

                im = ax.imshow(mat, aspect="auto", cmap="magma")
                fig.colorbar(im, ax=ax, shrink=0.8)
                ax.set_title(f"{mlabel(metric)}", fontsize=9)
                ax.set_xticks(range(len(cutoffs)))
                ax.set_xticklabels([str(int(c/100)) for c in cutoffs], fontsize=8)
                ax.set_xlabel("cutoff (Å⁻¹)", fontsize=8)
                ax.set_yticks(range(len(row_keys)))
                ax.set_yticklabels([f"{inf}/{nm}" for inf, nm in row_keys], fontsize=7)

        out = Path(__file__).parent / "results" / f"sweep_heatmap_noise{noise_val}.png"
        fig.savefig(out, dpi=150)
        print(f"Heatmap saved → {out}")
        plt.close(fig)


def print_best_per_noise(agg):
    for noise_val in sorted(agg["noise"].unique()):
        sub = agg[agg["noise"] == noise_val]
        print(f"\nBEST COMBO — noise={noise_val}% (lowest = better, except coverage)")
        for space in SPACES:
            for metric in METRICS:
                col = f"{space}_{metric}"
                if sub[col].isna().all():
                    continue
                if metric == "coverage":
                    row = sub.loc[sub[col].idxmax()]
                    direction = "↑"
                else:
                    row = sub.loc[sub[col].idxmin()]
                    direction = "↓"
                print(
                    f"  {space.upper()} {mlabel(metric):8s} {direction}: "
                    f"cutoff={int(row['cutoff'])}, inference={row['inference']}, "
                    f"noise_model={row['noise_model']}  →  {row[col]:.4f}"
                )


if __name__ == "__main__":
    print("Loading sweep results …")
    df = load_all()
    print(f"Loaded {len(df)} sample-rows from {df.groupby(['noise','cutoff','inference','noise_model']).ngroups} combos.")

    agg = print_summary(df)
    print_best_per_noise(agg)
    plot_metrics_per_noise(agg)
    plot_heatmaps_per_noise(agg)
