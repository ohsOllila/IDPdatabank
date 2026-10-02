#!/usr/bin/env python3
"""Compare all 4 model combinations at noise=20, cutoff=800.

Models: {laplace,nuts} × {constant,quadratic}
Metrics shown: RMSE, Coverage, NLPD  (chi² excluded)
Space: R only
Output: compare_models_noise20_cutoff800.png
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── config ────────────────────────────────────────────────────────────────────
SWEEP_DIR = Path(__file__).parent / "results" / "sweep"
NOISE     = 30
CUTOFF    = 800
METRICS   = ["rmse", "coverage", "nlpd"]
SPACES    = ["r"]

MODELS = [
    ("laplace", "constant"),
    ("laplace", "quadratic"),
    ("nuts",    "constant"),
    ("nuts",    "quadratic"),
]

COLORS = {
    ("laplace", "constant"):   "#4c78a8",
    ("laplace", "quadratic"):  "#72b7b2",
    ("nuts",    "constant"):   "#f58518",
    ("nuts",    "quadratic"):  "#e45756",
}

LABEL = {
    ("laplace", "constant"):   "Laplace\nconstant",
    ("laplace", "quadratic"):  "Laplace\nquadratic",
    ("nuts",    "constant"):   "NUTS\nconstant",
    ("nuts",    "quadratic"):  "NUTS\nquadratic",
}

METRIC_LABEL = {"rmse": "RMSE", "coverage": "Coverage", "nlpd": "NLPD"}
SPACE_LABEL  = {"q": "Q-space", "r": "R-space"}

# lower=better for rmse/nlpd; higher=better for coverage
HIGHER_BETTER = {"coverage"}


# ── data loading ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    rows = []
    for inf, nm in MODELS:
        pattern = f"noise{NOISE}_cutoff{CUTOFF}_{inf}_{nm}"
        combo_dir = SWEEP_DIR / pattern
        if not combo_dir.exists():
            print(f"WARNING: missing {combo_dir}")
            continue
        ts_dirs = sorted(combo_dir.iterdir())
        if not ts_dirs:
            print(f"WARNING: no timestamped subdirs in {combo_dir}")
            continue
        metrics_file = ts_dirs[-1] / "all_metrics.json"
        if not metrics_file.exists():
            print(f"WARNING: no all_metrics.json in {ts_dirs[-1]}")
            continue
        with open(metrics_file) as f:
            data = json.load(f)
        for idx, mdict in data.items():
            row = {"inference": inf, "noise_model": nm, "idx": int(idx)}
            for space in SPACES:
                for metric in METRICS:
                    row[f"{space}_{metric}"] = mdict.get(space, {}).get(metric, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


# ── plotting ──────────────────────────────────────────────────────────────────
def make_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    n_spaces  = len(SPACES)
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(
        n_spaces, n_metrics,
        figsize=(4.5 * n_metrics, 4.0 * n_spaces),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(
        f"Model comparison — noise={NOISE}%, cutoff={CUTOFF}",
        fontsize=14, fontweight="bold",
    )

    positions = np.arange(len(MODELS))
    width = 0.55

    for si, space in enumerate(SPACES):
        for mi, metric in enumerate(METRICS):
            ax = axes[si][mi]
            col = f"{space}_{metric}"
            series_list = []
            colors_list = []
            for inf, nm in MODELS:
                mask = (df["inference"] == inf) & (df["noise_model"] == nm)
                vals = df.loc[mask, col].dropna().values
                series_list.append(vals)
                colors_list.append(COLORS[(inf, nm)])

            show_fliers = metric == "coverage"
            bps = ax.boxplot(
                series_list,
                positions=positions,
                widths=width,
                patch_artist=True,
                showfliers=show_fliers,
                medianprops=dict(color="black", linewidth=1.8),
                flierprops=dict(marker=".", markersize=3, alpha=0.4),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
            )
            for patch, c in zip(bps["boxes"], colors_list):
                patch.set_facecolor(c)
                patch.set_alpha(0.75)

            # jittered points
            rng = np.random.default_rng(0)
            for xi, (vals, c) in enumerate(zip(series_list, colors_list)):
                jitter = rng.uniform(-0.18, 0.18, len(vals))
                ax.scatter(
                    np.full(len(vals), xi) + jitter, vals,
                    color=c, alpha=0.25, s=6, zorder=2,
                )

            # mean markers
            for xi, vals in enumerate(series_list):
                if len(vals):
                    ax.scatter(xi, np.mean(vals), marker="D", color="white",
                               edgecolors="black", s=30, zorder=5, linewidths=0.8)

            ax.set_title(METRIC_LABEL[metric], fontsize=10)
            ax.set_xticks(positions)
            ax.set_xticklabels(
                [LABEL[(inf, nm)] for inf, nm in MODELS],
                fontsize=8,
            )
            ax.grid(axis="y", alpha=0.3, linestyle="--")

            top = 20 if metric in ("rmse", "nlpd") else None
            ax.set_ylim(bottom=0, top=top)
            arrow = "↑ better" if metric in HIGHER_BETTER else "↓ better"
            ax.set_ylabel(f"{METRIC_LABEL[metric]}  ({arrow})", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close(fig)


def make_summary_table(df: pd.DataFrame) -> None:
    print(f"\n{'='*80}")
    print(f"MEAN METRICS — noise={NOISE}%  cutoff={CUTOFF}")
    print(f"{'='*80}")
    header = f"{'Model':<22}"
    for space in SPACES:
        for metric in METRICS:
            header += f"  {METRIC_LABEL[metric]:>8}"
    print(header)
    print("-" * len(header))
    for inf, nm in MODELS:
        mask = (df["inference"] == inf) & (df["noise_model"] == nm)
        sub = df[mask]
        line = f"{LABEL[(inf, nm)].replace(chr(10), ' '):<22}"
        for space in SPACES:
            for metric in METRICS:
                v = sub[f"{space}_{metric}"].mean()
                line += f"  {v:>12.4f}"
        print(line)
    print("=" * 80)


def make_rank_plot(df: pd.DataFrame, out_path: Path) -> None:
    """Per-curve rank comparison: for each curve, rank models 1–4 per metric."""
    fig, axes = plt.subplots(
        len(SPACES), len(METRICS),
        figsize=(4.5 * len(METRICS), 3.5 * len(SPACES)),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(
        f"Per-curve model ranks — noise={NOISE}%, cutoff={CUTOFF}\n"
        "(rank 1 = best for each curve)",
        fontsize=13, fontweight="bold",
    )

    for si, space in enumerate(SPACES):
        for mi, metric in enumerate(METRICS):
            ax = axes[si][mi]
            col = f"{space}_{metric}"
            # pivot: rows=idx, cols=model
            pivot = df.pivot_table(index="idx", columns=["inference", "noise_model"], values=col)
            if metric in HIGHER_BETTER:
                ranks = pivot.rank(axis=1, ascending=False)
            else:
                ranks = pivot.rank(axis=1, ascending=True)

            positions = np.arange(len(MODELS))
            rank_data = [ranks[(inf, nm)].dropna().values for inf, nm in MODELS]
            bps = ax.boxplot(
                rank_data,
                positions=positions,
                widths=0.5,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1.8),
                flierprops=dict(marker=".", markersize=3, alpha=0.4),
            )
            for patch, (inf, nm) in zip(bps["boxes"], MODELS):
                patch.set_facecolor(COLORS[(inf, nm)])
                patch.set_alpha(0.75)

            ax.set_title(METRIC_LABEL[metric], fontsize=10)
            ax.set_xticks(positions)
            ax.set_xticklabels([LABEL[(inf, nm)] for inf, nm in MODELS], fontsize=8)
            ax.set_ylabel("Rank (1=best)", fontsize=8)
            ax.set_yticks([1, 2, 3, 4])
            ax.invert_yaxis()
            ax.grid(axis="y", alpha=0.3, linestyle="--")

    rank_out = out_path.with_name(out_path.stem + "_ranks.png")
    fig.savefig(rank_out, dpi=150)
    print(f"Saved → {rank_out}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading noise={NOISE}% cutoff={CUTOFF} results …")
    df = load_data()
    if df.empty:
        print("No data found. Check SWEEP_DIR and noise/cutoff values.")
        raise SystemExit(1)

    n_curves = df.groupby(["inference", "noise_model"])["idx"].nunique()
    print(f"Loaded {len(df)} rows across {n_curves.to_dict()} curves per model.")

    make_summary_table(df)

    out_dir = Path(__file__).parent / "results"
    out_path = out_dir / f"compare_models_noise{NOISE}_cutoff{CUTOFF}.png"
    make_boxplot(df, out_path)
    make_rank_plot(df, out_path)
