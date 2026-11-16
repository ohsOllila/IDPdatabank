"""Create absolute GP-CRPS quality heatmaps for each experimental dataset."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from run_gp_quality import read_simulation


def safe_name(value):
    return value.replace("/", "_")


def read_experiment(path):
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            rows.append((float(fields[0]), float(fields[1]), float(fields[2]) if len(fields) >= 3 else np.nan))
        except ValueError:
            continue
    data = np.asarray(rows)
    q = data[:, 0]
    if q.max() > 2:
        q = q / 10
    order = np.argsort(q)
    return q[order], data[order, 1], data[order, 2]


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        type=Path,
        default=here / "outputs/databank_quality/gp_quality_scores.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "outputs/databank_quality/crps_quality",
    )
    args = parser.parse_args()
    root = here.parents[3]

    with args.scores.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    experiments = sorted({row["experiment"] for row in rows})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for experiment in experiments:
        selected = [row for row in rows if row["experiment"] == experiment]
        selected.sort(key=lambda row: float(row["gp_crps_sd_units"]))
        scores = np.array([float(row["gp_crps_sd_units"]) for row in selected])
        lower, upper = float(scores.min()), float(scores.max())
        if np.isclose(lower, upper):
            upper = lower * (1 + 1e-6)
        normalization = LogNorm(vmin=lower, vmax=upper)
        labels = [f"{row['system']}:{row['simulation_id']}" for row in selected]
        stem = safe_name(experiment)

        with (args.output_dir / f"{stem}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["simulation", "gp_crps_sd_units"])
            writer.writerows(zip(labels, scores))  # noqa: B905 -- scores are derived from labels

        height = max(4.5, min(14.0, 1.8 + 0.22 * len(labels)))
        fig, (quality_ax, curve_ax) = plt.subplots(
            1,
            2,
            figsize=(13.5, height),
            constrained_layout=True,
            gridspec_kw={"width_ratios": [1, 2.2]},
        )
        colormap = plt.get_cmap("RdYlGn_r")
        image = quality_ax.imshow(scores[:, None], cmap=colormap, norm=normalization, origin="upper", aspect="auto")
        fontsize = 6 if len(labels) > 25 else 8
        quality_ax.set_yticks(np.arange(len(labels)), labels, fontsize=fontsize)
        quality_ax.set_xticks([0], ["Integrated CRPS/SD"])
        quality_ax.set_ylabel("Simulation (best to worst)")
        quality_ax.set_title("Absolute GP agreement\n(lower is better; dataset-specific scale)")
        for index, score in enumerate(scores):
            color = "white" if score > np.sqrt(lower * upper) else "black"
            quality_ax.text(0, index, f"{score:.3f}", ha="center", va="center", fontsize=fontsize, color=color)

        experiment_path = root / "Data/Experiments/saxs" / experiment / "saxs.dat"
        q_exp, intensity_exp, sigma_exp = read_experiment(experiment_path)
        q_min = min(float(row["q_min_per_A"]) for row in selected)
        q_max = max(float(row["q_max_per_A"]) for row in selected)
        use_exp = (q_exp >= q_min) & (q_exp <= q_max) & (intensity_exp > 0)
        finite_sigma = np.isfinite(sigma_exp) & (sigma_exp > 0)
        if np.any(finite_sigma & use_exp):
            curve_ax.errorbar(
                q_exp[use_exp],
                intensity_exp[use_exp],
                yerr=np.where(finite_sigma[use_exp], sigma_exp[use_exp], 0),
                fmt=".",
                markersize=2.5,
                linewidth=0.5,
                color="black",
                alpha=0.55,
                zorder=5,
                label="Experiment",
            )
        else:
            curve_ax.plot(
                q_exp[use_exp],
                intensity_exp[use_exp],
                ".",
                markersize=3,
                color="black",
                alpha=0.65,
                zorder=5,
                label="Experiment",
            )
        for row, score in zip(selected, scores):  # noqa: B905 -- scores are derived from selected
            q_sim, intensity_sim = read_simulation(root / row["simulation_path"] / "SAXS.yaml")
            fitted = float(row["scale_sim_to_experiment"]) * intensity_sim + float(row["background"])
            use = (q_sim >= float(row["q_min_per_A"])) & (q_sim <= float(row["q_max_per_A"])) & (fitted > 0)
            curve_ax.plot(q_sim[use], fitted[use], color=colormap(normalization(score)), linewidth=1.0, alpha=0.72)
        curve_ax.set_yscale("log")
        curve_ax.set_xlabel(r"$q$ (Å$^{-1}$)")
        curve_ax.set_ylabel(r"$I(q)$ (a.u.)")
        curve_ax.set_title(f"{experiment}: experiment and fitted simulations")
        curve_ax.grid(alpha=0.18)
        curve_ax.legend(frameon=False)

        colorbar = fig.colorbar(image, ax=(quality_ax, curve_ax), shrink=0.82)
        colorbar.set_label("Integrated CRPS / mean GP predictive SD")
        fig.savefig(args.output_dir / f"{stem}.png", dpi=450)
        plt.close(fig)
        print(f"{experiment}: {len(labels)} simulations")


if __name__ == "__main__":
    main()
