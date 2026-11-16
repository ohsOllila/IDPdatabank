"""Fit sinc-transform GP models to the databank's experimental SAXS curves.

For every experimental SAXS dataset linked from at least one simulation
README.yaml, this script fits the GP model of fairmd.idp.saxs_gp_fit and
stores the portable fit summary as

    Data/Experiments/saxs/<experiment>/gp_fit.dat

which is the only artifact consumed by the quality scoring in
calcProperties.py (fairmd.idp.saxs_gp_quality.evaluate_SAXS_quality_GP).

Also written next to each experiment: gp_fit.png (fit diagnostics) and, when
two or more linked simulations can be scored, gp_quality_ranking.csv (all
linked simulations ranked by GP CRPS/SD with the full score set, best first)
and gp_quality_ranking.png (the same ranking as a heatmap joined with fitted
curves over the experimental data and GP mean).

Full fit diagnostics (hyperparameter chains, covariance, P(r), plots) are
cached under --cache-dir, defaulting to the Scratch pipeline's
outputs/databank_quality directory so existing fits are reused instead of
refitted; pass --refit to recompute. Fitting requires torch and pyro; the
exported gp_fit.dat removes that requirement from all downstream scoring.

Run from the repository root:

    python3 Scripts/AnalyzeDatabank/calcSAXSGPFits.py [--experiment ID ...]
        [--refit] [--cache-dir DIR] [--q-max 0.5] [--max-points 200]
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fairmd.idp import saxs_gp_fit

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def linked_experiments(root, with_simulations=False):
    """Experiment IDs linked from at least one simulation README.yaml.

    With with_simulations=True, also returns {experiment_id: [(system, saxs_file)]}
    for use by the per-experiment quality-ranking plots.
    """
    ids = []
    simulations = {}
    for readme in (root / "Data/Simulations").rglob("README.yaml"):
        try:
            metadata = yaml.safe_load(readme.read_text())
        except yaml.YAMLError:
            continue
        if not metadata:
            continue
        experiment = metadata.get("EXPERIMENT", {}).get("saxs", {})
        saxs_file = readme.parent / "SAXS.yaml"
        for experiment_id in experiment.get("path", []) or []:
            if experiment_id not in ids:
                ids.append(experiment_id)
            if with_simulations and saxs_file.exists():
                simulations.setdefault(experiment_id, []).append((metadata, saxs_file))
    if with_simulations:
        return sorted(ids), simulations
    return sorted(ids)


def lightning_label(system):
    """Short plot label for a simulation: SYSTEM:ID."""
    return f"{system.get('SYSTEM', '?')}:{system.get('ID', '?')}"


def score_linked_simulations(sims, fit):
    """Score every linked simulation against one experiment's stored GP fit."""
    from fairmd.idp.saxs_gp_quality import read_saxs_yaml, score_simulation

    scored = []
    for system, saxs_file in sims:
        q_sim, intensity_sim = read_saxs_yaml(saxs_file)
        try:
            result, detail = score_simulation(
                q_sim, intensity_sim, fit["q_per_A"], fit["I_mean"], fit["I_predictive_sd"], fit_background=True
            )
        except ValueError as error:
            print(f"  {lightning_label(system)} skipped: {error}")
            continue
        scored.append((system, result, detail))
    return scored


def write_ranking_csv(scored, output_file):
    """Machine-readable per-experiment ranking, best (lowest CRPS/SD) first.

    Stored with the experiment as gp_quality_ranking.csv, the numeric
    companion of gp_quality_ranking.png and of the per-simulation
    SAXS_quality_GP.yaml files.
    """
    ordered = sorted(scored, key=lambda item: item[1]["gp_crps_sd_units"])
    fields = [
        "rank",
        "system",
        "simulation_id",
        "gp_crps_sd_units",
        "gp_crps",
        "gp_crps_normalized",
        "gp_z_rmse",
        "gp_mean_abs_z",
        "gp_coverage_68",
        "gp_coverage_95",
        "gp_relative_rmse",
        "scale_sim_to_experiment",
        "background",
        "q_min_per_A",
        "q_max_per_A",
        "q_points",
    ]
    import csv

    with open(output_file, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, (system, result, _) in enumerate(ordered, start=1):
            row = {"rank": rank, "system": system.get("SYSTEM"), "simulation_id": system.get("ID")}
            row.update({key: result[key] for key in fields if key in result})
            writer.writerow(row)
    return len(ordered)


def plot_experiment_quality(experiment_id, experiment_dir, scored, fit, output_file):
    """Ranked GP quality ranking for one experiment across all linked simulations.

    Mirrors the Scratch pipeline's crps_quality figures: a ranked heat column of
    integrated CRPS/SD (best to worst, dataset-specific linear color scale) plus
    the experimental data, GP mean, and every fitted simulation curve colored by
    its score. Stored with the experiment as gp_quality_ranking.png so the
    databank itself carries the comparison, independent of Scratch.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(scored) < 2:
        print(f"  too few scoreable simulations ({len(scored)}); skipping ranking plot")
        return False
    labels = [lightning_label(system) for system, _, _ in scored]
    scores = np.array([result["gp_crps_sd_units"] for _, result, _ in scored])
    order = np.argsort(scores)
    labels = [labels[index] for index in order]
    scores = scores[order]

    q_exp, intensity_exp = saxs_gp_fit.read_saxs_dat(experiment_dir / "saxs.dat")
    height = max(4.5, min(14.0, 1.8 + 0.22 * len(labels)))
    fig, (quality_ax, curve_ax) = plt.subplots(
        1, 2, figsize=(13.5, height), constrained_layout=True, gridspec_kw={"width_ratios": [1, 2.2]}
    )
    lowest, highest = scores.min(), scores.max()
    if np.isclose(lowest, highest):
        highest = lowest * (1 + 1e-6)
    from matplotlib.colors import Normalize

    normalization = Normalize(lowest, highest)
    colormap = plt.get_cmap("RdYlGn_r")
    image = quality_ax.imshow(scores[:, None], cmap=colormap, norm=normalization, origin="upper", aspect="auto")
    fontsize = 6 if len(labels) > 25 else 8
    quality_ax.set_yticks(np.arange(len(labels)), labels, fontsize=fontsize)
    quality_ax.set_xticks([0], ["Integrated CRPS/SD"])
    quality_ax.set_ylabel("Simulation (best to worst)")
    quality_ax.set_title("GP agreement (lower is better)")
    for index, score in enumerate(scores):
        color = "white" if score > np.sqrt(lowest * highest) else "black"
        quality_ax.text(0, index, f"{score:.3f}", ha="center", va="center", fontsize=fontsize, color=color)
    fig.colorbar(image, ax=quality_ax, shrink=0.82, label="Integrated CRPS / mean predictive SD")

    use_exp = np.isfinite(intensity_exp) & (intensity_exp > 0) & (q_exp <= fit["q_per_A"][-1])
    curve_ax.plot(q_exp[use_exp], intensity_exp[use_exp], ".", markersize=3, color="black", alpha=0.65, zorder=5,
                  label="Experiment")
    curve_ax.plot(fit["q_per_A"], fit["I_mean"], color="C0", lw=1.5, label="GP mean", zorder=4)
    for index in order:
        result, detail = scored[index][1], scored[index][2]
        curve_ax.plot(
            detail["q"],
            detail["simulation_fitted"],
            color=colormap(normalization(result["gp_crps_sd_units"])),
            linewidth=1.0,
            alpha=0.72,
        )
    curve_ax.set_yscale("log")
    curve_ax.set_xlabel(r"$q$ (Å$^{-1}$)")
    curve_ax.set_ylabel(r"$I(q)$ (a.u.)")
    curve_ax.set_title(f"{experiment_id}: experiment, GP mean, and fitted simulations")
    curve_ax.grid(alpha=0.18)
    curve_ax.legend(frameon=False)
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    return True


def safe_name(value):
    return str(value).replace("/", "_")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "Scratch/GP/SASGP/IDPs/outputs/databank_quality",
        help="Directory for full fit diagnostics; existing fits are reused unless --refit",
    )
    parser.add_argument("--q-max", type=float, default=0.5)
    parser.add_argument(
        "--max-points", type=int, default=200, help="Deterministic thinning limit for cubic-cost GP inference"
    )
    parser.add_argument("--nuts-warmup", type=int, default=2000)
    parser.add_argument("--nuts-samples", type=int, default=1500)
    parser.add_argument("--nuts-chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--experiment", action="append", help="Fit only this experiment; may be repeated")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip diagnostic plots (useful on headless HPC nodes)"
    )
    args = parser.parse_args()
    args.root = args.root.resolve()
    os.environ["NMLDB_ROOT_PATH"] = str(args.root)

    experiment_ids, simulations = linked_experiments(args.root, with_simulations=True)
    if args.experiment:
        missing = set(args.experiment).difference(experiment_ids)
        if missing:
            raise ValueError(f"unknown experiment(s): {', '.join(sorted(missing))}")
        experiment_ids = [item for item in experiment_ids if item in set(args.experiment)]
    print(f"{len(experiment_ids)} linked experimental SAXS dataset(s)")

    for experiment_id in experiment_ids:
        experiment_dir = args.root / "Data/Experiments/saxs" / experiment_id
        source = experiment_dir / "saxs.dat"
        if not source.exists():
            print(f"{experiment_id}: saxs.dat not found, skipping", flush=True)
            continue
        fit_dir = args.cache_dir / safe_name(experiment_id)
        if args.refit or not (fit_dir / "saxs_gp_intensity.csv").exists():
            q, intensity = saxs_gp_fit.read_saxs_dat(source)
            use = (q <= args.q_max) & np.isfinite(intensity) & (intensity > 0)
            q, intensity = saxs_gp_fit.thin_evenly(q[use], intensity[use], args.max_points)
            if len(q) < 3:
                print(f"{experiment_id}: fewer than three points at q <= {args.q_max}, skipping", flush=True)
                continue
            report = saxs_gp_fit.fit_saxs(
                q,
                intensity,
                fit_dir,
                inference="nuts",
                warmup=args.nuts_warmup,
                samples=args.nuts_samples,
                chains=args.nuts_chains,
                seed=args.seed,
                make_plots=not args.no_plots,
            )
            inference = report.get("inference", {})
            if inference.get("diagnostic_checks_passed") is False:
                print(f"{experiment_id}: WARNING NUTS diagnostics failed; fit is provisional", flush=True)
        else:
            print(f"{experiment_id}: reusing cached fit in {fit_dir}", flush=True)
        saxs_gp_fit.export_gp_fit(fit_dir, experiment_dir)
        print(f"{experiment_id}: wrote {experiment_dir / 'gp_fit.dat'}", flush=True)
        fit_plot = fit_dir / "saxs_gp_fit.png"
        if fit_plot.exists():
            shutil.copyfile(fit_plot, experiment_dir / "gp_fit.png")
            print(f"{experiment_id}: wrote {experiment_dir / 'gp_fit.png'}", flush=True)

        sims = simulations.get(experiment_id, [])
        if sims:
            from fairmd.idp.saxs_gp_quality import read_gp_fit

            fit, _ = read_gp_fit(experiment_dir / "gp_fit.dat")
            scored = score_linked_simulations(sims, fit)
            if scored:
                ranking_csv = experiment_dir / "gp_quality_ranking.csv"
                count = write_ranking_csv(scored, ranking_csv)
                print(f"{experiment_id}: wrote {ranking_csv} ({count} simulations)", flush=True)
                if not args.no_plots:
                    output_file = experiment_dir / "gp_quality_ranking.png"
                    if plot_experiment_quality(experiment_id, experiment_dir, scored, fit, output_file):
                        print(f"{experiment_id}: wrote {output_file}", flush=True)


if __name__ == "__main__":
    main()
