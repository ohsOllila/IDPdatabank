"""Score synthetic simulations with the REAL BIFT binary and the notebook pipeline.

Runs the compiled GenApp-BayesApp bift.f on every synthetic-experiment
realization (accepting all defaults, exactly like saxs_bift_comparison_SO.ipynb),
then scores all simulations with the notebook's exact convention:

- use (q, I, sigma) from BIFT's rescale.dat -- the rebinned curve with
  error-corrected sigma (constant or q-dependent factor, whichever BIFT chose),
- compute_chi2 exactly as the notebook: cubic interpolation of the simulation
  onto the BIFT q grid, unweighted affine (scale + offset) lstsq fit, residuals
  normalized by sigma, reduced by (n - 1).

GP scores are read from the existing per-realization saxs_gp_intensity.csv
files produced by run_synthetic_truth_study.py (Laplace fits for
synthetic_truth_replicates); nothing GP-related is recomputed.

Outputs outputs/<tag>/bift_scores.csv (one row per realization x simulation)
and bift_summary.csv with the median [min, max] Spearman rho against the exact
discrepancy ranking across realizations, alongside the same statistics for the
GP on identical realizations.
"""

import argparse
import csv
import shutil
import subprocess
import zlib
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
from scipy.stats import spearmanr

import run_synthetic_truth_study as study


def notebook_chi2(q_exp, i_exp, sigma_exp, q_sim, i_sim):
    """saxs_bift_comparison_SO.ipynb compute_chi2, replicated exactly (cubic interp)."""
    q_min, q_max = max(q_sim.min(), q_exp.min()), min(q_sim.max(), q_exp.max())
    mask = (q_exp >= q_min) & (q_exp <= q_max)
    if mask.sum() < 3:
        return np.nan, np.nan
    interpolated = griddata(q_sim, i_sim, q_exp[mask], method="cubic")
    if np.isnan(interpolated).any():
        raise ValueError("cubic interpolation left NaNs inside the overlap")
    design = np.column_stack((interpolated, np.ones_like(interpolated)))
    scale, offset = np.linalg.lstsq(design, i_exp[mask], rcond=None)[0]
    residual = (i_exp[mask] - (interpolated * scale + offset)) / sigma_exp[mask]
    dof = mask.sum() - 1
    return float(np.sum(residual**2) / dof), float(scale)


def run_bift(bift_exec, work_dir, q, intensity, sigma, rescale_mode="C", sigma_floor=1.0):
    """Run the binary in work_dir with the notebook's default-accepting input.

    rescale_mode selects BIFT's error-correction mode at its 15th stdin prompt:
    'C' (blank answer, the notebook's default) rescales sigma by a constant
    factor; 'N' lets BIFT apply q-/intensity-dependent correction factors so
    chi_r^2 ~ 1 per Shannon channel.

    sigma_floor: BIFT's reader rejects rows with sigma <= 0, which the
    counting-statistics scenario produces where the truth intensity decays
    to ~0. The floor matches real data (sigma always positive) and is
    negligible against the ~300-count noise level.
    """
    if rescale_mode not in ("C", "N"):
        raise ValueError(f"unknown rescale mode {rescale_mode}")
    work_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        work_dir / "experimental.dat",
        np.column_stack([q, intensity, np.maximum(sigma, sigma_floor)]),
        fmt="%.6e",
    )
    answers = [""] * 17
    answers[14] = rescale_mode
    (work_dir / "inputfile.dat").write_text("experimental.dat\n" + "\n".join(answers) + "\n")
    with (work_dir / "inputfile.dat").open() as stdin_handle, (work_dir / "bift.log").open("w") as log_handle:
        subprocess.run(
            [str(bift_exec)],
            cwd=work_dir,
            stdin=stdin_handle,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    rescale_file = work_dir / "rescale.dat"
    if not rescale_file.exists():
        raise RuntimeError(f"bift did not produce rescale.dat in {work_dir}; see {work_dir / 'bift.log'}")
    rescaled = np.loadtxt(rescale_file)
    return rescaled[:, 0], rescaled[:, 1], rescaled[:, 2]


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates-dir", type=Path, default=here / "outputs/synthetic_truth_replicates")
    parser.add_argument("--study-output-dir", type=Path, default=here / "outputs/synthetic_truth_replicates")
    parser.add_argument("--bift-exec", type=Path, default=here.parent / "bift" / "bift")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here.parent / "outputs" / "synthetic_truth_bift_binary",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--scenarios", action="append")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.bift_exec.exists():
        raise FileNotFoundError(f"bift binary not found at {args.bift_exec}")

    # Rebuild the truth and simulation set deterministically, exactly as the study does.
    template = np.genfromtxt(here / "outputs/databank_quality/SASDQK7/saxs_gp_intensity.csv", delimiter=",", names=True)
    q_grid = np.sort(np.unique(np.asarray(template["q_per_A"], float)))
    r = np.linspace(0, study.R_MAX, study.R_POINTS)
    weights = np.full(study.R_POINTS, r[1])
    weights[[0, -1]] *= 0.5
    operator = np.sinc(np.outer(q_grid, r) / np.pi) * weights
    s1 = study.solve_s1(r, weights, study.RG_TARGET_NM, 2.5, 0.30)
    sims_real = study.simulation_set(r, weights, operator, s1, 2.5, 0.30)
    truth = study.q_intensity(sims_real["truth_000"][1], operator) * 1.0
    truth *= study.I_ZERO / truth[0]
    q_sim = np.linspace(q_grid[0], q_grid[-1], 400)
    sim_curves = {}
    for name, (label, density) in sims_real.items():
        density = density * study.I_ZERO / study.q_intensity(density, operator)[0]
        sim_curves[name] = np.interp(q_sim, q_grid, study.q_intensity(density, operator))
    discrepancies = {
        name: study.true_discrepancy(q_sim, curve, q_grid, truth) for name, curve in sim_curves.items()
    }
    truth_rank = {}
    for rank, name in enumerate(sorted(discrepancies, key=lambda item: discrepancies[item]), start=1):
        truth_rank[name] = rank

    scenarios = study.scenario_definitions(q_grid, truth, study.NOISE_SIGMA)
    if args.scenarios:
        scenarios = {name: scenarios[name] for name in args.scenarios}

    rows = []
    shutil.rmtree(args.output_dir / "bift_runs", ignore_errors=True)
    for replicate in range(args.replicates):
        seed_replicate = args.seed + 1000 * replicate
        for name, definition in scenarios.items():
            scenario_rng = np.random.default_rng([seed_replicate, zlib.crc32(name.encode("utf-8"))])
            realization = (
                truth + definition["systematic"] + scenario_rng.normal(0, 1, len(truth)) * definition["noise"]
            )
            bift_results = {}
            for mode in ("C", "N"):
                bift_results[mode] = run_bift(
                    args.bift_exec,
                    args.output_dir / "bift_runs" / f"{study.safe_name(name)}_rep{replicate}_{mode}",
                    q_grid,
                    realization,
                    definition["sigma"],
                    rescale_mode=mode,
                )
            gp = np.genfromtxt(
                args.study_output_dir / f"{study.safe_name(name)}_rep{replicate}" / "saxs_gp_intensity.csv",
                delimiter=",",
                names=True,
            )
            for sim_name, curve in sim_curves.items():
                gp_score = study.score_simulation(
                    q_sim, curve, gp["q_per_A"], gp["I_mean"], gp["I_predictive_sd"], fit_background=True
                )[0]["gp_crps_sd_units"]
                for mode, (bift_q, bift_i, bift_sigma) in bift_results.items():
                    chi2, _ = notebook_chi2(bift_q, bift_i, bift_sigma, q_sim, curve)
                    rows.append(
                        {
                            "scenario": name,
                            "replicate": replicate,
                            "rescale_mode": mode,
                            "simulation": sim_name,
                            "true_discrepancy": discrepancies[sim_name],
                            "true_rank": truth_rank[sim_name],
                            "bift_binary_chi2": chi2,
                            "gp_crps_sd_units": gp_score,
                        }
                    )
        print(f"replicate {replicate + 1}/{args.replicates} done", flush=True)

    with (args.output_dir / "bift_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def replicate_stats(selected, column):
        rhos, top5s = [], []
        for replicate in range(args.replicates):
            block = [row for row in selected if row["replicate"] == replicate]
            true_order = [row["simulation"] for row in sorted(block, key=lambda row: row["true_rank"])]
            method_order = [row["simulation"] for row in sorted(block, key=lambda row: row[column])]
            rhos.append(spearmanr([row["true_discrepancy"] for row in block], [row[column] for row in block])[0])
            top5s.append(len(set(true_order[:5]) & set(method_order[:5])) / 5)
        return rhos, top5s

    summary = []
    for name in sorted(scenarios):
        selected = [row for row in rows if row["scenario"] == name]
        entry = {"scenario": name, "replicates": args.replicates}
        for mode in ("C", "N"):
            mode_rows = [row for row in selected if row["rescale_mode"] == mode]
            rhos, top5s = replicate_stats(mode_rows, "bift_binary_chi2")
            entry[f"bift_{mode}_rho_median"] = float(np.median(rhos))
            entry[f"bift_{mode}_rho_min"] = float(np.min(rhos))
            entry[f"bift_{mode}_rho_max"] = float(np.max(rhos))
            entry[f"bift_{mode}_top5_median"] = float(np.median(top5s))
        rhos, top5s = replicate_stats(
            [row for row in selected if row["rescale_mode"] == "C"], "gp_crps_sd_units"
        )
        entry["gp_rho_median"] = float(np.median(rhos))
        entry["gp_rho_min"] = float(np.min(rhos))
        entry["gp_rho_max"] = float(np.max(rhos))
        entry["gp_top5_median"] = float(np.median(top5s))
        summary.append(entry)
    with (args.output_dir / "bift_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {args.output_dir}/bift_summary.csv")


if __name__ == "__main__":
    main()
