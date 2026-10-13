#!/usr/bin/env python3
"""
Convergence curve: mean R-hat vs. number of NUTS samples.

Runs NUTS once per simulation (max_samples total), then truncates the chain
at each checkpoint (100, 200, … max_samples) and recomputes R-hat without
re-sampling. Averages across N random simulations and saves data + plot.

Usage
-----
python membrane/convergence_curve.py <run_dir> [options]

  <run_dir>   timestamped directory containing per-index subdirs with gp.pt
              e.g. membrane/results/sweep/noise20_cutoff800_nuts_constant/20260501_035057

Options
-------
--n-sims N        number of random simulations to average over (default 10)
--chains N        NUTS chains per simulation (default 4)
--warmup N        NUTS warmup steps (default 200)
--max-samples N   total NUTS samples per chain (default 1000)
--step N          checkpoint interval (default 100)
--seed N          RNG seed for simulation selection (default 42)
--data-dir PATH   directory with all_ff_x/y.npy (default: script dir)
--config PATH     fallback TOML config if config.json absent
--out-dir PATH    output directory (default: <run_dir>)
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import gptransform
from membrane.run import deabsolute
from membrane.nuts_convergence import (
    load_cfg, reconstruct_training_data, split_rhat, bulk_ess,
)

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


def rhat_scalar(chain_np):
    """Mean R-hat across all parameters for a (C, S, P) array."""
    rhat = split_rhat(chain_np)
    return float(np.mean(rhat)), float(np.max(rhat))


def run_one_sim(idx_dir, data_dir, fallback_cfg, chains, warmup, max_samples, seed, checkpoints):
    """
    Load GP from idx_dir, run NUTS once, then compute mean/max R-hat
    at each checkpoint by truncating the chain.

    Returns
    -------
    dict with keys 'mean_rhat', 'max_rhat' — each a list aligned to checkpoints.
    None if the simulation fails.
    """
    idx = int(idx_dir.name)

    gp_path = idx_dir / "gp.pt"
    if not gp_path.exists():
        return None

    try:
        gp = torch.load(gp_path, map_location="cpu", weights_only=False)
        gp.eval()
    except Exception as e:
        print(f"  [skip {idx}] could not load gp.pt: {e}")
        return None

    try:
        cfg = load_cfg(idx_dir, fallback_cfg)
        ff_x = np.load(data_dir / "all_ff_x.npy", allow_pickle=True)
        ff_y = np.load(data_dir / "all_ff_y.npy", allow_pickle=True)
        q_vals, sq_vals, r_grid = reconstruct_training_data(idx, cfg, ff_x, ff_y)
    except Exception as e:
        print(f"  [skip {idx}] data load failed: {e}")
        return None

    try:
        mcmc, _ = gptransform.nuts_sample(
            gp, r_grid, q_vals, sq_vals,
            num_samples=max_samples,
            num_warmup=warmup,
            num_chains=chains,
            seed=seed,
        )
        chain_np = mcmc.get_samples(group_by_chain=True)["theta_raw"].cpu().numpy()
    except Exception as e:
        print(f"  [skip {idx}] NUTS failed: {e}")
        return None

    mean_rhats, max_rhats = [], []
    for T in checkpoints:
        trunc = chain_np[:, :T, :]   # (C, T, P)
        if trunc.shape[1] < 4:
            mean_rhats.append(np.nan)
            max_rhats.append(np.nan)
            continue
        m, mx = rhat_scalar(trunc)
        mean_rhats.append(m)
        max_rhats.append(mx)

    return {"mean_rhat": mean_rhats, "max_rhat": max_rhats}


def plot_curve(checkpoints, results, out_path):
    mean_arr = np.array([r["mean_rhat"] for r in results])   # (N_sims, N_checkpoints)
    max_arr  = np.array([r["max_rhat"]  for r in results])

    mu_mean = np.nanmean(mean_arr, axis=0)
    sd_mean = np.nanstd(mean_arr,  axis=0)
    mu_max  = np.nanmean(max_arr,  axis=0)
    sd_max  = np.nanstd(max_arr,   axis=0)

    n_valid = np.sum(~np.isnan(mean_arr), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    x = np.array(checkpoints)

    for ax, mu, sd, label in [
        (axes[0], mu_mean, sd_mean, "Mean R-hat (avg over params)"),
        (axes[1], mu_max,  sd_max,  "Max R-hat (worst param)"),
    ]:
        ax.fill_between(x, mu - sd, mu + sd, alpha=0.25, color="#4c78a8", label="±1 std")
        ax.plot(x, mu, color="#4c78a8", lw=2, marker="o", ms=5, label="mean across sims")
        ax.axhline(1.01, color="#e45756", lw=1.2, ls="--", label="R-hat = 1.01")
        ax.axhline(1.05, color="#f58518", lw=1.0, ls=":",  label="R-hat = 1.05")
        ax.set_xlabel("NUTS samples per chain", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    n_sims = len(results)
    fig.suptitle(
        f"NUTS convergence curve  ({n_sims} simulations, {len(checkpoints[0:])} checkpoints)\n"
        f"Shaded: ±1 std across simulations",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Curve plot → {out_path}")


def plot_per_sim(checkpoints, results, sim_indices, out_path):
    """Thin lines for each simulation on both mean and max R-hat."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    x = np.array(checkpoints)

    cmap = plt.cm.tab10
    for ax, key, ylabel in [
        (axes[0], "mean_rhat", "Mean R-hat"),
        (axes[1], "max_rhat",  "Max R-hat"),
    ]:
        for i, (r, idx) in enumerate(zip(results, sim_indices)):
            ax.plot(x, r[key], color=cmap(i % 10), lw=1.2, alpha=0.7, label=f"sim {idx}")
        ax.axhline(1.01, color="#e45756", lw=1.2, ls="--", label="1.01")
        ax.axhline(1.05, color="#f58518", lw=1.0, ls=":",  label="1.05")
        ax.set_xlabel("NUTS samples per chain", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel, fontsize=12)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    fig.suptitle("Per-simulation convergence curves", fontsize=12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Per-sim plot → {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir",      help="Timestamped run dir with per-index subdirs")
    parser.add_argument("--n-sims",     type=int, default=10)
    parser.add_argument("--chains",     type=int, default=4)
    parser.add_argument("--warmup",     type=int, default=200)
    parser.add_argument("--max-samples",type=int, default=1000)
    parser.add_argument("--step",       type=int, default=100)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--data-dir",   default=None)
    parser.add_argument("--config",     default=None)
    parser.add_argument("--out-dir",    default=None)
    args = parser.parse_args()

    if tomllib is None:
        sys.exit("tomllib/tomli required — pip install tomli")

    run_dir  = Path(args.run_dir).resolve()
    here     = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).resolve() if args.data_dir else here
    fallback = Path(args.config) if args.config else (here / "config.toml")
    out_dir  = Path(args.out_dir).resolve() if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = list(range(args.step, args.max_samples + 1, args.step))
    print(f"Checkpoints : {checkpoints}")

    # Collect candidate idx dirs
    idx_dirs = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit() and (d / "gp.pt").exists()]
    )
    if not idx_dirs:
        sys.exit(f"No per-index dirs with gp.pt found under {run_dir}")

    rng = random.Random(args.seed)
    chosen = rng.sample(idx_dirs, min(args.n_sims, len(idx_dirs)))
    chosen_indices = [int(d.name) for d in chosen]
    print(f"Selected simulations: {chosen_indices}")

    results = []
    for i, idx_dir in enumerate(chosen):
        idx = int(idx_dir.name)
        print(f"\n[{i+1}/{len(chosen)}] sim {idx} …")
        res = run_one_sim(
            idx_dir, data_dir, fallback,
            args.chains, args.warmup, args.max_samples,
            args.seed, checkpoints,
        )
        if res is None:
            print(f"  Skipped.")
            continue
        results.append(res)
        print(f"  mean R-hat @ {checkpoints[-1]} samples: {res['mean_rhat'][-1]:.4f}")

    if not results:
        sys.exit("All simulations failed — nothing to plot.")

    # Save data
    data_out = {
        "checkpoints": checkpoints,
        "n_sims": len(results),
        "selected_indices": chosen_indices[:len(results)],
        "simulations": results,
        "summary": {
            "mean_rhat": list(np.nanmean([r["mean_rhat"] for r in results], axis=0)),
            "mean_rhat_std": list(np.nanstd( [r["mean_rhat"] for r in results], axis=0)),
            "max_rhat":  list(np.nanmean([r["max_rhat"]  for r in results], axis=0)),
            "max_rhat_std":  list(np.nanstd( [r["max_rhat"]  for r in results], axis=0)),
        },
    }
    json_path = out_dir / "convergence_curve.json"
    with open(json_path, "w") as f:
        json.dump(data_out, f, indent=2)
    print(f"\nData saved → {json_path}")

    # Print table
    print("\n" + "=" * 60)
    print(f"{'Samples':>10}  {'Mean R-hat':>12}  {'±std':>8}  {'Max R-hat':>12}  {'±std':>8}")
    print("-" * 60)
    for T, m, ms, mx, mxs in zip(
        checkpoints,
        data_out["summary"]["mean_rhat"],
        data_out["summary"]["mean_rhat_std"],
        data_out["summary"]["max_rhat"],
        data_out["summary"]["max_rhat_std"],
    ):
        print(f"{T:>10}  {m:>12.4f}  {ms:>8.4f}  {mx:>12.4f}  {mxs:>8.4f}")
    print("=" * 60)

    plot_curve(checkpoints, results, out_dir / "convergence_curve.png")
    plot_per_sim(checkpoints, results, chosen_indices[:len(results)],
                 out_dir / "convergence_curve_per_sim.png")


if __name__ == "__main__":
    main()
