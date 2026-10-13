#!/usr/bin/env python3
"""
NUTS convergence diagnostics for a saved GP run.

Usage
-----
python membrane/nuts_convergence.py <idx_dir> [options]

  <idx_dir>   path to a per-index run dir containing gp.pt
              e.g. membrane/results/sweep/.../20260508_120000/238

Options
-------
--chains N        number of independent chains (default 4)
--samples N       samples per chain after warmup (default 500)
--warmup N        NUTS warmup steps (default 200)
--seed N          RNG seed (default 0)
--data-dir PATH   directory with all_ff_x/y.npy and all_td_x/y.npy
                  (default: same directory as this script)
--config PATH     fallback config if config.json absent in run dir
--out-dir PATH    where to save plots (default: <idx_dir>)
--no-plots        print diagnostics only, skip figure generation
"""

import argparse
import json
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
from membrane.run import deabsolute, build_bounds

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


# ── param labels ──────────────────────────────────────────────────────────────

def param_labels(noise_model, mean_fn):
    labels = ["ell", "c", "k", "a", "noise_c"]
    if noise_model == "quadratic":
        labels.append("noise_q")
    labels.append("q_base")
    if mean_fn == "cosine":
        labels += ["mean_A", "mean_B"]
    elif mean_fn == "tophat":
        labels += ["mean_A", "mean_B", "mean_C", "mean_D"]
    return labels


# ── data reconstruction ────────────────────────────────────────────────────────

def load_cfg(idx_dir: Path, fallback: Path) -> dict:
    # config.json lives one level up (timestamped root)
    for candidate in [idx_dir / "config.json",
                      idx_dir.parent / "config.json"]:
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)
    if fallback is not None and fallback.exists():
        with open(fallback, "rb") as f:
            return tomllib.load(f)
    raise FileNotFoundError("No config.json or fallback TOML found")


def reconstruct_training_data(idx: int, cfg: dict, ff_x, ff_y):
    d   = cfg["data"]
    q_np = ff_x[idx][75:d["cutoff"]] * 10
    F_np = ff_y[idx][75:d["cutoff"]]
    q_np, F_np = q_np[1:], F_np[1:]
    F_signed = deabsolute(F_np)

    q_train  = torch.tensor(q_np,     dtype=torch.float64).unsqueeze(1)
    sq_train = torch.tensor(F_signed, dtype=torch.float64).unsqueeze(1)

    sl = d.get("slicing", 1)
    q_vals  = q_train[::sl]
    sq_vals = sq_train[::sl]
    r_grid  = torch.linspace(-5, 5, len(q_vals), dtype=torch.float64).unsqueeze(1)
    return q_vals, sq_vals, r_grid


# ── convergence stats ──────────────────────────────────────────────────────────

def split_rhat(chains):
    """
    Gelman-Rubin split R-hat.
    chains: (C, S, P) ndarray
    Returns R-hat per parameter, shape (P,).
    """
    C, S, P = chains.shape
    half = S // 2
    # split each chain → 2*C chains of length half
    splits = np.concatenate([chains[:, :half, :], chains[:, half:half*2, :]], axis=0)
    M = splits.shape[0]  # 2*C
    N = half

    chain_means = splits.mean(axis=1)          # (M, P)
    chain_vars  = splits.var(axis=1, ddof=1)   # (M, P)
    grand_mean  = chain_means.mean(axis=0)     # (P,)

    B = N * ((chain_means - grand_mean) ** 2).sum(axis=0) / (M - 1)
    W = chain_vars.mean(axis=0)

    var_hat = (N - 1) / N * W + B / N
    rhat = np.sqrt(var_hat / (W + 1e-30))
    return rhat


def bulk_ess(chains):
    """
    Bulk ESS via rank-normalisation.
    chains: (C, S, P)
    Returns ESS per parameter.
    """
    from scipy.stats import rankdata
    C, S, P = chains.shape
    all_draws = chains.reshape(C * S, P)

    ess_vals = np.empty(P)
    for p in range(P):
        ranks = rankdata(all_draws[:, p]).reshape(C, S)
        # normalise to z-scores
        z = (ranks - 0.5) / (C * S)
        z = np.clip(z, 1e-6, 1 - 1e-6)
        from scipy.special import ndtri
        z = ndtri(z)
        ess_vals[p] = _ess_from_chains(z)
    return ess_vals


def _ess_from_chains(z):
    """ESS from rank-normalised chains (C, S) array."""
    C, S = z.shape
    # variogram-based estimate
    rho = []
    var_z = z.var(ddof=1)
    if var_z < 1e-15:
        return float(C * S)
    for t in range(1, S):
        v = ((z[:, t:] - z[:, :-t]) ** 2).mean()
        rho.append(1.0 - v / (2 * var_z))
    # sum positive pairs
    n_terms = 0
    rho_sum = 0.0
    for i in range(0, len(rho) - 1, 2):
        pair = rho[i] + rho[i + 1]
        if pair <= 0:
            break
        rho_sum += pair
        n_terms += 2
    ess = C * S / (1 + 2 * rho_sum) if rho_sum > 0 else float(C * S)
    return max(ess, 1.0)


def autocorr(x, max_lag=50):
    """Autocorrelation of 1-D array x up to max_lag."""
    x = x - x.mean()
    n = len(x)
    result = np.correlate(x, x, mode="full")
    result = result[n - 1:]
    result /= result[0] + 1e-30
    return result[:max_lag + 1]


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_traces(chain_samples, labels, out_path):
    C, S, P = chain_samples.shape
    colors = plt.cm.tab10(np.linspace(0, 1, C))
    ncols = 2
    nrows = (P + 1) // 2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(12, 2.5 * nrows),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for p in range(P):
        ax = axes[p // ncols][p % ncols]
        for c in range(C):
            ax.plot(chain_samples[c, :, p], color=colors[c], alpha=0.7,
                    linewidth=0.8, label=f"chain {c+1}" if p == 0 else None)
        ax.set_title(labels[p], fontsize=9)
        ax.set_xlabel("sample", fontsize=7)
        ax.tick_params(labelsize=7)
    # hide unused subplots
    for p in range(P, nrows * ncols):
        axes[p // ncols][p % ncols].set_visible(False)
    if C > 1:
        fig.legend(*axes[0][0].get_legend_handles_labels(),
                   loc="lower right", fontsize=8, ncol=C)
    fig.suptitle("Trace plots (theta_raw)", fontsize=11)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Trace plot  → {out_path}")


def plot_rhat_ess(rhat, ess, labels, out_path):
    P = len(labels)
    x = np.arange(P)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, P * 0.7), 6),
                                   constrained_layout=True)

    colors_rhat = ["#e15759" if r > 1.05 else
                   "#f28e2b" if r > 1.01 else "#4e9a3f"
                   for r in rhat]
    ax1.bar(x, rhat, color=colors_rhat)
    ax1.axhline(1.01, color="#f28e2b", linestyle="--", linewidth=1, label="1.01")
    ax1.axhline(1.05, color="#e15759", linestyle="--", linewidth=1, label="1.05")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("R-hat")
    ax1.set_title("Split R-hat per parameter")
    ax1.legend(fontsize=8)

    colors_ess = ["#e15759" if e < 100 else
                  "#f28e2b" if e < 400 else "#4e9a3f"
                  for e in ess]
    ax2.bar(x, ess, color=colors_ess)
    ax2.axhline(400, color="#f28e2b", linestyle="--", linewidth=1, label="400")
    ax2.axhline(100, color="#e15759", linestyle="--", linewidth=1, label="100")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Bulk ESS")
    ax2.set_title("Bulk ESS per parameter")
    ax2.legend(fontsize=8)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"R-hat/ESS   → {out_path}")


def plot_acf(chain_samples, labels, out_path, max_lag=50):
    C, S, P = chain_samples.shape
    ncols = 2
    nrows = (P + 1) // 2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(12, 2.5 * nrows),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    lags = np.arange(max_lag + 1)
    for p in range(P):
        ax = axes[p // ncols][p % ncols]
        for c in range(C):
            ac = autocorr(chain_samples[c, :, p], max_lag)
            ax.plot(lags, ac, alpha=0.7, linewidth=0.9)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axhline(0.05, color="grey", linestyle="--", linewidth=0.7)
        ax.set_title(labels[p], fontsize=9)
        ax.set_xlabel("lag", fontsize=7)
        ax.set_ylim(-0.2, 1.05)
        ax.tick_params(labelsize=7)
    for p in range(P, nrows * ncols):
        axes[p // ncols][p % ncols].set_visible(False)
    fig.suptitle("Autocorrelation (theta_raw)", fontsize=11)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"ACF plot    → {out_path}")


def plot_pairs(chain_samples, labels, out_path):
    """Corner-style pair plot for the first min(P, 6) parameters."""
    C, S, P = chain_samples.shape
    P_show = min(P, 6)
    flat = chain_samples.reshape(C * S, P)[:, :P_show]
    labs = labels[:P_show]

    fig, axes = plt.subplots(P_show, P_show,
                             figsize=(2.5 * P_show, 2.5 * P_show),
                             constrained_layout=True)
    for i in range(P_show):
        for j in range(P_show):
            ax = axes[i][j]
            if i == j:
                ax.hist(flat[:, i], bins=40, color="#4c78a8", alpha=0.8)
            elif j < i:
                ax.scatter(flat[:, j], flat[:, i],
                           s=1, alpha=0.3, color="#4c78a8", rasterized=True)
            else:
                ax.set_visible(False)
            if j == 0:
                ax.set_ylabel(labs[i], fontsize=14)
            if i == P_show - 1:
                ax.set_xlabel(labs[j], fontsize=14)
            ax.tick_params(labelsize=6)
    fig.suptitle("Pair plot (first 6 params)", fontsize=11)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Pair plot   → {out_path}")

    # Separate single-row figure: one histogram per parameter
    dist_path = out_path.parent / (out_path.stem + "_distributions.png")
    fig2, axes2 = plt.subplots(1, P_show,
                               figsize=(3.5 * P_show, 3.5),
                               constrained_layout=True)
    if P_show == 1:
        axes2 = [axes2]
    for i, ax in enumerate(axes2):
        ax.hist(flat[:, i], bins=40, color="#4c78a8", alpha=0.85, density=True)
        ax.set_title(labs[i], fontsize=14)
        ax.tick_params(labelsize=9)
        ax.set_ylabel("density" if i == 0 else "")
    fig2.suptitle("Marginal distributions", fontsize=13)
    fig2.savefig(dist_path, dpi=150)
    plt.close(fig2)
    print(f"Distributions → {dist_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def print_summary(rhat, ess, labels, n_diverge, C, S):
    print("\n" + "=" * 60)
    print(f"NUTS CONVERGENCE DIAGNOSTICS  ({C} chains × {S} samples)")
    print("=" * 60)
    print(f"{'Param':<14} {'R-hat':>8} {'Bulk ESS':>10}  Status")
    print("-" * 60)
    for label, r, e in zip(labels, rhat, ess):
        if r > 1.05:
            status = "FAIL (R-hat)"
        elif r > 1.01:
            status = "WARN (R-hat)"
        elif e < 100:
            status = "FAIL (ESS)"
        elif e < 400:
            status = "WARN (ESS)"
        else:
            status = "OK"
        print(f"{label:<14} {r:>8.4f} {e:>10.1f}  {status}")
    print("-" * 60)
    n_fail_rhat = sum(r > 1.05 for r in rhat)
    n_warn_rhat = sum(1.01 < r <= 1.05 for r in rhat)
    n_fail_ess  = sum(e < 100 for e in ess)
    print(f"R-hat > 1.05: {n_fail_rhat}  |  R-hat 1.01-1.05: {n_warn_rhat}")
    print(f"ESS < 100:    {n_fail_ess}")
    print(f"Divergences:  {n_diverge}")
    verdict = "CONVERGED" if (n_fail_rhat == 0 and n_fail_ess == 0 and n_diverge == 0) else "NOT CONVERGED"
    print(f"\nVerdict: {verdict}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("idx_dir", help="Per-index run directory containing gp.pt")
    parser.add_argument("--chains",   type=int, default=4)
    parser.add_argument("--samples",  type=int, default=500)
    parser.add_argument("--warmup",   type=int, default=200)
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--data-dir", default=None,
                        help="Directory with all_ff_x/y.npy (default: script dir)")
    parser.add_argument("--config",   default=None,
                        help="Fallback TOML config if config.json absent")
    parser.add_argument("--out-dir",  default=None,
                        help="Output directory for plots (default: idx_dir)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if tomllib is None:
        sys.exit("tomllib/tomli required — pip install tomli")

    idx_dir  = Path(args.idx_dir).resolve()
    idx      = int(idx_dir.name)
    out_dir  = Path(args.out_dir).resolve() if args.out_dir else idx_dir
    here     = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).resolve() if args.data_dir else here
    fallback = Path(args.config) if args.config else (here / "config.toml")

    print(f"Run dir : {idx_dir}")
    print(f"Index   : {idx}")

    # load GP
    gp_path = idx_dir / "gp.pt"
    if not gp_path.exists():
        sys.exit(f"gp.pt not found in {idx_dir}")
    gp = torch.load(gp_path, map_location="cpu", weights_only=False)
    gp.eval()

    noise_model = getattr(gp, "noise_model", "constant")
    mean_fn     = getattr(gp, "mean_fn",     "zero")
    labels      = param_labels(noise_model, mean_fn)
    print(f"Params  : {len(labels)}  ({noise_model} noise, {mean_fn} mean)")

    # reconstruct training data
    cfg     = load_cfg(idx_dir, fallback)
    ff_x    = np.load(data_dir / "all_ff_x.npy", allow_pickle=True)
    ff_y    = np.load(data_dir / "all_ff_y.npy", allow_pickle=True)
    q_vals, sq_vals, r_grid = reconstruct_training_data(idx, cfg, ff_x, ff_y)
    print(f"Training points: {len(q_vals)}")

    # run NUTS
    print(f"\nRunning NUTS: {args.chains} chains × {args.samples} samples "
          f"(+{args.warmup} warmup) …")
    mcmc, _ = gptransform.nuts_sample(
        gp, r_grid, q_vals, sq_vals,
        num_samples=args.samples,
        num_warmup=args.warmup,
        num_chains=args.chains,
        seed=args.seed,
    )

    # per-chain samples: (C, S, P)
    chain_samples = mcmc.get_samples(group_by_chain=True)["theta_raw"].cpu().numpy()
    C, S, P = chain_samples.shape

    # divergences
    try:
        diag = mcmc.diagnostics()
        n_diverge = int(diag.get("diverging", torch.tensor(0)).sum().item())
    except Exception:
        n_diverge = 0

    # convergence stats
    rhat = split_rhat(chain_samples)
    try:
        ess = bulk_ess(chain_samples)
    except ImportError:
        # scipy not available — fall back to Pyro's n_eff
        diag2 = mcmc.diagnostics()
        ess = diag2["theta_raw"]["n_eff"].cpu().numpy()

    print_summary(rhat, ess, labels, n_diverge, C, S)

    # save JSON summary
    summary = {
        "chains": C, "samples_per_chain": S, "divergences": n_diverge,
        "params": {
            lab: {"rhat": float(r), "ess": float(e)}
            for lab, r, e in zip(labels, rhat, ess)
        },
    }
    out_json = out_dir / "convergence.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON summary → {out_json}")

    if args.no_plots:
        return

    print("\nGenerating plots …")
    plot_traces(chain_samples, labels, out_dir / "conv_traces.png")
    plot_rhat_ess(rhat, ess, labels, out_dir / "conv_rhat_ess.png")
    plot_acf(chain_samples, labels, out_dir / "conv_acf.png")
    plot_pairs(chain_samples, labels, out_dir / "conv_pairs.png")


if __name__ == "__main__":
    main()
