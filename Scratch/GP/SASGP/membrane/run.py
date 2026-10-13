#!/usr/bin/env python3
"""
membrane/run.py — Fit membrane GP to form factor data.

Examples
--------
python run.py --index 238
python run.py --index 238 403 --inference nuts
python run.py --all --config config.toml --out-dir results
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import savgol_filter, argrelextrema

sys.path.insert(0, str(Path(__file__).parent.parent))
import gptransform

torch.set_default_dtype(torch.float64)


# ── Preprocessing ────────────────────────────────────────────────────────────

def add_noise(F_abs_np, max_magnitude=10.0, steepness=10.0):
    n = len(F_abs_np)
    frac = np.linspace(0.0, 1.0, n)
    scale = max_magnitude / (1.0 + np.exp(-steepness * (frac - 0.5)))
    return F_abs_np + np.random.normal(0.0, scale, n)


def deabsolute(F_abs_np):
    """Sign-correct |F(q)| via smooth backbone valley detection."""
    backbone = savgol_filter(F_abs_np, window_length=50, polyorder=3)
    minima = argrelextrema(backbone, np.less)[0]
    threshold = np.max(backbone) * 0.08
    low = [i for i in minima if backbone[i] < threshold]

    #head = F_abs_np[:100]
    #head_minima = argrelextrema(head, np.less)[0]
    #head_threshold = np.max(head) * 0.06
    #head_low = [i for i in head_minima if head[i] < head_threshold]
    #print(f"Low head {head_low}")
    #low = sorted(set(low) | set(head_low))

    radius, verified = 50, []
    for i in low:
        s, e = max(0, i - radius), min(len(backbone), i + radius)
        ref = F_abs_np[s:e] if i < 100 else backbone[s:e]
        val = F_abs_np[i] if i < 100 else backbone[i]
        if val == ref.min():
            verified.append(i)

    flips, min_dist = [], 40
    if verified:
        group = [verified[0]]
        for i in range(1, len(verified)):
            if verified[i] - verified[i - 1] < min_dist:
                group.append(verified[i])
            else:
                flips.append(group[np.argmin(backbone[group])])
                group = [verified[i]]
        flips.append(group[np.argmin(backbone[group])])

    out, sign = F_abs_np.copy(), 1
    for fp in flips:
        sign *= -1
        out[fp:] = F_abs_np[fp:] * sign
    peak_idx = np.argmax(np.abs(F_abs_np))
    if out[peak_idx] > 0:
        out *= -1
    return out


# ── Config ───────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "data": {
        "ff_x": "all_ff_x.npy",
        "ff_y": "all_ff_y.npy",
        "td_x": "all_td_x.npy",
        "td_y": "all_td_y.npy",
        "cutoff": 800,
        "start_index": 76,
        "q_scale": 10.0,
        "recover_sign": True,
        "r_min": -5.0,
        "r_max": 5.0,
        "slicing": 8,
        "add_noise": True,
        "max_noise_magnitude": 10.0,
    },
    "run": {
        "noise_model": "constant",
        "inference": "laplace",
        "steps": 2000,
        "lr": 0.01,
        "mean_fn": "zero",
        "prior_fn": "uniform",
    },
    "inference": {
        "laplace_samples": 100,
        "nuts_samples": 200,
        "nuts_warmup": 100,
        "nuts_chains": 1,
        "seed": 0,
    },
    "bounds": {
        "ell": [0.1, 2.5],
        "c": [50.0, 200.0],
        "k": [0.1, 15.0],
        "a": [0.5, 5.0],
        "sigma_n_base": [0.01, 20.0],
        "sigma_n_slope": [0.0001, 2.0],
        "q_base": [-50.0, 20.0],
    },
    "output": {
        "out_dir": "results",
        "save_cov": False,
    },
}


def deep_merge(base, override):
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_toml(path):
    if tomllib is None:
        raise RuntimeError("TOML requires Python 3.11+ or: pip install tomli")
    with open(path, "rb") as f:
        return tomllib.load(f)


# ── GP setup ─────────────────────────────────────────────────────────────────

def build_bounds(cfg_bounds, noise_model, mean_fn="zero"):
    mean_keys = {"cosine": ["mean_A", "mean_B"], "tophat": ["mean_A", "mean_B", "mean_C", "mean_D"]}
    extra = mean_keys.get(mean_fn, [])
    n = (7 if noise_model == "quadratic" else 6) + len(extra)
    bounds = torch.zeros(n, 2)
    for i, key in enumerate(["ell", "c", "k", "a"]):
        bounds[i, 0], bounds[i, 1] = cfg_bounds[key]
    if noise_model == "quadratic":
        bounds[4, 0], bounds[4, 1] = cfg_bounds["sigma_n_base"]
        bounds[5, 0], bounds[5, 1] = cfg_bounds["sigma_n_slope"]
    else:
        bounds[4, 0], bounds[4, 1] = cfg_bounds["sigma_n_base"]
    q_base_idx = 6 if noise_model == "quadratic" else 5
    bounds[q_base_idx, 0], bounds[q_base_idx, 1] = cfg_bounds["q_base"]
    for j, key in enumerate(extra):
        bounds[q_base_idx + 1 + j, 0], bounds[q_base_idx + 1 + j, 1] = cfg_bounds[key]
    return bounds


# ── Fitting ──────────────────────────────────────────────────────────────────

def fit_single(idx, ff_x, ff_y, td_x, td_y, cfg):
    print(f"\n{'=' * 60}")
    print(f"  Index {idx}")
    print(f"{'=' * 60}")

    out_path = Path(cfg["output"]["out_dir"]) / str(idx)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"  Output → {out_path}")

    d = cfg["data"]
    r = cfg["run"]
    noise_model = r["noise_model"]
    inference   = r["inference"]

    # Validate configurable membrane preprocessing before fitting.
    if d["start_index"] < 0 or d["cutoff"] <= d["start_index"]:
        raise ValueError("Require 0 <= start_index < cutoff")
    if not np.isfinite(d["q_scale"]) or d["q_scale"] <= 0:
        raise ValueError("q_scale must be finite and positive")
    if d["slicing"] < 1 or not np.isfinite([d["r_min"], d["r_max"]]).all() or d["r_min"] >= d["r_max"]:
        raise ValueError("Require positive slicing and finite r_min < r_max")

    # Preprocess
    q_orig_np = ff_x[idx][d["start_index"]:] * d["q_scale"]
    F_orig_np = ff_y[idx][d["start_index"]:]
    q_orig = torch.tensor(q_orig_np, dtype=torch.float64).unsqueeze(1)
    sq_orig = torch.tensor(F_orig_np, dtype=torch.float64).unsqueeze(1)

    q_np  = ff_x[idx][d["start_index"]:d["cutoff"]] * d["q_scale"]
    F_np  = ff_y[idx][d["start_index"]:d["cutoff"]]

    if len(q_np[::d["slicing"]]) < 2:
        raise ValueError("Preprocessing must retain at least two training points")
    if d["recover_sign"] and len(F_np) < 50:
        raise ValueError("Sign recovery requires at least 50 retained points")

    if d["add_noise"]:
        F_np = add_noise(F_np, d["max_noise_magnitude"])
    F_signed = deabsolute(F_np) if d["recover_sign"] else F_np.copy()

    q_train  = torch.tensor(q_np,    dtype=torch.float64).unsqueeze(1)
    sq_train = torch.tensor(F_signed, dtype=torch.float64).unsqueeze(1)

    q_vals = q_train[::d["slicing"]]
    sq_vals = sq_train[::d["slicing"]]

    r_grid = torch.linspace(d["r_min"], d["r_max"], len(q_vals), dtype=torch.float64).unsqueeze(1)

    perm = torch.randperm(len(sq_vals)).unsqueeze(0)
    dataset = gptransform.data(
        q_vals[perm].reshape(1, len(q_vals), 1),
        sq_vals[perm].reshape(1, len(q_vals), 1),
    )

    # Build and train GP
    mean_fn     = r.get("mean_fn", "zero")
    bounds      = build_bounds(cfg["bounds"], noise_model, mean_fn)
    init_params = bounds.sum(dim=1) / 2
    gp = gptransform.GP(
        init_params, bounds, 0, 1.0, 1,
        noise_model=noise_model,
        prior_fn=r["prior_fn"],
        mean_fn=mean_fn,
    )
    optimizer = torch.optim.AdamW(gp.parameters(), lr=r["lr"])
    ylo = float(sq_vals.min()) - 50
    yhi = float(sq_vals.max()) + 50
    losses = gptransform.train_loop(
        dataset, gp, optimizer, r["steps"],
        r_grid, q_vals, sq_vals, q_vals, r_grid,
        ylo, yhi, plot=False,
    )
    torch.save(gp, out_path / "gp.pt")
    torch.save(torch.tensor(losses), out_path / "losses.pt")

    # Inference grids
    q_max       = float(q_train.max())
    q_infer     = torch.linspace(float(q_train.min()), q_max * 1.5, 300,
                                 dtype=torch.float64).unsqueeze(1)
    r_infer     = torch.linspace(d["r_min"], d["r_max"], 1000, dtype=torch.float64).unsqueeze(1)

    inf = cfg["inference"]
    seed = inf.get("seed", 0)

    if inference == "laplace":
        mu_q, cov_q, mu_r, cov_r = gptransform.laplace_predict(
            gp, r_grid, q_vals, sq_vals,
            q_infer=q_infer, r_infer=r_infer,
            num_samples=inf.get("laplace_samples", 50),
            seed=seed,
        )
    elif inference == "nuts":
        mu_q, cov_q, mu_r, cov_r = gptransform.nuts_predict(
            gp, r_grid, q_vals, sq_vals,
            q_infer=q_infer, r_infer=r_infer,
            num_samples=inf.get("nuts_samples", 200),
            num_warmup=inf.get("nuts_warmup", 100),
            num_chains=inf.get("nuts_chains", 1),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown inference: {inference!r}")

    mu_q, cov_q = mu_q.detach(), cov_q.detach()
    mu_r, cov_r = mu_r.detach(), cov_r.detach()

    # Metrics — q-space on inference grid (interpolation region only).
    # Avoids collapsed posterior variance at training points which inflates chi2/NLPD.
    # Ground truth is noiseless sq_train interpolated onto q_infer.
    q_infer_np  = q_infer.reshape(-1).numpy()
    q_train_np  = q_train.reshape(-1).numpy()
    sq_train_np = sq_train.reshape(-1).numpy()
    in_range    = q_infer_np <= q_train_np.max()
    y_true_q    = torch.tensor(
        np.interp(q_infer_np[in_range], q_train_np, sq_train_np),
        dtype=torch.float64,
    )
    q_mask    = torch.from_numpy(in_range)
    metrics_q = gptransform.compute_metrics(
        mu_q[q_mask], cov_q[q_mask][:, q_mask], y_true_q
    )

    # Metrics — r-space (interpolate ground truth onto inference grid)
    td_y_single  = td_y[idx].reshape(-1)
    td_y_shifted = td_y_single - td_y_single[0]
    r_infer_np   = r_infer.reshape(-1).numpy()
    y_true_r     = torch.tensor(
        np.interp(r_infer_np, td_x.reshape(-1), td_y_shifted),
        dtype=torch.float64,
    )
    # Optimal vertical shift: minimises RMSE analytically (mean of residuals)
    r_shift = (y_true_r - mu_r.reshape(-1)).mean()
    mu_r = mu_r + r_shift
    print(f"  r-space vertical shift: {r_shift:.6f}")
    # Restrict to where the GP has meaningful uncertainty (std_r >= 5% of max).
    # In the tails std_r → 0, causing trivial coverage failures unrelated to fit quality.
    std_r    = torch.diag(cov_r).clamp(min=0).sqrt()
    r_active = std_r >= 0.05 * std_r.max()
    metrics_r = gptransform.compute_metrics(
        mu_r[r_active], cov_r[r_active][:, r_active], y_true_r[r_active]
    )

    print(f"\nMetrics q-space: {metrics_q}")
    print(f"Metrics r-space: {metrics_r}")

    # Save
    np.savetxt(out_path / "posterior_q.txt",
               np.column_stack([
                   q_infer.numpy(),
                   mu_q.numpy(),
                   2 * torch.diag(cov_q).clamp(min=0).sqrt().numpy(),
               ]),
               header="q  mu_q  2*std_q")

    np.savetxt(out_path / "posterior_r.txt",
               np.column_stack([
                   r_infer.numpy(),
                   mu_r.numpy(),
                   2 * torch.diag(cov_r).clamp(min=0).sqrt().numpy(),
               ]),
               header="r  mu_r  2*std_r")

    with open(out_path / "metrics.json", "w") as f:
        json.dump({
            "q": {"rmse": metrics_q.rmse, "chi2": metrics_q.chi2,
                  "coverage": metrics_q.coverage, "nlpd": metrics_q.nlpd},
            "r": {"rmse": metrics_r.rmse, "chi2": metrics_r.chi2,
                  "coverage": metrics_r.coverage, "nlpd": metrics_r.nlpd},
        }, f, indent=2)

    if cfg["output"].get("save_cov", False):
        np.savetxt(out_path / "cov_q.txt", cov_q.numpy())
        np.savetxt(out_path / "cov_r.txt", cov_r.numpy())

    _plot_summary(
        losses=losses,
        q_train=q_vals, sq_train=sq_vals,
        q_full=q_orig, sq_full=sq_orig,
        q_infer=q_infer, r_infer=r_infer,
        mu_q=mu_q, cov_q=cov_q,
        mu_r=mu_r, cov_r=cov_r,
        td_x=td_x.reshape(-1),
        td_y=td_y[idx].reshape(-1),
        label=inference.capitalize(),
        out_file=out_path / "_plots",
        gp=gp,
    )
    return metrics_q, metrics_r


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_summary(losses, q_train, sq_train, q_infer, r_infer,
                  mu_q, cov_q, mu_r, cov_r, td_x, td_y, label, out_file, gp=None,
                  q_full=None, sq_full=None):
    std_q   = torch.diag(cov_q).clamp(min=0).sqrt().numpy()
    std_r   = torch.diag(cov_r).clamp(min=0).sqrt().numpy()
    mu_q_np = mu_q.reshape(-1).numpy()
    mu_r_np = mu_r.reshape(-1).numpy()
    q_np    = q_infer.reshape(-1).numpy()
    r_np    = r_infer.reshape(-1).numpy()
    q_dat   = q_train.reshape(-1).numpy()
    sq_dat  = sq_train.reshape(-1).numpy()
    out_dir = Path(out_file).parent

    # ── Loss curve ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, lw=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("−LMLH")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir / "loss.png")

    # ── q-space posterior ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    q_max_data = float(q_train.max())
    ax.axvspan(q_max_data, float(q_infer.max()), color="grey", alpha=0.08,
               label="Extrapolation")
    if q_full is not None and sq_full is not None:
        ax.plot(q_full.reshape(-1).numpy(), sq_full.reshape(-1).numpy(),
                color="orange", linestyle="--", lw=1.5, alpha=0.8, label="True |F(q)|", zorder=2)
    ax.scatter(q_dat, sq_dat, s=6, alpha=0.3, c="k", label="Data", zorder=3)
    ax.plot(q_np, mu_q_np, label=f"{label} posterior", lw=1.5)
    ax.fill_between(q_np, mu_q_np - 2 * std_q, mu_q_np + 2 * std_q,
                    alpha=0.18, label="±2σ")
    ax.fill_between(q_np, mu_q_np - 1 * std_q, mu_q_np + 1 * std_q,
                    alpha=0.35, label="±1σ")
    ax.set_xlabel("q [Å⁻¹]")
    ax.set_ylabel("F(q)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir / "posterior_q.png")

    # ── r-space posterior ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    td_y_shifted = td_y - td_y[0]
    ax.plot(r_np, mu_r_np, label=f"{label} posterior", lw=1.5)
    ax.fill_between(r_np, mu_r_np - 2 * std_r, mu_r_np + 2 * std_r,
                    alpha=0.18, label="±2σ")
    ax.fill_between(r_np, mu_r_np - 1 * std_r, mu_r_np + 1 * std_r,
                    alpha=0.35, label="±1σ")
    ax.plot(td_x, td_y_shifted, label="True ρ(r)", alpha=0.8,
            linestyle="--", lw=1.5)
    ax.set_xlabel("r [Å]")
    ax.set_ylabel("ρ(r)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir / "posterior_r.png")

    # ── q-space uncertainty breakdown ────────────────────────────
    if gp is not None:
        q_ext = torch.linspace(float(q_train.min()), 10.0, 300,
                               dtype=torch.float64).unsqueeze(1)
        with torch.no_grad():
            r_grid_unc = torch.linspace(float(r_infer.min()), float(r_infer.max()), len(q_train), dtype=torch.float64).unsqueeze(1)
            mu_q_tr, cov_q_tr = gp.predict_sq_trapz(
                r_grid_unc, q_train, q_train, sq_train, adjust=False,
            )
            residuals = sq_train.reshape(-1) - mu_q_tr.reshape(-1)
            mu_q_ext, cov_q_ext = gp.predict_sq_trapz(
                r_grid_unc, q_ext, q_train, sq_train, adjust=False,
            )
            noise_vec_ext = gp.noise_q(q_ext)
            std_gp    = torch.diag(cov_q_ext).clamp(min=0).sqrt()
            std_total = (torch.diag(cov_q_ext) + noise_vec_ext).clamp(min=0).sqrt()

        q_ext_np   = q_ext.reshape(-1).numpy()
        q_max_data = float(q_train.max())

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.axvspan(q_max_data, 10.0, color="grey", alpha=0.08, label="Extrapolation (no data)")
        ax.axhline(0, color="k", lw=0.8, alpha=0.4)
        ax.fill_between(q_ext_np,
                        -2 * std_gp.numpy(), 2 * std_gp.numpy(),
                        alpha=0.18, color="steelblue", label="Noiseless GP ±2σ")
        ax.fill_between(q_ext_np,
                        -1 * std_gp.numpy(), 1 * std_gp.numpy(),
                        alpha=0.35, color="steelblue", label="Noiseless GP ±1σ")
        ax.plot(q_ext_np, 2 * std_total.numpy(),
                color="purple", linestyle="--", lw=1.2, label="Total (GP+noise) ±2σ")
        ax.plot(q_ext_np, 1 * std_total.numpy(),
                color="purple", linestyle=":", lw=1.2, label="Total (GP+noise) ±1σ")
        ax.plot(q_ext_np, -2 * std_total.numpy(),
                color="purple", linestyle="--", lw=1.2)
        ax.plot(q_ext_np, -1 * std_total.numpy(),
                color="purple", linestyle=":", lw=1.2)
        ax.scatter(q_dat, residuals.numpy(),
                   s=8, alpha=0.4, c="k", label="Residuals (data − pred)", zorder=3)
        ax.set_xlim(float(q_train.min()), 10.0)
        ax.set_xlabel("q [Å⁻¹]")
        ax.set_ylabel("ΔF(q)")
        ax.set_title("q-space uncertainty & residuals")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(fig, out_dir / "uncertainty_q.png")

    # ── r-space uncertainty breakdown ─────────────────────────────
    # Bands centred at 0 (posterior mean subtracted); truth plotted as residual
    if gp is not None:
        with torch.no_grad():
            r_grid_gp = torch.linspace(float(r_infer.min()), float(r_infer.max()), len(q_train), dtype=torch.float64).unsqueeze(1)
            r_infer_t = torch.tensor(r_np, dtype=torch.float64).unsqueeze(1)
            mu_r_map, cov_r_map = gp.predict_ed_trapz(
                r_grid_gp, r_infer_t, q_train, sq_train, adjust=False,
            )
            std_r_map = torch.diag(cov_r_map).clamp(min=0).sqrt().numpy()

        # Interpolate truth onto r_infer grid so we can plot truth − posterior_mean
        truth_on_grid = np.interp(r_np, td_x, td_y_shifted)
        truth_residual = truth_on_grid - mu_r_np   # how far truth is from posterior mean

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.axhline(0, color="k", lw=0.8, alpha=0.4, label="Posterior mean")
        # MAP std (lighter, behind)
        ax.fill_between(r_np, -2 * std_r_map, 2 * std_r_map,
                        alpha=0.10, color="grey", label="MAP ±2σ")
        ax.fill_between(r_np, -1 * std_r_map, 1 * std_r_map,
                        alpha=0.20, color="grey", label="MAP ±1σ")
        # Full inference std
        ax.fill_between(r_np, -2 * std_r, 2 * std_r,
                        alpha=0.18, color="steelblue", label=f"{label} ±2σ")
        ax.fill_between(r_np, -1 * std_r, 1 * std_r,
                        alpha=0.35, color="steelblue", label=f"{label} ±1σ")
        # Truth relative to posterior mean
        ax.plot(r_np, truth_residual, color="darkorange", lw=1.5,
                linestyle="--", alpha=0.85, label="True ρ(r) − posterior mean")
        ax.set_xlabel("r [Å]")
        ax.set_ylabel("Δρ(r)")
        ax.set_title("r-space uncertainty centred at posterior mean")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(fig, out_dir / "uncertainty_r.png")

    # ── Kernel diagonals ─────────────────────────────────────────
    if gp is not None:
        with torch.no_grad():
            r_plot    = torch.linspace(float(r_infer.min()), float(r_infer.max()), 400, dtype=torch.float64).unsqueeze(1)
            q_plot    = torch.linspace(float(q_infer.min()), float(q_infer.max()), 200,
                                       dtype=torch.float64).unsqueeze(1)
            r_grid_gp = torch.linspace(float(r_infer.min()), float(r_infer.max()), len(q_train), dtype=torch.float64).unsqueeze(1)
            k_rr_diag = torch.diag(gp.K_rr(r_plot, r_plot, adjust=False)).clamp(min=0).sqrt()
            k_qq_diag = torch.diag(gp.K_qq(r_grid_gp, r_grid_gp, q_plot, q_plot,
                                            adjust=False)).clamp(min=0).sqrt()
            noise_q   = gp.noise_q(q_plot).sqrt()

        fig, ax = plt.subplots(figsize=(9, 4))
        ax2 = ax.twinx()
        ax.plot(r_plot.numpy(), k_rr_diag.numpy(), label="K_rr std", color="steelblue")
        ax2.plot(q_plot.numpy(), k_qq_diag.numpy(), label="K_qq std",
                 color="darkorange", linestyle="--")
        ax2.plot(q_plot.numpy(), noise_q.numpy(), label="noise std",
                 color="red", linestyle=":", lw=1)
        ax.set_xlabel("r [Å] / q [Å⁻¹]")
        ax.set_ylabel("K_rr std", color="steelblue")
        ax2.set_ylabel("K_qq / noise std", color="darkorange")
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
        ax.set_title("Learned kernel std diagonals")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(fig, out_dir / "kernel.png")

    # ── Posterior covariance 2D heatmaps ─────────────────────────
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))

    pcm0 = axs[0].pcolormesh(r_np, r_np, cov_r.numpy(), shading="auto", cmap="magma")
    fig.colorbar(pcm0, ax=axs[0], fraction=0.046, pad=0.04)
    axs[0].set_title("$\\Sigma_{r,\\mathrm{post}}$")
    axs[0].set_xlabel("$r$ [Å]")
    axs[0].set_ylabel("$r$ [Å]")
    axs[0].set_aspect("equal")

    pcm1 = axs[1].pcolormesh(q_np, q_np, cov_q.numpy(), shading="auto", cmap="magma")
    fig.colorbar(pcm1, ax=axs[1], fraction=0.046, pad=0.04)
    axs[1].set_title("$\\Sigma_{q,\\mathrm{post}}$")
    axs[1].set_xlabel("$q$ [Å⁻¹]")
    axs[1].set_ylabel("$q$ [Å⁻¹]")
    axs[1].set_aspect("equal")

    plt.tight_layout()
    _save(fig, out_dir / "kernels_2d.png")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Membrane GP fitting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", metavar="FILE", help="TOML config file")
    p.add_argument("--index", type=int, nargs="+", metavar="N", help="Dataset index/indices")
    p.add_argument("--all", action="store_true", help="Fit all indices in the dataset")
    p.add_argument("--inference", choices=["laplace", "nuts"], help="Inference method")
    p.add_argument("--noise-model", choices=["constant", "quadratic"])
    p.add_argument("--steps", type=int, help="Training epochs")
    p.add_argument("--mean-fn", choices=["zero"], help="GP prior mean function")
    p.add_argument("--prior-fn", choices=["uniform", "gaussian"], help="Hyperparameter prior")
    p.add_argument("--out-dir", metavar="DIR", help="Output directory")
    p.add_argument("--cutoff", type=int, help="Data cutoff index")
    p.add_argument("--max-noise", type=float, help="Max noise magnitude")
    p.add_argument("--sample", type=int, nargs="?", const=100, metavar="N",
                   help="Fit a fixed random sample of N indices (default 100)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = DEFAULT_CONFIG

    if args.config:
        cfg = deep_merge(cfg, load_toml(args.config))

    # CLI overrides (highest priority)
    if args.inference:
        cfg["run"]["inference"] = args.inference
    if args.noise_model:
        cfg["run"]["noise_model"] = args.noise_model
    if args.steps is not None:
        cfg["run"]["steps"] = args.steps
    if args.mean_fn:
        cfg["run"]["mean_fn"] = args.mean_fn
    if args.prior_fn:
        cfg["run"]["prior_fn"] = args.prior_fn
    if args.out_dir:
        cfg["output"]["out_dir"] = args.out_dir
    if args.cutoff is not None:
        cfg["data"]["cutoff"] = args.cutoff
    if args.max_noise is not None:
        cfg["data"]["max_noise_magnitude"] = args.max_noise

    # Stamp the run dir with date-time so runs don't overwrite each other
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg["output"]["out_dir"] = str(Path(cfg["output"]["out_dir"]) / timestamp)
    print(f"Run dir: {cfg['output']['out_dir']}")

    run_dir = Path(cfg["output"]["out_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Config   → {run_dir / 'config.json'}")

    # Load data (relative to this script's directory)
    here = Path(__file__).parent
    d    = cfg["data"]
    ff_x = np.load(here / d["ff_x"])
    ff_y = np.load(here / d["ff_y"])
    td_x = np.load(here / d["td_x"])
    td_y = np.load(here / d["td_y"])
    n_total = len(ff_x)

    # Determine indices
    if args.all or cfg["run"].get("all", False):
        indices = list(range(n_total))
    elif args.sample is not None:
        n_sample = args.sample
        rng = np.random.default_rng(42)
        indices = sorted(rng.choice(n_total, size=min(n_sample, n_total), replace=False).tolist())
    elif args.index:
        indices = args.index
    elif "indices" in cfg["run"]:
        indices = cfg["run"]["indices"]
    else:
        print("Specify --index N [M ...], --all, or --sample [N]", file=sys.stderr)
        sys.exit(1)

    all_metrics = {}
    for idx in indices:
        if idx >= n_total:
            print(f"Warning: index {idx} >= n={n_total}, skipping")
            continue
        mq, mr = fit_single(idx, ff_x, ff_y, td_x, td_y, cfg)
        all_metrics[idx] = {
            "q": {"rmse": mq.rmse, "chi2": mq.chi2,
                  "coverage": mq.coverage, "nlpd": mq.nlpd},
            "r": {"rmse": mr.rmse, "chi2": mr.chi2,
                  "coverage": mr.coverage, "nlpd": mr.nlpd},
        }

    if len(all_metrics) >= 1:
        out_dir = Path(cfg["output"]["out_dir"])
        out_json = out_dir / "all_metrics.json"
        with open(out_json, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nAll metrics → {out_json}")

        out_csv = out_dir / "all_metrics.csv"
        with open(out_csv, "w") as f:
            f.write("index,q_rmse,q_chi2,q_coverage,q_nlpd,r_rmse,r_chi2,r_coverage,r_nlpd\n")
            for idx, m in all_metrics.items():
                f.write(f"{idx},"
                        f"{m['q']['rmse']:.6g},{m['q']['chi2']:.6g},"
                        f"{m['q']['coverage']:.6g},{m['q']['nlpd']:.6g},"
                        f"{m['r']['rmse']:.6g},{m['r']['chi2']:.6g},"
                        f"{m['r']['coverage']:.6g},{m['r']['nlpd']:.6g}\n")
        print(f"All metrics CSV → {out_csv}")


if __name__ == "__main__":
    main()
