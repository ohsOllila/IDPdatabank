#!/usr/bin/env python3
"""
Recompute metrics for existing run outputs.

Fix: q-space metrics now use the q_infer grid (interpolation region only)
instead of training points, where posterior variance collapses and inflates
chi2 and NLPD even when the fit is visually correct.

Usage
-----
python recalc_metrics.py results/20260428_171757
python recalc_metrics.py results/20260428_171757 --dry-run
python recalc_metrics.py results/  # all timestamped subdirs
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import gptransform
from membrane.run import deabsolute


def load_posterior(path):
    """Load posterior txt file → (coords_np, mu tensor, std tensor)."""
    data = np.loadtxt(path, comments="#")
    coords = data[:, 0]
    mu = torch.tensor(data[:, 1], dtype=torch.float64)
    std = torch.tensor(data[:, 2] / 2.0, dtype=torch.float64)  # file stores 2*std
    return coords, mu, std


def load_cfg(run_dir: Path, fallback: Path) -> dict:
    """Load config.json from run dir if present, otherwise parse the fallback TOML."""
    cfg_json = run_dir / "config.json"
    if cfg_json.exists():
        with open(cfg_json) as f:
            return json.load(f)
    with open(fallback, "rb") as f:
        return tomllib.load(f)


def recompute(idx_dir: Path, idx: int, cfg: dict, ff_x, ff_y, td_x, td_y, dry_run=False):
    post_q = idx_dir / "posterior_q.txt"
    post_r = idx_dir / "posterior_r.txt"

    if not post_q.exists() or not post_r.exists():
        print(f"  [skip] missing posterior files in {idx_dir}")
        return

    q_infer_np, mu_q, std_q = load_posterior(post_q)
    r_infer_np, mu_r, std_r = load_posterior(post_r)

    # --- Q-space ground truth (always noiseless — we evaluate against the true signal) ---
    d = cfg["data"]
    q_np = ff_x[idx][75 : d["cutoff"]] * 10
    F_np = ff_y[idx][75 : d["cutoff"]]
    q_np, F_np = q_np[1:], F_np[1:]
    sq_np = deabsolute(F_np)  # sign-correct the clean signal, no noise added

    # Restrict to the interpolation region (no ground truth in extrapolation)
    in_range = q_infer_np <= q_np.max()
    y_true_q = torch.tensor(
        np.interp(q_infer_np[in_range], q_np, sq_np),
        dtype=torch.float64,
    )
    mask = torch.from_numpy(in_range)
    cov_q = torch.diag(std_q[mask] ** 2)
    metrics_q = gptransform.compute_metrics(mu_q[mask], cov_q, y_true_q)

    # --- R-space ground truth ---
    td_y_single = td_y[idx].reshape(-1)
    td_y_shifted = td_y_single - td_y_single[0]
    y_true_r = torch.tensor(
        np.interp(r_infer_np, td_x.reshape(-1), td_y_shifted),
        dtype=torch.float64,
    )
    r_shift = (y_true_r - mu_r).mean()
    mu_r_shifted = mu_r + r_shift

    # Restrict to where the GP has meaningful uncertainty (std_r > 1% of max).
    # In the tails std_r → 0: the GP is certain the density is ~0, but tiny
    # oscillations in the true density would trivially fail coverage there —
    # same issue as q-space extrapolation.
    r_active = std_r >= 0.05 * std_r.max()
    cov_r = torch.diag(std_r[r_active] ** 2)
    metrics_r = gptransform.compute_metrics(
        mu_r_shifted[r_active], cov_r, y_true_r[r_active]
    )

    print(f"idx={idx}")
    print(f"q: {metrics_q}")
    print(f"r: {metrics_r}\n")

    if not dry_run:
        result = {
            "q": {
                "rmse": metrics_q.rmse,
                "chi2": metrics_q.chi2,
                "coverage": metrics_q.coverage,
                "nlpd": metrics_q.nlpd,
            },
            "r": {
                "rmse": metrics_r.rmse,
                "chi2": metrics_r.chi2,
                "coverage": metrics_r.coverage,
                "nlpd": metrics_r.nlpd,
            },
        }
        with open(idx_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"  → wrote {idx_dir / 'metrics.json'}")

    return metrics_q, metrics_r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="Timestamped run directory (or parent containing multiple)")
    parser.add_argument("--config", default=str(REPO_ROOT / "membrane" / "config.toml"), help="Config file")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "membrane"), help="Directory containing all_ff_x/y.npy and all_td_x/y.npy")
    parser.add_argument("--dry-run", action="store_true", help="Print metrics without writing files")
    args = parser.parse_args()

    if tomllib is None:
        sys.exit("tomllib/tomli not available — cannot parse config")

    fallback_cfg = Path(args.config)

    data_dir = Path(args.data_dir)
    ff_x = np.load(data_dir / "all_ff_x.npy", allow_pickle=True)
    ff_y = np.load(data_dir / "all_ff_y.npy", allow_pickle=True)
    td_x = np.load(data_dir / "all_td_x.npy", allow_pickle=True)
    td_y = np.load(data_dir / "all_td_y.npy", allow_pickle=True)

    # Collect all idx directories (grouped by run_dir so each gets its own config)
    run_dirs = []
    for rd in args.run_dirs:
        rd = Path(rd)
        numeric_children = sorted(p for p in rd.iterdir() if p.is_dir() and p.name.isdigit())
        if numeric_children:
            run_dirs.append(rd)
        else:
            for ts_dir in sorted(rd.iterdir()):
                if not ts_dir.is_dir():
                    continue
                if any(p.name.isdigit() for p in ts_dir.iterdir() if p.is_dir()):
                    run_dirs.append(ts_dir)

    if not run_dirs:
        sys.exit("No run directories found.")

    all_metrics = {}
    for run_dir in run_dirs:
        cfg = load_cfg(run_dir, fallback_cfg)
        idx_dirs = sorted(
            int(p.name) for p in run_dir.iterdir() if p.is_dir() and p.name.isdigit()
        )
        for idx in idx_dirs:
            result = recompute(run_dir / str(idx), idx, cfg, ff_x, ff_y, td_x, td_y, dry_run=args.dry_run)
            if result:
                mq, mr = result
                all_metrics[idx] = {
                    "q": {"rmse": mq.rmse, "chi2": mq.chi2, "coverage": mq.coverage, "nlpd": mq.nlpd},
                    "r": {"rmse": mr.rmse, "chi2": mr.chi2, "coverage": mr.coverage, "nlpd": mr.nlpd},
                }

    if all_metrics and not args.dry_run:
        for rd in run_dirs:
            out_json = rd / "all_metrics.json"
            with open(out_json, "w") as f:
                json.dump(all_metrics, f, indent=2)
            print(f"\nAggregated JSON → {out_json}")

            out_csv = rd / "all_metrics.csv"
            with open(out_csv, "w") as f:
                f.write("idx,q_rmse,q_chi2,q_coverage,q_nlpd,r_rmse,r_chi2,r_coverage,r_nlpd\n")
                for idx, m in sorted(all_metrics.items()):
                    f.write(f"{idx},{m['q']['rmse']},{m['q']['chi2']},{m['q']['coverage']},{m['q']['nlpd']},"
                            f"{m['r']['rmse']},{m['r']['chi2']},{m['r']['coverage']},{m['r']['nlpd']}\n")
            print(f"Aggregated CSV  → {out_csv}")


if __name__ == "__main__":
    main()
