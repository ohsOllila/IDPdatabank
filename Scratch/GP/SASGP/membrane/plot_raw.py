#!/usr/bin/env python3
"""Plot raw form factor and electron density for a given index.

Usage
-----
python membrane/plot_raw.py 137
python membrane/plot_raw.py 137 238 403
python membrane/plot_raw.py --random
python membrane/plot_raw.py --random --seed 7
python membrane/plot_raw.py 137 --data-dir membrane --out-dir membrane/figs
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_one(idx, ff_x, ff_y, td_x, td_y, out_dir):
    q = ff_x[idx] * 10
    F = ff_y[idx]

    r   = td_x.reshape(-1) if td_x.ndim == 1 else td_x[idx].reshape(-1)
    rho = td_y[idx].reshape(-1)
    rho = rho - rho[0]

    # Form factor
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(q, F, color="#4c78a8", linewidth=1.2)
    ax.set_xlabel("q  (nm⁻¹)")
    ax.set_ylabel("|F(q)|")
    ax.set_title(f"Form Factor — idx {idx}")
    out_ff = out_dir / f"idx{idx}_ff.png"
    fig.savefig(out_ff, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_ff}")

    # Electron density
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(r, rho, color="#e15759", linewidth=1.4)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("r  (nm)")
    ax.set_ylabel("ρ(r)")
    ax.set_title(f"Electron Density — idx {idx}")
    out_ed = out_dir / f"idx{idx}_ed.png"
    fig.savefig(out_ed, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_ed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("indices", nargs="*", type=int, help="Sample indices to plot")
    parser.add_argument("--random", action="store_true", help="Pick one random index")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for --random")
    parser.add_argument("--data-dir", default=str(Path(__file__).parent),
                        help="Directory with all_ff_x/y.npy and all_td_x/y.npy")
    parser.add_argument("--out-dir",  default=str(Path(__file__).parent / "figs"),
                        help="Output directory for plots")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ff_x = np.load(data_dir / "all_ff_x.npy", allow_pickle=True)
    ff_y = np.load(data_dir / "all_ff_y.npy", allow_pickle=True)
    td_x = np.load(data_dir / "all_td_x.npy", allow_pickle=True)
    td_y = np.load(data_dir / "all_td_y.npy", allow_pickle=True)

    indices = list(args.indices)
    if args.random:
        rng = np.random.default_rng(args.seed)
        idx = int(rng.integers(len(ff_x)))
        print(f"Random index: {idx}")
        indices.append(idx)

    if not indices:
        parser.error("Provide at least one index or use --random")

    for idx in indices:
        plot_one(idx, ff_x, ff_y, td_x, td_y, out_dir)


if __name__ == "__main__":
    main()
