"""Overlay a soft-hat shading on an electron density curve.

Two input modes:
  1. all_td npy files (default):
       python membrane/plot_soft_hat.py --index 0
  2. posterior txt file:
       python membrane/plot_soft_hat.py --txt Results/Sullivan_PhysGP_Membrane_r.txt

Hat is drawn symmetrically at ±max(|density|), shaped by a sigmoid-edged window.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def soft_hat(r, center: float, half_width: float, slope: float) -> np.ndarray:
    """Sigmoid-edged hat: 1 inside, 0 outside, smooth transition."""
    lo = center - half_width
    hi = center + half_width
    left  = 1.0 / (1.0 + np.exp(-slope * (r - lo)))
    right = 1.0 / (1.0 + np.exp(-slope * (-(r - hi))))
    return left * right


def main():
    ap = argparse.ArgumentParser()
    # input
    ap.add_argument("--index",     type=int, default=50,
                    help="Curve index into all_td_y.npy (default 0)")
    ap.add_argument("--td-x",      default="membrane/all_td_x.npy",
                    help="Path to all_td_x.npy")
    ap.add_argument("--td-y",      default="membrane/all_td_y.npy",
                    help="Path to all_td_y.npy")
    ap.add_argument("--txt",       default=None,
                    help="Use posterior_r.txt instead of all_td npy files")
    # hat shape
    ap.add_argument("--center",     type=float, default=0.0)
    ap.add_argument("--half-width", type=float, default=2.4,
                    help="Flat-top half-width in Å (default 2.4)")
    ap.add_argument("--slope",      type=float, default=6.0,
                    help="Edge steepness — larger = sharper (default 6)")
    # appearance
    ap.add_argument("--hat-color",  default="red")
    ap.add_argument("--hat-alpha",  type=float, default=0.40)
    ap.add_argument("--out",        default=None)
    args = ap.parse_args()

    # ── load data ────────────────────────────────────────────────────
    if args.txt:
        data  = np.loadtxt(args.txt, comments="#")
        r     = data[:, 0]
        rho   = data[:, 1]
        label = Path(args.txt).stem
    else:
        r     = np.load(args.td_x)
        y_all = np.load(args.td_y)
        rho   = y_all[args.index]
        label = f"all_td curve {args.index}"

    # shift density so edges sit at 0
    rho = rho - rho[0]

    # ── hat ──────────────────────────────────────────────────────────
    hat    = soft_hat(r, args.center, args.half_width, args.slope)
    height = np.max(np.abs(rho))   # ← peak set to max(|density|)
    hat_pos =  height * hat
    hat_neg = -height * hat

    # ── plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    # ± hat shading
    ax.fill_between(r, 0, hat_pos,
                    color=args.hat_color, alpha=args.hat_alpha,
                    label="Soft hat", zorder=1)
    ax.fill_between(r, hat_neg, 0,
                    color=args.hat_color, alpha=args.hat_alpha,
                    zorder=1)
    # outlines
    ax.plot(r,  hat_pos, color=args.hat_color, lw=1.2, alpha=0.7, zorder=2)
    ax.plot(r,  hat_neg, color=args.hat_color, lw=1.2, alpha=0.7, zorder=2)

    # density
    ax.plot(r, rho, lw=1.8, color="steelblue", label="electron density ρ(r)", zorder=3)

    ax.axhline(0, color="k", lw=0.7, alpha=0.4)
    ax.set_xlabel("r [Å]")
    ax.set_ylabel("ρ(r)")
    ax.set_title(f"Electron density with ±soft-hat shading  [height = max|ρ| = {height:.2f}]")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if args.out:
        out = Path(args.out)
    elif args.txt:
        out = Path(args.txt).with_suffix(".soft_hat.png")
    else:
        out = Path(args.td_x).parent / f"soft_hat_{args.index}.png"

    fig.savefig(out, dpi=150)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
