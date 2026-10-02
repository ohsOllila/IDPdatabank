"""Extract SAXS data, add synthetic noise, and fit a sinc-transform GP."""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = (
    HERE.parent.parent / "IDPdatabank/Data/Simulations/8e2/c75"
    / "8e2c75ee99bf9c665240a422124d0ce0ac6ea357"
    / "5e0ddfacaae9d723c7422b27d1ccda41d8b80358/SAXS.yaml"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic noise random seed")
    parser.add_argument("--skip-gp", action="store_true", help="Only generate the data preview")
    parser.add_argument("--r-max", type=float, default=400., help="GP integration limit in Å")
    parser.add_argument("--r-points", type=int, default=401, help="GP real-space grid size")
    parser.add_argument('--inference', choices=['nuts', 'laplace'], default='nuts')
    parser.add_argument('--prior-fn', choices=['uniform', 'gaussian'], default='uniform')
    parser.add_argument('--nuts-samples', type=int, default=500)
    parser.add_argument('--nuts-warmup', type=int, default=500)
    parser.add_argument('--nuts-chains', type=int, default=2)
    parser.add_argument('--laplace-samples', type=int, default=500)
    parser.add_argument('--inference-seed', type=int, default=43)
    args = parser.parse_args()
    source = args.source.resolve()
    rows = yaml.safe_load(source.read_text())
    values = np.array([
        [row["q[1/A]"], row["mean_I(q)[a.u.]"], row["sd_I(q)[a.u.]"]]
        for row in rows
    ], dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 3:
        raise ValueError("Expected at least two q, mean intensity, SD rows")
    if not np.isfinite(values).all():
        raise ValueError("SAXS data contain non-finite values")
    values = values[np.argsort(values[:, 0])]
    q, intensity, sd = values.T
    if (q < 0).any() or (np.diff(q) <= 0).any() or (sd < 0).any():
        raise ValueError("Require unique nonnegative q and nonnegative SD")
    metadata_path = source.with_name("README.yaml")
    metadata = yaml.safe_load(metadata_path.read_text()) if metadata_path.exists() else {}
    title = (
        f"Simulation {metadata.get('ID', '?')} · {metadata.get('SYSTEM', 'unknown protein')}"
        f" · {metadata.get('FF', 'unknown force field')}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_dir / "saxs_data.csv", values, delimiter=",",
               header="q_A^-1,mean_I_au,sd_I_au", comments="")
    # Treat the mean curve as the truth and use one common noise SD at every q.
    noise_sigma = float(sd.mean())
    noise = np.random.default_rng(args.seed).normal(0.0, noise_sigma, size=q.size)
    noisy_intensity = intensity + noise
    np.savetxt(
        args.output_dir / "saxs_synthetic.csv",
        np.column_stack((q, intensity, noisy_intensity, np.full(q.size, noise_sigma), noise)),
        delimiter=",", comments="",
        header="q_A^-1,mean_I_au,noisy_I_au,noise_sigma_au,added_noise_au",
    )
    (args.output_dir / "source.txt").write_text(
        f"Source: {source}\n{title}\n"
        "Values are unchanged; shaded band is the reported SD, not a standard error.\n"
        f"Synthetic noise: independent Normal(0, sigma^2), sigma = mean(reported SD) = {noise_sigma:.17g}; seed = {args.seed}.\n"
        "Synthetic values are not clipped, including negative intensities.\n"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax in axes:
        ax.plot(q, intensity, ".-", markersize=3, linewidth=1.2, label="Mean intensity")
        lower = intensity - sd
        if ax is axes[1]:
            lower = np.where(lower > 0, lower, np.nan)
        ax.fill_between(q, lower, intensity + sd, alpha=0.22, label="Reported ±1 SD")
        ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.grid(alpha=0.2)
    axes[0].set_title("Linear intensity")
    if (intensity <= 0).any():
        raise ValueError("Log plot requires positive mean intensities")
    axes[1].set_yscale("log")
    axes[1].set_title("Log intensity")
    axes[0].legend(frameon=False)
    fig.suptitle(title)
    output = args.output_dir / "saxs_curve.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax in axes:
        ax.plot(q, intensity, color="C0", linewidth=1.6, label="CRYSOL mean")
        ax.axhline(0, color="0.5", linewidth=0.7)
        ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
        ax.set_ylabel("Intensity (databank units)")
        ax.grid(alpha=0.2)
    axes[0].fill_between(q, intensity - sd, intensity + sd, color="C0", alpha=0.22,
                         label="Reported ±1 SD across frames")
    axes[0].set_title("Original CRYSOL intensity")
    axes[1].plot(q, noisy_intensity, ".", markersize=4,
                 color="C1", alpha=0.75, label="Synthetic experiment")
    axes[1].set_title(f"Constant Gaussian noise: σ = {noise_sigma:.3g}")
    for ax in axes:
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"{title} · seed {args.seed}")
    comparison = args.output_dir / "saxs_noise_comparison.png"
    fig.savefig(comparison, dpi=180)
    plt.close(fig)
    print(f"Loaded {len(q)} points; q = {q.min():g}–{q.max():g} Å^-1")
    print(f"Plot: {output.resolve()}")
    print(f"Extracted data: {(args.output_dir / 'saxs_data.csv').resolve()}")
    print(f"Noise sigma: {noise_sigma:.8g}; seed: {args.seed}")
    print(f"Noise comparison: {comparison.resolve()}")
    if not args.skip_gp:
        from saxs_gp import fit_saxs
        report = fit_saxs(q, noisy_intensity, args.output_dir,
                          r_max=args.r_max, points=args.r_points,
                          inference=args.inference, prior_fn=args.prior_fn,
                          samples=args.nuts_samples if args.inference == 'nuts' else args.laplace_samples,
                          warmup=args.nuts_warmup, chains=args.nuts_chains, seed=args.inference_seed)
        print(f"GP fit: {(args.output_dir / 'saxs_gp_fit.png').resolve()}")
        print(f"GP parameters: {report}")


if __name__ == "__main__":
    main()
