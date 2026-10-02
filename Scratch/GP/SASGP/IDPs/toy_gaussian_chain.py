"""Ideal Gaussian chains, their pair-distance prior, and normalized scattering.

Equal point scatterers; no excluded volume, hydration, atom form factors or GP.
Reference: https://www.eng.yale.edu/polymers/docs/classes/polyphys/lecture_notes/6/handout6_wsu5.html
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--beads', type=int, default=60)
    parser.add_argument('--chains', type=int, default=3000)
    parser.add_argument('--step', type=float, default=3.8, help='RMS Gaussian step in angstroms; illustrative, not a fixed bond length')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'outputs/toy_chain')
    args = parser.parse_args()
    n, b = args.beads, args.step
    if n < 2 or args.chains < 3 or not np.isfinite(b) or b <= 0:
        parser.error('Require beads >= 2, chains >= 3, and finite step > 0')
    rng = np.random.default_rng(args.seed)
    steps = rng.normal(0, b / np.sqrt(3), (args.chains, n - 1, 3))
    xyz = np.concatenate((np.zeros((args.chains, 1, 3)), steps.cumsum(axis=1)), axis=1)
    xyz -= xyz.mean(axis=1, keepdims=True)
    rg2 = b*b*(n*n-1)/(6*n)
    measured_rg2 = np.mean(np.sum(xyz*xyz, axis=2))
    # Distinct unordered pairs, normalized to integral one. Self pairs added below.
    i, j = np.triu_indices(n, 1)
    edges = np.linspace(0, 6*b*np.sqrt(n-1), 1201)
    hist = np.zeros(len(edges)-1)
    for chain in xyz:
        d = np.linalg.norm(chain[i] - chain[j], axis=1)
        hist += np.histogram(d, bins=edges)[0]
    r = (edges[:-1]+edges[1:])/2
    dr = edges[1]-edges[0]
    empirical = hist/(args.chains*len(i)*dr)
    s = np.arange(1, n)
    weights = 2*(n-s)/(n*(n-1))
    def pair_pdf(separation):
        variance = separation*b*b/3
        return np.sqrt(2/np.pi)*r*r/variance**1.5*np.exp(-r*r/(2*variance))
    components = pair_pdf(s[:, None])
    prior = weights @ components
    q = np.linspace(0, 0.5, 251)
    kernel = np.sinc(np.outer(q, r)/np.pi)
    # Include equal-scatterer self contribution 1/N for exact finite-chain I/I(0).
    analytic_i = 1/n + (1-1/n)*(np.exp(-q[:, None]**2*s*b*b/6) @ weights)
    transformed_i = 1/n + (1-1/n)*(kernel @ prior)*dr
    empirical_i = 1/n + (1-1/n)*(kernel @ empirical)*dr
    transform_error = np.max(np.abs(transformed_i-analytic_i))
    assert transform_error < 1e-4, transform_error
    assert abs(prior.sum()*dr-1) < 1e-4
    assert abs(empirical.sum()*dr-1) < 1e-6
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    ax = fig.add_subplot(221, projection='3d')
    for k in range(3):
        ax.plot(*xyz[k].T, linewidth=1.2, label=f'Chain {k+1}')
    limit = np.abs(xyz[:3]).max()*1.05
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit),
           xlabel='x (Å)', ylabel='y (Å)', zlabel='z (Å)', title='Three independent random-walk conformations')
    ax.set_box_aspect((1, 1, 1))
    ax = fig.add_subplot(222)
    for sep in sorted(set([1, max(1, n//10), max(1, n//2), n-1])):
        ax.plot(r, pair_pdf(sep), label=f'{sep} steps apart')
    ax.set(xlim=(0, 5*np.sqrt(rg2)), xlabel='Pair separation r (Å)', ylabel='Probability density (Å⁻¹)',
           title='Pairs further apart along the chain spread further')
    ax.legend()
    ax = fig.add_subplot(223)
    ax.plot(r, empirical, color='C1', label=f'Histogram from {args.chains:,} chains')
    ax.plot(r, prior, '--', color='black', label='Analytic Gaussian-chain prior mean')
    ax.set(xlim=(0, 5*np.sqrt(rg2)), xlabel='Pair separation r (Å)', ylabel='Distinct-pair P(r) (Å⁻¹)',
           title='Combine all pair separations: this is the prior shape')
    ax.legend()
    ax = fig.add_subplot(224)
    ax.plot(q, empirical_i, color='C1', label='Sinc transform of sampled pairs')
    ax.plot(q, analytic_i, '--', color='black', label='Exact Gaussian-chain ensemble')
    ax.set(yscale='log', xlabel='q (Å⁻¹)', ylabel='I(q)/I(0)', title='The same ensemble in scattering space')
    ax.legend()
    fig.suptitle(f'Ideal Gaussian chain: {n} beads, RMS step {b:g} Å, ensemble RMS Rg = {np.sqrt(rg2):.2f} Å\nEqual point scatterers · independent steps · chains may cross', fontsize=14)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/'gaussian_chain.png', dpi=180)
    plt.close(fig)
    np.save(out/'coordinates_A.npy', xyz)
    np.savetxt(out/'pair_distribution.csv', np.column_stack((r, prior, empirical)), delimiter=',', header='r_A,analytic_P_per_A,sampled_P_per_A', comments='')
    np.savetxt(out/'intensity.csv', np.column_stack((q, analytic_i, transformed_i, empirical_i)), delimiter=',', header='q_per_A,exact_I_normalized,prior_transform_I_normalized,sampled_I_normalized', comments='')
    report = (f'N={n}; RMS step={b} Å; chains={args.chains}; seed={args.seed}\n'
              f'Theory sqrt(mean Rg^2)={np.sqrt(rg2):.6g} Å; sampled={np.sqrt(measured_rg2):.6g} Å\n'
              f'Max analytic-vs-sinc intensity error={transform_error:.6g}\n'
              'P(r) integrates to one for distinct pairs; I(q)/I(0) = 1/N + (1-1/N) integral P(r) sinc(qr) dr.\n')
    (out/'summary.txt').write_text(report)
    print(report)
    print(out/'gaussian_chain.png')


if __name__ == '__main__':
    main()
