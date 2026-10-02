"""Fit experimental SAXS intensities with NUTS and inferred constant noise."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from saxs_gp import fit_saxs


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=here/'data/SASDNV6/curve_native.csv')
    parser.add_argument('--output-dir', type=Path, default=here/'outputs/SASDNV6')
    parser.add_argument('--q-units', choices=['A^-1', 'nm^-1'], default='A^-1')
    parser.add_argument('--q-max', type=float, default=.15, help='Maximum fitted q in inverse Å')
    parser.add_argument('--nuts-warmup', type=int, default=750)
    parser.add_argument('--nuts-samples', type=int, default=1500)
    parser.add_argument('--nuts-chains', type=int, default=2)
    parser.add_argument('--seed', type=int, default=43)
    args = parser.parse_args()
    data = np.loadtxt(args.source, delimiter=',', skiprows=1)
    q = data[:, 0] * (.1 if args.q_units == 'nm^-1' else 1.)
    use = q <= args.q_max
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = dict(source=str(args.source.resolve()), native_q_units=args.q_units,
                  unit_basis='For SASDNV6, inferred from low-q Guinier slope (~36 Å) agreeing with deposited Rg 3.6 nm; native file lacks a unit declaration.',
                  q_max_per_A=args.q_max, points_total=len(q), points_fitted=int(use.sum()),
                  preprocessing='Upper-q cutoff only; no averaging, normalization, added noise, or intensity clipping',
                  likelihood='Constant Gaussian noise SD sampled jointly with kernel parameters; reported errors unused')
    (args.output_dir/'experimental_input.json').write_text(json.dumps(record, indent=2)+'\n')
    np.savetxt(args.output_dir/'experimental_input.csv',np.column_stack((q[use],data[use,1])),
               delimiter=',',header='q_per_A,I',comments='')
    print(json.dumps(record), flush=True)
    report = fit_saxs(q[use], data[use,1], args.output_dir, inference='nuts',
                     warmup=args.nuts_warmup, samples=args.nuts_samples,
                     chains=args.nuts_chains, seed=args.seed,
                     guinier_q_max=.036, reference_rg_nm=(3.6 if args.source.resolve() == (here/'data/SASDNV6/curve_native.csv').resolve() else None))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
