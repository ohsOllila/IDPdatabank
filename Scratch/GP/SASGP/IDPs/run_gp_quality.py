"""Fit each SAXS experiment once and score every linked CRYSOL simulation."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import yaml

from gp_quality import score_simulation
from saxs_gp import fit_saxs


def read_experiment(path):
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            rows.append((float(fields[0]), float(fields[1])))
        except ValueError:
            continue
    if len(rows) < 3:
        raise ValueError(f"too few numeric points in {path}")
    data = np.asarray(rows)
    q = data[:, 0]
    if q.max() > 2:
        q = q / 10
    order = np.argsort(q)
    return q[order], data[order, 1]


def read_simulation(path):
    records = yaml.safe_load(path.read_text())
    q, intensity = [], []
    for record in records:
        q_value = record.get("q[1/A]", record.get("q"))
        value = next((record[key] for key in
                      ("mean_I(q)[a.u.]", "mean_Inten[a.u.]", "I(q)[a.u.]")
                      if key in record), None)
        if q_value is not None and value is not None:
            q.append(q_value)
            intensity.append(value)
    return np.asarray(q, float), np.asarray(intensity, float)


def thin_evenly(q, intensity, maximum):
    if len(q) <= maximum:
        return q, intensity
    indices = np.unique(np.round(np.linspace(0, len(q) - 1, maximum)).astype(int))
    return q[indices], intensity[indices]


def linked_simulations(root):
    for readme in (root / "Data/Simulations").rglob("README.yaml"):
        metadata = yaml.safe_load(readme.read_text())
        experiment = metadata.get("EXPERIMENT", {}).get("saxs", {})
        for experiment_id in experiment.get("path", []) or []:
            saxs = readme.parent / "SAXS.yaml"
            if saxs.exists():
                yield experiment_id, metadata, saxs


def safe_name(value):
    return value.replace("/", "_")


def main():
    here = Path(__file__).resolve().parent
    root_default = here.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--output-dir", type=Path, default=here / "outputs/databank_quality")
    parser.add_argument("--q-max", type=float, default=0.5)
    parser.add_argument("--max-points", type=int, default=200,
                        help="Deterministic thinning limit for cubic-cost GP inference")
    parser.add_argument("--nuts-warmup", type=int, default=500)
    parser.add_argument("--nuts-samples", type=int, default=500)
    parser.add_argument("--nuts-chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--scale-only", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    os.environ["NMLDB_ROOT_PATH"] = str(args.root)

    links = list(linked_simulations(args.root))
    experiment_ids = sorted({item[0] for item in links})
    for experiment_id in experiment_ids:
        source = args.root / "Data/Experiments/saxs" / experiment_id / "saxs.dat"
        destination = args.output_dir / safe_name(experiment_id)
        intensity_file = destination / "saxs_gp_intensity.csv"
        if args.refit or not intensity_file.exists():
            q, intensity = read_experiment(source)
            use = q <= args.q_max
            q, intensity = thin_evenly(q[use], intensity[use], args.max_points)
            if len(q) < 3:
                raise ValueError(f"fewer than three points at q <= {args.q_max} for {experiment_id}")
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "experimental_input.json").write_text(json.dumps({
                "source": str(source.relative_to(args.root)), "q_max_per_A": args.q_max,
                "points_fitted": len(q), "maximum_points": args.max_points,
                "thinning": "Evenly spaced row indices after q cutoff",
            }, indent=2) + "\n")
            fit_saxs(q, intensity, destination, inference="nuts",
                     warmup=args.nuts_warmup, samples=args.nuts_samples,
                     chains=args.nuts_chains, seed=args.seed)

    rows = []
    residual_dir = args.output_dir / "residuals"
    residual_dir.mkdir(parents=True, exist_ok=True)
    for experiment_id, metadata, saxs_path in links:
        gp = np.genfromtxt(args.output_dir / safe_name(experiment_id) /
                           "saxs_gp_intensity.csv", delimiter=",", names=True)
        q_sim, intensity_sim = read_simulation(saxs_path)
        result, detail = score_simulation(
            q_sim, intensity_sim, gp["q_per_A"], gp["I_mean"], gp["I_predictive_sd"],
            fit_background=not args.scale_only,
        )
        result.update(system=metadata.get("SYSTEM"), simulation_id=metadata.get("ID"),
                      trajectory=metadata.get("TRJ", [[None]])[0][0],
                      experiment=experiment_id,
                      simulation_path=str(saxs_path.parent.relative_to(args.root)))
        rows.append(result)
        name = f"{metadata.get('ID')}_{safe_name(experiment_id)}.csv"
        np.savetxt(residual_dir / name,
                   np.column_stack([detail[key] for key in
                                    ("q", "gp_mean", "gp_sd", "simulation_fitted", "z", "q_weight")]),
                   delimiter=",", comments="",
                   header="q_per_A,gp_mean,gp_predictive_sd,simulation_fitted,z,q_weight")

    if rows:
        output = args.output_dir / "gp_quality_scores.csv"
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} comparisons to {output}")


if __name__ == "__main__":
    main()
