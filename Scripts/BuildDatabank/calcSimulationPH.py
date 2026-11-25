"""Determine the pH of each simulation from its TPR file and store it in README.yaml.

For every simulation in Data/Simulations, the TPR file listed in README.yaml
is downloaded to the simulation folder (if not already there), the
protonation states of the titratable groups are read from it, and the pH
they correspond to is estimated with fairmd.idp.simulation_ph. The result is
written to the simulation's README.yaml as

    PH: 7.0
    PH_DETERMINATION:
      CONSISTENT: true
      DEPROTONATED: {ASP: 4, CTERM: 1, GLU: 7}
      METHOD: protonation states in TPR
      PKA_SOURCE: Thurlkill et al., Protein Sci. 15, 1214 (2006)
      PROTONATED: {LYS: 12, NTERM: 1, TYR: 3}
      RANGE: [4.25, 8.0]

RANGE is the pH interval consistent with all protonation states; PH is the
point estimate (see fairmd.idp.simulation_ph for the rules).

Run from anywhere:

    python3 Scripts/BuildDatabank/calcSimulationPH.py [--overwrite] [--id ID ...]
        [--include-archived]

Simulations whose TPR is inside an archive in the repository are skipped by
default (see SKIP_ARCHIVED_TPR below).
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fairmd.idp import NMLDB_SIMU_PATH
from fairmd.idp.databankio import ARCHIVE_EXTENSIONS, download_system_file
from fairmd.idp.simulation_ph import ph_from_topology

# If True, simulations whose TPR is stored inside an archive (zip, tar, ...)
# in the repository are skipped, because getting the TPR would download the
# whole archive, often including the trajectories (many GB). A TPR that is
# already in the simulation folder is used regardless of this setting.
# Can also be overridden from the command line with --include-archived.
SKIP_ARCHIVED_TPR = True


def simulation_readmes(root):
    """All simulation README.yaml files, in a machine-independent order."""
    return sorted(Path(root).rglob("README.yaml"))


def tpr_path(readme_path, info):
    tpr = info.get("TPR")
    if not tpr or not tpr[0] or not tpr[0][0]:
        return None
    return readme_path.parent / tpr[0][0]


def tpr_is_archived(info, tpr_name):
    """True if the repository stores this TPR inside an archive file."""
    source = (info.get("SOURCE_FILES") or {}).get(tpr_name, tpr_name)
    return any(
        part.lower().endswith(ARCHIVE_EXTENSIONS) for part in Path(source).parts
    )


def ensure_tpr(readme_path, info):
    """Return the local TPR path, downloading it in databank style if needed."""
    path = tpr_path(readme_path, info)
    if path is None:
        raise FileNotFoundError("README.yaml lists no TPR file")
    if path.is_file():
        return path
    if info.get("DOI") == "localhost":
        raise FileNotFoundError(f"{path} must be provided by the user (DOI: localhost)")
    print(f"  downloading {path.name} from {info.get('DOI')}", flush=True)
    download_system_file(info, path.name, str(path))
    return path


def process(readme_path, overwrite=False, skip_archived=SKIP_ARCHIVED_TPR):
    with open(readme_path) as handle:
        info = yaml.safe_load(handle)

    if info.get("TYPEOFSYSTEM", "protein") != "protein":
        return "skipped (not a protein system)"
    if "PH" in info and not overwrite:
        return f"skipped (PH already {info['PH']})"
    software = str(info.get("SOFTWARE") or "gromacs").lower()
    if software != "gromacs":
        return f"skipped (software {software}, TPR needed)"

    local_tpr = tpr_path(readme_path, info)
    if (
        skip_archived
        and local_tpr is not None
        and not local_tpr.is_file()
        and tpr_is_archived(info, local_tpr.name)
    ):
        return "skipped (TPR is inside an archive; use --include-archived to download it)"

    tpr = ensure_tpr(readme_path, info)
    result = ph_from_topology(str(tpr))

    info["PH"] = result.pop("PH")
    result["METHOD"] = "protonation states in TPR"
    info["PH_DETERMINATION"] = result

    # Same serialization as the other BuildDatabank scripts (FileHandler.write_yaml).
    with open(readme_path, "w") as handle:
        yaml.safe_dump(info, handle)

    note = "" if result["CONSISTENT"] else "  WARNING: protonation states are mutually inconsistent"
    return f"PH = {info['PH']}  range {result['RANGE']}{note}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overwrite", action="store_true", help="Recompute PH even if README.yaml already has it")
    parser.add_argument("--id", type=int, action="append", help="Process only the simulation with this ID; may be repeated")
    parser.add_argument(
        "--include-archived",
        action="store_true",
        help="Also download TPRs stored inside archives (overrides SKIP_ARCHIVED_TPR = True)",
    )
    args = parser.parse_args()
    skip_archived = SKIP_ARCHIVED_TPR and not args.include_archived

    failures = []
    for readme_path in simulation_readmes(NMLDB_SIMU_PATH):
        if args.id:
            with open(readme_path) as handle:
                if (yaml.safe_load(handle) or {}).get("ID") not in args.id:
                    continue
        relative = os.path.relpath(readme_path.parent, NMLDB_SIMU_PATH)
        print(relative, flush=True)
        try:
            print("  " + process(readme_path, overwrite=args.overwrite, skip_archived=skip_archived), flush=True)
        except Exception as error:  # keep going; report at the end
            print(f"  FAILED: {error}", flush=True)
            traceback.print_exc(limit=1)
            failures.append((relative, str(error)))

    if failures:
        print(f"\n{len(failures)} simulation(s) failed:")
        for relative, error in failures:
            print(f"  {relative}: {error}")


if __name__ == "__main__":
    main()
