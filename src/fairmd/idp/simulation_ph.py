"""Estimate the pH that a simulation represents from its protonation states.

A GROMACS simulation has no pH as such: the pH is encoded in the protonation
states chosen for the titratable groups when the topology was built. This
module reads those states from a topology (normally the TPR file) and
converts them into a pH estimate.

Protonation is determined by counting the hydrogens bonded to each
titratable heavy atom. Residue names are not used for this, because they do
not reliably encode the state: in TPR files built with CHARMM36m, for
example, HSD/HSE/HSP all appear as ``HIS``.

pH estimate
-----------
Each titratable site implies a bound on the pH, using model pKa values of
unfolded peptides (Thurlkill et al., Protein Sci. 15, 1214 (2006)), which
are appropriate for intrinsically disordered proteins:

- a protonated site implies pH < pKa (upper bound),
- a deprotonated site implies pH > pKa (lower bound).

The consistent pH range is [max lower bound, min upper bound]. The reported
pH is 7.0 if that value lies in the range (the usual intent of a simulation
with standard protonation states), otherwise the midpoint of the range. If
the range is open on one side, the estimate is one pH unit beyond the finite
bound. If the bounds contradict each other (lower > upper), the midpoint is
reported and the result is flagged as inconsistent.
"""

import math
from collections import Counter

# Model pKa values of ionizable groups in alanine pentapeptides,
# Thurlkill et al., Protein Sci. 15, 1214-1218 (2006).
MODEL_PKA = {
    "CTERM": 3.67,
    "ASP": 3.67,
    "GLU": 4.25,
    "HIS": 6.54,
    "CYS": 8.55,
    "TYR": 9.84,
    "LYS": 10.40,
    "NTERM": 8.00,
}

DEFAULT_PH = 7.0

# Residue-name aliases used by common GROMACS force fields (AMBER, CHARMM,
# GROMOS, OPLS). Only used to decide which atoms to inspect, not the state.
RESIDUE_FAMILIES = {
    "ASP": {"ASP", "ASH", "ASPP", "ASPH", "ASP1"},
    "GLU": {"GLU", "GLH", "GLUP", "GLUH", "GLU1"},
    "HIS": {"HIS", "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "HISD", "HISE",
            "HISH", "HISA", "HISB", "HIS1", "HIS2", "HSC"},
    "LYS": {"LYS", "LYN", "LSN", "LYSH", "LYP"},
    "CYS": {"CYS", "CYX", "CYM", "CYS2", "CYSH", "CYN"},
    "TYR": {"TYR", "TYM"},
}

# Titratable heavy atoms per family, and the number of hydrogens bonded to
# them in the protonated and deprotonated state.
TITRATABLE_ATOMS = {
    "ASP": (("OD1", "OD2"), 1, 0),
    "GLU": (("OE1", "OE2"), 1, 0),
    "HIS": (("ND1", "NE2"), 2, 1),
    "LYS": (("NZ",), 3, 2),
    "CYS": (("SG",), 1, 0),
    "TYR": (("OH",), 1, 0),
}

_FAMILY_OF = {name: family for family, names in RESIDUE_FAMILIES.items() for name in names}


def _is_hydrogen(atom):
    name = atom.name.lstrip("0123456789").upper()
    return name.startswith("H") or 0.5 < float(atom.mass) < 4.5


def _hydrogen_count(atom):
    return sum(1 for neighbor in atom.bonded_atoms if _is_hydrogen(neighbor))


def _heavy_neighbors(atom):
    return [neighbor for neighbor in atom.bonded_atoms if not _is_hydrogen(neighbor)]


def _atom(residue, name):
    matches = [atom for atom in residue.atoms if atom.name == name]
    return matches[0] if matches else None


def _side_chain_site(residue):
    """Protonation state of a titratable side chain, or None if not titratable."""
    family = _FAMILY_OF.get(residue.resname.upper())
    if family is None:
        return None
    names, protonated_h, deprotonated_h = TITRATABLE_ATOMS[family]
    atoms = [_atom(residue, name) for name in names]
    if any(atom is None for atom in atoms):
        return None
    if family == "CYS" and any(
        neighbor.name == "SG" and neighbor.resindex != residue.resindex
        for neighbor in atoms[0].bonded_atoms
    ):
        return None  # disulfide bridge, not titratable
    n_h = sum(_hydrogen_count(atom) for atom in atoms)
    if n_h == protonated_h:
        protonated = True
    elif n_h == deprotonated_h:
        protonated = False
    else:
        raise ValueError(
            f"{residue.resname}{residue.resid}: unexpected number of hydrogens "
            f"({n_h}) on titratable atoms {names}"
        )
    return {"group": family, "residue": f"{residue.resname}{residue.resid}", "protonated": protonated}


def _n_terminal_site(residue):
    """Charge state of a free N-terminal amine, or None if the residue is not one."""
    nitrogen, alpha = _atom(residue, "N"), _atom(residue, "CA")
    if nitrogen is None or alpha is None:
        return None
    if any(neighbor.resindex != residue.resindex for neighbor in _heavy_neighbors(nitrogen)):
        return None  # peptide bond or cap
    n_h = _hydrogen_count(nitrogen)
    # Proline has one hydrogen fewer, its nitrogen being bonded to CD.
    delta_carbon = _atom(residue, "CD")
    is_proline = delta_carbon is not None and delta_carbon in nitrogen.bonded_atoms
    charged_h = 2 if is_proline else 3
    if n_h not in (charged_h, charged_h - 1):
        raise ValueError(f"{residue.resname}{residue.resid}: unexpected N-terminal hydrogen count {n_h}")
    return {"group": "NTERM", "residue": f"{residue.resname}{residue.resid}", "protonated": n_h == charged_h}


def _c_terminal_site(residue):
    """Charge state of a free C-terminal carboxyl, or None if the residue is not one."""
    carbon = _atom(residue, "C")
    if carbon is None or _atom(residue, "CA") is None:
        return None
    neighbors = _heavy_neighbors(carbon)
    if any(neighbor.resindex != residue.resindex for neighbor in neighbors):
        return None  # peptide bond or cap
    oxygens = [neighbor for neighbor in neighbors if neighbor.name.upper().startswith("O")]
    if len(oxygens) != 2:
        return None  # e.g. amidated C-terminus
    n_h = sum(_hydrogen_count(oxygen) for oxygen in oxygens)
    if n_h not in (0, 1):
        raise ValueError(f"{residue.resname}{residue.resid}: unexpected C-terminal hydrogen count {n_h}")
    return {"group": "CTERM", "residue": f"{residue.resname}{residue.resid}", "protonated": n_h == 1}


def protonation_sites(universe, selection="protein"):
    """List the titratable sites of the selected residues and their states.

    Args:
        universe: MDAnalysis Universe built from a topology with bonds (TPR).
        selection: atom selection defining the protein.

    Returns:
        list of dicts with keys ``group`` (ASP, GLU, HIS, LYS, CYS, TYR,
        NTERM, CTERM), ``residue`` and ``protonated`` (bool).
    """
    if not hasattr(universe.atoms, "bonds") or len(universe.bonds) == 0:
        raise ValueError("Topology has no bond information; protonation states cannot be determined")
    protein = universe.select_atoms(selection)
    if len(protein) == 0:
        raise ValueError(f"Selection '{selection}' matched no atoms")
    sites = []
    for residue in protein.residues:
        for finder in (_n_terminal_site, _side_chain_site, _c_terminal_site):
            site = finder(residue)
            if site is not None:
                sites.append(site)
    return sites


def estimate_ph(sites, pka=MODEL_PKA, default_ph=DEFAULT_PH):
    """Estimate the pH implied by a list of protonation sites.

    Returns a dict with ``PH`` (float), ``RANGE`` ([lower, upper], None for an
    open side), ``CONSISTENT`` (bool), ``PKA_SOURCE`` and per-group counts of
    ``PROTONATED`` and ``DEPROTONATED`` sites.
    """
    lower, upper = -math.inf, math.inf
    protonated, deprotonated = Counter(), Counter()
    for site in sites:
        value = pka[site["group"]]
        if site["protonated"]:
            protonated[site["group"]] += 1
            upper = min(upper, value)
        else:
            deprotonated[site["group"]] += 1
            lower = max(lower, value)

    consistent = lower <= upper
    if consistent and lower <= default_ph <= upper:
        ph = default_ph
    elif math.isfinite(lower) and math.isfinite(upper):
        ph = (lower + upper) / 2
    elif math.isfinite(upper):
        ph = upper - 1.0
    elif math.isfinite(lower):
        ph = lower + 1.0
    else:
        ph = default_ph  # no titratable sites

    return {
        "PH": round(float(ph), 2),
        "RANGE": [
            round(float(lower), 2) if math.isfinite(lower) else None,
            round(float(upper), 2) if math.isfinite(upper) else None,
        ],
        "CONSISTENT": bool(consistent),
        "PROTONATED": dict(sorted(protonated.items())),
        "DEPROTONATED": dict(sorted(deprotonated.items())),
        "PKA_SOURCE": "Thurlkill et al., Protein Sci. 15, 1214 (2006)",
    }


def ph_from_topology(topology_file, selection="protein"):
    """Estimate the simulation pH from a topology file (e.g. TPR)."""
    import MDAnalysis as mda

    universe = mda.Universe(topology_file)
    return estimate_ph(protonation_sites(universe, selection))
