"""
This module contains the buffer molecule data dictionary used by BufferManager.
It's separated to keep the main buffer management code clean and focused.
"""
from typing import Dict, Any

# Buffer molecule data containing properties of different buffer components
# Each entry contains information about whether the component is pH dependent,
# its charges, and stoichiometry. For pH-dependent components, it includes
# the pH ranges with their corresponding charges and stoichiometry.
BUFFER_MOLECULE_DATA: Dict[str, Dict[str, Any]] = {
    "LysRS residues 1-72": {
        "ph_dependent": False,
        "charges": [0],
        "stoichiometry": [1],
    },
    "sodium chloride": {
        "ph_dependent": False,
        "charges": [1, -1],
        "stoichiometry": [1, 1],
    },
    "sodium phosphate": {
        "ph_dependent": True,
        "ph_ranges": [
            {"min_ph": 0, "max_ph": 2, "charges": [0], "stoichiometry": [1]},
            {"min_ph": 2, "max_ph": 7, "charges": [1, -1], "stoichiometry": [1, 1]},
            {"min_ph": 7, "max_ph": 12, "charges": [1, -2], "stoichiometry": [2, 1]},
            {"min_ph": 12, "max_ph": 14, "charges": [1, -3], "stoichiometry": [3, 1]},
        ],
    },
    "SOD": {
        "ph_dependent": False,
        "charges": [1],
        "stoichiometry": [1],
    },
    "CLA": {
        "ph_dependent": False,
        "charges": [-1],
        "stoichiometry": [1],
    },
    "HEPES": {
        "ph_dependent": True,
        "ph_ranges": [
            {"min_ph": 0, "max_ph": 7.5, "charges": [0], "stoichiometry": [1]},
            {"min_ph": 7.6, "max_ph": 14, "charges": [1], "stoichiometry": [1]},
        ],
    },
    "DTT": {"ph_dependent": False, "charges": [0], "stoichiometry": [1]},
    "PMSF": {"ph_dependent": False, "charges": [0], "stoichiometry": [1]},
    "D2O": {"ph_dependent": False, "charges": [0], "stoichiometry": [1]},
    "H2O": {"ph_dependent": False, "charges": [0], "stoichiometry": [1]},
    "DSS": {"ph_dependent": False, "charges": [1, -1], "stoichiometry": [1, 1]},
}
