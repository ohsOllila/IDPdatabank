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
    "TRIS": {
        "ph_dependent": True,
        "ph_ranges": [
            {"min_ph": 0, "max_ph": 8.07, "charges": [0], "stoichiometry": [1]},
            {"min_ph": 8.07, "max_ph": 14, "charges": [1], "stoichiometry": [1]},
        ],
    },
    "HPO4": {
    "ph_dependent": True,
    "ph_ranges": [
        {"min_ph": 0, "max_ph": 6.2, "charges": [-1], "stoichiometry": [1]},  # mostly H2PO4-
        {"min_ph": 6.2, "max_ph": 8.2, "charges": [-1, -2], "stoichiometry": [0.5, 0.5]},  # buffer region
        {"min_ph": 8.2, "max_ph": 14, "charges": [-2], "stoichiometry": [1]},  # mostly HPO4^2-
    ],
    },
    "H2PO4": {
    "ph_dependent": True,
    "ph_ranges": [
        {"min_ph": 0, "max_ph": 6.2, "charges": [-1], "stoichiometry": [1]},  # mostly H2PO4-
        {"min_ph": 6.2, "max_ph": 8.2, "charges": [-1, -2], "stoichiometry": [0.5, 0.5]},  # buffer region
        {"min_ph": 8.2, "max_ph": 14, "charges": [-2], "stoichiometry": [1]},  # mostly HPO4^2-
    ],
    },
    "TCEP": {
        "ph_dependent": False,
        "charges": [0, 0],
        "stoichiometry": [1, 1],
    },
    "glycerol": {
        "ph_dependent": False,
        "charges": [0, 0],
        "stoichiometry": [1, 1],
    },
    "sodium chloride": {
        "ph_dependent": False,
        "charges": [1, -1],
        "stoichiometry": [1, 1],
    },
    "NaCl": {
        "ph_dependent": False,
        "charges": [1, -1],
        "stoichiometry": [1, 1],
    },
    "sodium azide": {
        "ph_dependent": False,
        "charges": [1, -1],
        "stoichiometry": [1, 1],
    },
    "NaN3": {
        "ph_dependent": False,
        "charges": [1, -1],
        "stoichiometry": [1, 1],
    },

    "NaN3 solution": {
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
    "sodium succinate": {
    "ph_dependent": True,
    "ph_ranges": [
        {"min_ph": 0, "max_ph": 4, "charges": [0], "stoichiometry": [1]},
        {"min_ph": 4, "max_ph": 6, "charges": [1, -1], "stoichiometry": [1, 1]},
        {"min_ph": 6, "max_ph": 14, "charges": [1, -2], "stoichiometry": [2, 1]},
    ],
    },
    "sodium succinat": {
    "ph_dependent": True,
    "ph_ranges": [
        {"min_ph": 0, "max_ph": 4, "charges": [0], "stoichiometry": [1]},
        {"min_ph": 4, "max_ph": 6, "charges": [1, -1], "stoichiometry": [1, 1]},
        {"min_ph": 6, "max_ph": 14, "charges": [1, -2], "stoichiometry": [2, 1]},
    ],
    },
    "SOD": {
        "ph_dependent": False,
        "charges": [1],
        "stoichiometry": [1],
    },
    "POT": {
        "ph_dependent": False,
        "charges": [1],
        "stoichiometry": [1],
    },
    "calcium": {
        "ph_dependent": False,
        "charges": [2],
       "stoichiometry": [2],
    },
    "CAL": {
        "ph_dependent": False,
        "charges": [2],
       "stoichiometry": [2],
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
    "MES": {
    "ph_dependent": True,
    "ph_ranges": [
        {"min_ph": 0, "max_ph": 6.0, "charges": [0], "stoichiometry": [1]},
        {"min_ph": 6.1, "max_ph": 14, "charges": [-1], "stoichiometry": [1]}
    ]
    },
    "EDTA": {
    "ph_dependent": True,
    "ph_ranges": [
        {"min_ph": 0, "max_ph": 1.5, "charges": [0], "stoichiometry": [1]},
        {"min_ph": 1.5, "max_ph": 2.5, "charges": [1, -1], "stoichiometry": [1, 1]},
        {"min_ph": 2.5, "max_ph": 4, "charges": [1, -2], "stoichiometry": [2, 1]},
        {"min_ph": 4, "max_ph": 7, "charges": [1, -3], "stoichiometry": [3, 1]},
        {"min_ph": 7, "max_ph": 11, "charges": [1, -4], "stoichiometry": [4, 1]},
        {"min_ph": 11, "max_ph": 14, "charges": [1, -4], "stoichiometry": [4, 1]},
    ],
    },
    "DTT": {"ph_dependent": False, "charges": [0], "stoichiometry": [1]},
    "PMSF": {"ph_dependent": False, "charges": [0], "stoichiometry": [1]},
    "D2O": {"ph_dependent": False, "charges": [0], "stoichiometry": [1]},
    "H2O": {"ph_dependent": False, "charges": [0], "stoichiometry": [1]},
    "DSS": {"ph_dependent": False, "charges": [1, -1], "stoichiometry": [1, 1]},
}
