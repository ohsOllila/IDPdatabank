#!/bin/bash

replicas=(01 02 03 04 05)
forcefields=("CHARMM36M" "DESAMBER" "AMBER03WS" "AMBER99SB-DISP" "AMBER99SBWS")

# detect highest existing folder
maxdir=$(ls -d [0-9]* 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)

if [[ -z "$maxdir" ]]; then
    maxdir=0
fi

dir=$((maxdir + 1))

for r in "${replicas[@]}"; do
for ff in "${forcefields[@]}"; do

    mkdir -p "$dir"

    # DOI logic
    if [[ "$r" == "01" ]]; then
        doi="10.5281/zenodo.19466393"
    else
        doi="10.5281/zenodo.18835936"
    fi

    # water mapping
    if [[ "$ff" == "CHARMM36M" || "$ff" == "DESAMBER" ]]; then
        watermap="mappingTIP4Pwater.yaml"
    else
        watermap="mappingTIP4PSwater.yaml"
    fi

    # ion mapping (UPDATED RULE)
    if [[ "$ff" == "CHARMM36M" ]]; then
        ionname="SOD"
        ionmap="mappingSOD.yaml"
    else
        ionname="NA"
        ionmap="mappingSODecc.yaml"
    fi

    yaml="$dir/replica_${r}_${ff}_md_2000ns.yaml"

    cat > "$yaml" <<EOF
DOI: $doi

SOFTWARE: gromacs
TRJ: replica_${r}_${ff}_md_2000ns.xtc
TPR: replica_${r}_${ff}_md_2000ns.tpr
PREEQTIME: 0
TIMELEFTOUT: 0

COMPOSITION:
 PROTEIN:
 SOL:
  NAME: SOL
  MAPPING: $watermap
 SOD:
  NAME: $ionname
  MAPPING: $ionmap

DIR_WRK: /home/cmcajsa/Documents

PUBLICATION: https://doi.org/10.26434/chemrxiv-2025-m5m0p
AUTHORS_CONTACT: Samuli Ollila

SYSTEM: SNAP25_noions
SOFTWARE_VERSION:

FF: $ff
FF_SOURCE:
FF_DATE:

CPT:
LOG:
TOP:
GRO:

TYPEOFSYSTEM: protein
EOF

    ((dir++))

done
done
