#!/usr/bin/env python3

import yaml
import re

with open("chemical_shifts.yaml") as f:
    data = yaml.safe_load(f)

new_data = {}

for residue, values in data.items():
    m = re.match(r"(\d+)([A-Z]+)", residue)
    if not m:
        continue

    resnum = int(m.group(1)) + 193
    resname = m.group(2)

    new_data[f"{resnum}{resname}"] = values

with open("chemical_shifts_renumbered.yaml", "w") as f:
    yaml.dump(new_data, f, sort_keys=False)

print("Written chemical_shifts_renumbered.yaml")
