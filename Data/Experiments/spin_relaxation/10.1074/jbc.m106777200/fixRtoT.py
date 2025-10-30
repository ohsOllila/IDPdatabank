import yaml

# Load the dictionary from file
with open("spin_relaxation_times.yaml") as f:
    data = yaml.safe_load(f)

# Iterate over residues and fields
for residue_key, residue_data in data.items():
    for field_key, field_data in residue_data.items():  # e.g., '600.0'
        for relaxation_type in ["T1", "T2"]:
            if relaxation_type in field_data:
                value = field_data[relaxation_type]["value"]
                if value != 0:  # avoid division by zero
                    field_data[relaxation_type]["value"] = 1.0 / value
                else:
                    field_data[relaxation_type]["value"] = None  # or np.nan

# Optional: Save back to YAML
with open("spin_relaxation_fixed.yaml", "w") as f:
    yaml.dump(data, f, sort_keys=True)
