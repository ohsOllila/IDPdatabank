import os
import re

# Root folder containing the subfolders
ROOT_FOLDER = "../../Data/Simulations/"

# Regex to match lines like "ID: 123"
id_pattern = re.compile(r"ID:\s*(\d+)")

# Step 1: Collect all existing IDs
max_id = 0
file_ids = {}

for root, dirs, files in os.walk(ROOT_FOLDER):
    if "README.yaml" in files:
        filepath = os.path.join(root, "README.yaml")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        match = id_pattern.search(content)
        if match:
            file_ids[filepath] = int(match.group(1))
            max_id = max(max_id, int(match.group(1)))
        else:
            file_ids[filepath] = None  # No ID yet

# Step 2: Assign new IDs where missing
for filepath, file_id in file_ids.items():
    if file_id is None:
        max_id += 1
        new_id_line = f"ID: {max_id}\n"

        # Prepend ID at the top of the file
        with open(filepath, "r+", encoding="utf-8") as f:
            old_content = f.read()
            f.seek(0, 0)
            f.write(new_id_line + old_content)

        print(f"Added {new_id_line.strip()} to {filepath}")
    else:
        print(f"{filepath} already has ID: {file_id}")



