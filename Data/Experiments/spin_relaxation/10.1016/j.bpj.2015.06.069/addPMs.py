def insert_pm_symbols(line):
    # Split by whitespace, preserving numeric parts
    parts = line.strip().split()
    
    if len(parts) < 3:
        return line  # return unchanged if not enough parts
    
    # First two parts are residue and first value
    modified = parts[:2]
    
    # Insert ± between every two following numbers
    for i in range(2, len(parts), 2):
        if i + 1 < len(parts):
            modified.append(f"{parts[i]} ± {parts[i+1]}")
        else:
            modified.append(parts[i])  # add remaining if odd number

    return "  ".join(modified)


def format_line_with_plus_minus(line):
    parts = line.strip().split()
    
    if len(parts) < 3:
        return line  # Not enough values to format

    residue = parts[0]
    values = parts[1:]

    # Group every two numbers with ± between them
    formatted_values = [f"{values[i]} ± {values[i+1]}" for i in range(0, len(values) - 1, 2)]

    return residue + " " + " ".join(formatted_values)


with open("EN2_exp/EN2_NOE_exp.dat", "r", encoding="utf-8") as f:
    lines = f.readlines()

#for line in lines:
#    print(format_line_with_plus_minus(line))

formatted_lines = [format_line_with_plus_minus(line) for line in lines]

# Write to ASCII file
output_path = "EN2_exp/EN2_NOE_exp_pms.dat"
with open(output_path, "w", encoding="utf-8") as f:
    for line in formatted_lines:
        f.write(line + "\n")

print(f"Formatted ASCII data saved to {output_path}")
