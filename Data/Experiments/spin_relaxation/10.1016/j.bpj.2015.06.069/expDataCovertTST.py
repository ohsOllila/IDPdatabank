import re
import yaml

#print(insert_pm_symbols(content))

#for line in content:
#    print(line)

#259 0.783 ± 0.003 0.762 ± 0.003 0.756 ± 0.005 0.769 ± 0.001 0.843 ± 0.001"""

# Define the magnetic field strengths


def parse_value_error(value_str, Rdata = False):
    """Parse a value ± error string and return a dictionary with value and error."""
    parts = value_str.split(' ± ')
    if Rdata:
        return {
            'value': 1/float(parts[0]),
            'error': 1/float(parts[1])
        }
    else:
        return {
            'value': float(parts[0]),
            'error': float(parts[1])
        }

def create_nmr_dictionary(nmr_data,raw_data,data_type,field_strengths):
    """Create the nested dictionary structure from the raw data."""
    #nmr_data = {}
    
    # Split the raw data into lines
    lines = raw_data.strip().split('\n')
    
    for line in lines:
        # Split each line into components
        parts = line.split()
        residue = int(parts[0])
        
        # Initialize the residue dictionary
        
        
        # Process each magnetic field strength
        for i, field in enumerate(field_strengths):

            if residue in nmr_data and field not in nmr_data[residue]:
                nmr_data[residue][field] = {}
            elif residue not in nmr_data:
                nmr_data[residue] = {field: {}}

            
            # Extract the value ± error string (parts[i+1])
            value_error_str = parts[i+1] + ' ± ' + parts[i+3] if '±' not in parts[i+1] else parts[i+1]
            
            # Find the correct value±error pair
            value_error_pairs = []
            j = 1
            while j < len(parts):
                if '±' in parts[j]:
                    value_error_pairs.append(parts[j])
                    j += 1
                elif j + 2 < len(parts) and parts[j+1] == '±':
                    value_error_pairs.append(f"{parts[j]} ± {parts[j+2]}")
                    j += 3
                else:
                    j += 1
            
            # Parse the value and error for this field strength
            if i < len(value_error_pairs):
                if data_type == 'T1' or data_type == 'T2':
                    nmr_data[residue][field][data_type] = parse_value_error(value_error_pairs[i], Rdata=True)
                else:
                    nmr_data[residue][field][data_type] = parse_value_error(value_error_pairs[i], Rdata=False)

    return nmr_data


with open("EN2_exp/EN2_R1_exp_pms.dat", "r", encoding="utf-8") as f:
    content = f.read()

raw_data = content

field_strengths = ['9.4T', '11.8T', '14.1T', '18.8T', '23.5T']

# Gyromagnetic ratio of hydrogen in MHz/T
gamma_H = 42.577

# Convert field strengths to frequencies
frequencies = []
for field in field_strengths:
    value_T = float(field.replace('T', ''))  # Remove 'T' and convert to float
    freq = value_T * gamma_H  # MHz
    frequencies.append(round(freq,-1))

#print(frequencies)
# Display results
#for field, freq in zip(field_strengths, frequencies):
#    print(f"{field} => {freq}")


# Create the dictionary
nmr_data = {}
nmr_dict = create_nmr_dictionary(nmr_data,raw_data,'T1',frequencies)




with open("EN2_exp/EN2_NOE_exp_pms.dat", "r", encoding="utf-8") as f:
    content = f.read()

raw_data = content

#field_strengths = ['9.4T', '11.8T', '14.1T', '18.8T', '23.5T']

# Create the dictionary
nmr_data = nmr_dict
nmr_dict = create_nmr_dictionary(nmr_data,raw_data,'hetNOE',frequencies)




#print(nmr_dict)

with open("EN2_exp/EN2_R2_exp_pms.dat", "r", encoding="utf-8") as f:
    content = f.read()

raw_data = content

field_strengths = ['18.8T']

frequencies = []
for field in field_strengths:
    value_T = float(field.replace('T', ''))  # Remove 'T' and convert to float
    freq = value_T * gamma_H  # MHz
    frequencies.append(round(freq,-1))


# Create the dictionary
nmr_data = nmr_dict
nmr_dict = create_nmr_dictionary(nmr_data,raw_data,'T2',frequencies)



with open('spin_relaxation_times.yaml', 'w') as file:
    yaml.dump(nmr_dict, file, default_flow_style=False)


#print(nmr_dict)

# Display a sample of the data structure
print("Sample of the NMR R1 relaxation data dictionary:")
print("=" * 50)

# Show first few residues as examples
#sample_residues = [145, 146, 147]
#for residue in sample_residues:
#    if residue in nmr_dict:
#        print(f"\nResidue {residue}:")
#        print(f"  R1:")
#        for field in field_strengths:
#            data = nmr_dict[residue]['R1'][field]
#            print(f"    {field}: value={data['value']:.3f}, error={data['error']:.3f}")

print(f"\nTotal number of residues: {len(nmr_dict)}")
print(f"Magnetic field strengths: {field_strengths}")

# Show how to access specific data
#print(f"\nExample access:")
#print(f"Residue 145, R1 at 9.4T: {nmr_dict[145]['R1']['9.4T']}")
#print(f"Value: {nmr_dict[145]['R1']['9.4T']['value']}")
#print(f"Error: {nmr_dict[145]['R1']['9.4T']['error']}")

# Export the complete dictionary (uncomment to see full output)
import json
print("\nComplete dictionary (JSON format):")
print(json.dumps(nmr_dict, indent=2))
