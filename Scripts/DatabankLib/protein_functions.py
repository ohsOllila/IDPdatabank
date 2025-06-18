import os
import subprocess
import numpy as np
import pandas as pd
import glob
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import matplotlib.pyplot as plt
import csv
from scipy import optimize
import requests
from pathlib import Path
import yaml
import requests
import re
import maicos
from typing import List, Dict, Optional

def calculate_contact_probabilities(gro_file, xtc_file, cutoff):
    u = mda.Universe(gro_file, xtc_file)
    CAatoms = u.select_atoms("name CA")
    num_residues = len(CAatoms)

    within_cutoff_count = np.zeros((num_residues, num_residues))
    num_frames = 0

    for ts in u.trajectory:
        distances_array = distances.distance_array(CAatoms.positions, CAatoms.positions)
        within_cutoff_count += (distances_array <= cutoff)
        num_frames += 1

    probabilities = within_cutoff_count / num_frames

    df = pd.DataFrame(probabilities)
    df.columns = [f"{i+1}" for i in range(num_residues)]
    df.index = [f"{i+1}" for i in range(num_residues)]

    csv_filename = f"CA_prob_within_{cutoff}A.csv"
    df.to_csv(csv_filename)
    print(f"Saved {csv_filename} in current directory.")

    return df

# Function to create and save contact map plot
def create_contact_map_plot(probabilities_df, output_file):
    color_map = plt.imshow(probabilities_df.values, cmap='jet', vmin=0, vmax=1, origin='lower')

    cbar = plt.colorbar(color_map)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Probability of contact (≤ 15.0 Å)', rotation=270, labelpad=25, fontsize=18)

    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel("Residue", fontsize=16)
    plt.ylabel("Residue", fontsize=16)

    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
    plt.close()

    print(f"Contact map saved as {output_file}.")

def calculate_distances(gro_file, xtc_file):
	u = mda.Universe(gro_file, xtc_file)
	CAatoms = u.select_atoms("name CA")
            
	num_residues = len(CAatoms)
	sum_distances = np.zeros((num_residues, num_residues))
	num_frames = 0
            
	# Iterate through all frames
	for ts in u.trajectory:
		distances_array = distances.distance_array(CAatoms.positions, CAatoms.positions)
		sum_distances += distances_array
		num_frames += 1
           
	# Calculate average distances
	avg_distances = sum_distances / num_frames
            
	# Create a DataFrame from the average distances
	df = pd.DataFrame(avg_distances)
            
	# Add column and index names
	df.columns = [f"{i+1}" for i in range(num_residues)]
	df.index = [f"{i+1}" for i in range(num_residues)]
            
	# Save the DataFrame as a CSV file
	csv_filename = "CA_avg_distances.csv"
	df.to_csv(csv_filename)
	print(f"Saved {csv_filename}")
       
	return df

# Function to create and save contact map plot
def create_distance_map_plot(probabilities_df, output_file):
#    vmin = np.floor(np.min(values.probabilities_df) / 10) * 10
#    vmax = np.ceil(np.max(values.probabilities_df) / 10) * 10
    vmin = 0
    vmax = 20

    color_map = plt.imshow(probabilities_df.values, vmin=vmin, vmax=vmax, cmap='jet', origin='lower')

    cbar = plt.colorbar(color_map)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Distance (Å)', rotation=270, labelpad=25, fontsize=18)

    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel("Residue", fontsize=16)
    plt.ylabel("Residue", fontsize=16)

    plt.tight_layout()

    plt.savefig(output_file, dpi=600)
    plt.close()
    print(f"Distance map saved as {output_file}.")




def calculate_backbone_correlations(gro_file, xtc_file, output_file):

    outfile_png = output_file
    outfile_csv = output_file.replace(".png",".csv")

    u = mda.Universe(gro_file, xtc_file)
    CAatoms = u.select_atoms("name CA")

    num_atoms = len(CAatoms.ix)
    num_vectors = num_atoms - 1
    vec = np.zeros((num_vectors, 3))
    matrix = np.zeros((num_vectors, num_vectors))

    for frame in u.trajectory:
        for i in range(num_vectors):
            vec[i]=CAatoms[i+1].position-CAatoms[i].position
            vec[i] /= np.linalg.norm(vec[i])
        matrix += np.dot(vec, vec.T)
    matrix /= len(u.trajectory)

    np.savetxt(outfile_csv, matrix, delimiter=',')

    color_map = plt.imshow(matrix,vmin=-1, vmax=1, cmap="seismic", origin='lower')

    cbar = plt.colorbar(color_map)
    cbar.ax.tick_params(labelsize=16)

    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel(r"C$_{\alpha}$ Pair Index", fontsize=16)
    plt.ylabel(r"C$_{\alpha}$ Pair Index", fontsize=16)

    plt.tight_layout()

    plt.savefig(outfile_png, dpi=600)
    plt.close("all")


def calculate_backbone_NH_correlation_functions(gro_file,trj_fname,top_fname,sim_folder):
    NHindex_file = sim_folder + 'NH.ndx'
    correlation_function_folder =  sim_folder + '/correlation_functions/'
    if (not os.path.isdir(correlation_function_folder)):
        execStr = (f"mkdir {correlation_function_folder}")
    try:
        os.system(execStr)
    except:
        pass
    resids = make_index_file_for_backbone_nh_pairs(gro_file,NHindex_file)
    list_of_correlation_functions = []
    for resname in resids:
        correlation_function = sim_folder + '/correlation_functions/NHrotaCF_' + resname +'.xvg'
        list_of_correlation_functions.append( correlation_function)
        if (not os.path.isfile(correlation_function)):
            execStr = (
                f"echo {resids[resname]} | gmx rotacf -f {trj_fname} -s {top_fname} -n {NHindex_file} -o {correlation_function} -P 2 -d -xvg none" 
            )
            os.system(execStr)
    return list_of_correlation_functions

            

def make_index_file_for_backbone_nh_pairs(gro_file, output_ndx):
    # Load the structure
    u = mda.Universe(gro_file)

    # Ensure hydrogens are present
    if not any(atom.name == 'H' for atom in u.atoms):
        raise ValueError("Hydrogen atoms not found. Ensure the gro file includes hydrogens.")

    resids_with_nh = {}
    
    with open(output_ndx, 'w') as ndx:
        pair_count = 0

        for res in u.select_atoms('protein').residues:
            try:
                N_atom = res.atoms.select_atoms('name N')[0]
                H_atom = res.atoms.select_atoms('name H')[0]
            except IndexError:
                continue  # Skip residues without N or H (e.g., termini or incomplete)

            resid = res.resid
            resname =  str(res.resid) + res.resname
            resids_with_nh[resname] = pair_count    #.append(resname)
            
            #group_name = f"N_H_pair_{pair_count}"
            ndx.write(f"[ {resname} ]\n")
            ndx.write(f"{N_atom.index + 1} {H_atom.index + 1}\n\n")  # GROMACS uses 1-based indexing
            pair_count += 1

    return resids_with_nh




def read_correlation_function(input_data):
    # for reading the correlation function data
    opf = open(input_data, 'r')
    lines = opf.readlines()
    data_times = []
    data_F = []
    for i,line in enumerate(lines):
        if '#' in line:
            continue
        if '&' in line:
            continue
        if '@' in line:
            continue    
        if 'label' in line:
            continue
        if line == "":
            continue
        parts = line.split()
        if np.shape(parts)[0]==2:
            try:
                data_F.append(float(parts[1]))
                data_times.append(float(parts[0]))
            except:
                print(i)
                break

    data_Fout = np.array(data_F)
    times_out = np.array(data_times)
    return data_Fout, times_out


def calc_relax_time(correlation_function,magnetic_field,biggest_corr_time,smallest_corr_time,N_exp_to_fit):
   
    # normalized correlation fuction
    ## NcorrF = (self.org_corrF - self.OP ** 2) / (1 - self.OP ** 2);

    correlation_function_data = read_correlation_function(correlation_function)
    
    # Create correlation times from the times and number of exponential specified by the user
    step_exp=(biggest_corr_time-smallest_corr_time)/N_exp_to_fit
    Ctimes = 10 ** np.arange(smallest_corr_time,biggest_corr_time,step_exp)

    # First, no forcing the plateou
    # create exponential functions and put them into a matrix, individual exponentials in columns
    #the lengthe of correlationd data to be used is specified by the user
    NcorrF, times_out = correlation_function_data
    n = len(times_out)
    m = len(Ctimes)
    Cexp_mat = np.zeros((n, m))

    for i in range(0, n):
        for j in range(0, m):
            Cexp_mat[i, j] = np.exp(-times_out[i] / Ctimes[j])

    #least square solution
    Coeffs, res = optimize.nnls(Cexp_mat, NcorrF[0:n])

    # Effective correlation time from components, in units of sec
    
    Teff = sum(Coeffs * Ctimes * 0.001 * 10 ** (-9)) 

    # calculate t_eff from area
    dt = times_out[2] - times_out[1]
    pos = np.argmax(NcorrF[0:n] < 0);

    if pos > 0:
        tau_eff_area = sum(NcorrF[0:pos]) * dt * 0.001 * 10 ** (-9);
        conv = 1
    else:
        tau_eff_area = sum(NcorrF[0:n]) * dt * 0.001 * 10 ** (-9);
        conv = 0

   

    # changin the unit of time permanently
    Ctimes = Ctimes * 0.001 * 10 ** (-9);
    #self.Coeffs=Coeffs
    #self.Ctimes=Ctimes
    Ctimes_ns=Ctimes*10**(9)
	
	
    #Calculate the relaxation times for chosen nuclei
    T1, T2, NOE = get_relaxation_N(magnetic_field,Coeffs,Ctimes)


        
    #get the reconstucted correlation function
    #self.rec_corrF=Cexp_mat.dot(Coeffs)
    #self.plot_fit(self.rec_corrF)
    #self.plot_exp_hist(Ctimes,Coeffs)
        

    return Teff, tau_eff_area, T1, T2, NOE, Coeffs, Ctimes_ns


def calc_dynamic_landscape(correlation_function,biggest_corr_time,smallest_corr_time,N_exp_to_fit):
   
    # normalized correlation fuction
    ## NcorrF = (self.org_corrF - self.OP ** 2) / (1 - self.OP ** 2);

    correlation_function_data = read_correlation_function(correlation_function)
    
    # Create correlation times from the times and number of exponential specified by the user
    step_exp=(biggest_corr_time-smallest_corr_time)/N_exp_to_fit
    Ctimes = 10 ** np.arange(smallest_corr_time,biggest_corr_time,step_exp)

    # First, no forcing the plateou
    # create exponential functions and put them into a matrix, individual exponentials in columns
    #the lengthe of correlationd data to be used is specified by the user
    NcorrF, times_out = correlation_function_data
    n = len(times_out)
    m = len(Ctimes)
    Cexp_mat = np.zeros((n, m))

    for i in range(0, n):
        for j in range(0, m):
            Cexp_mat[i, j] = np.exp(-times_out[i] / Ctimes[j])

    #least square solution
    Coeffs, res = optimize.nnls(Cexp_mat, NcorrF[0:n])

    # Effective correlation time from components, in units of sec
    
    Teff = sum(Coeffs * Ctimes * 0.001 * 10 ** (-9)) 

    # calculate t_eff from area
    dt = times_out[2] - times_out[1]
    pos = np.argmax(NcorrF[0:n] < 0);

    if pos > 0:
        tau_eff_area = sum(NcorrF[0:pos]) * dt * 0.001 * 10 ** (-9);
        conv = 1
    else:
        tau_eff_area = sum(NcorrF[0:n]) * dt * 0.001 * 10 ** (-9);
        conv = 0

   

    # changin the unit of time permanently
    Ctimes = Ctimes * 0.001 * 10 ** (-9);
    #self.Coeffs=Coeffs
    #self.Ctimes=Ctimes
    Ctimes_ns=Ctimes*10**(9)
	
    return Teff, tau_eff_area, Coeffs, Ctimes_ns



def get_relaxation_N(magnetic_field,Coeffs,Ctimes):
    
    gammaH=267.513*10**6;
    gammaN=-27.166*10**6;
    
    #wh = gammaH * magnetic_field
    wh =  magnetic_field * 10**6 * 2 * np.pi
    wn =  wh * gammaN/gammaH 
    
    #initiate spectral densities
    J0 = 0
    JhMn = 0
    JhPn = 0
    Jh = 0
    Jn = 0

    m = len(Ctimes)
    for i in range(0, m):
        Ctimes[i] = Ctimes[i] * 10**(-9)
        #print(Ctimes[i],Coeffs[i])
        w = 0
      
        J0 = J0 + 2 * Coeffs[i] * Ctimes[i] / (1.0 + w * w * Ctimes[i] * Ctimes[i])
        
        w = wh-wn;
        JhMn = JhMn + 2 * Coeffs[i]* Ctimes[i] / (1.0 + w * w * Ctimes[i] * Ctimes[i])

        w = wn;
        Jn = Jn + 2 * Coeffs[i]* Ctimes[i] / (1.0 + w * w * Ctimes[i] * Ctimes[i])
        
        w = wh;
        Jh= Jh + 2 * Coeffs[i]* Ctimes[i] / (1.0 + w * w * Ctimes[i] * Ctimes[i])

        w = wn+wh;
        JhPn = JhPn + 2 * Coeffs[i]* Ctimes[i] / (1.0 + w * w * Ctimes[i] * Ctimes[i])


    mu = 4 * np.pi * 10**(-7) #magnetic constant of vacuum permeability
    h_planck = 1.055 * 10**(-34); #reduced Planck constant
    rN = 0.101 * 10**(-9); # average cubic length of N-H bond
    d = 1 * (mu * gammaN * gammaH * h_planck) / (4 * np.pi * rN**3); # dipolar coupling constant

    #units were corrected by S.Ollila and E.Mantzari, removed 2*pi from R1 and R2
    R1 = (d**2 / 20) * (1 * JhMn + 3 * Jn + 6 * JhPn) + Jn * (wn * 160 * 10**(-6))**2 / 15   ; 
    R2 = 0.5 * (d**2 / 20) * (4 * J0 + 3 * Jn + 1 * JhMn + 6 * Jh + 6 * JhPn) + (wn * 160 * 10**(-6))**2 / 90 * (4 * J0 + 3 * Jn);
    NOE = 1 + (d**2 / 20) * (6 * JhPn - 1 * JhMn) * gammaH / (gammaN * R1);


    #print("T1: {}, T2: {}, NOE: {}".format(1/R1, 1/R2, NOE))
    
    
           
    return 1/R1, 1/R2, NOE


def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

def extract_heteronucl_NOE_data(data):
    for entry in data:
        for entry_id, entry_content in entry.items():
            for noe in entry_content.get('heteronucl_NOEs', []):
                for loop in noe.get('loops', []):
                    if loop.get('category') == '_Heteronucl_NOE':
                        return loop.get('data', [])
    return []        
    
def extract_data_from_BMRB(ID, datatype):

    output = {}
    data = []
    x = requests.get("http://api.bmrb.io/v2/entry/" + ID + "?saveframe_category=" + datatype)
    data.append (x.json())
    for entry in data:
        for entry_id, entry_content in entry.items():
            for noe in entry_content.get(datatype, []):
                for loop in noe.get('loops', []):

                    if datatype == 'heteronucl_NOEs' and loop.get('category') == '_Heteronucl_NOE':
                        rel_data = loop.get('data', [])
                        for i in rel_data:
                            residue = i[5] + i[6]
                            output[residue] = {
                                'value': float(i[19]),
                                'error': float(i[20])
                                }
                        return output

                    if datatype == 'heteronucl_T1_relaxation' and loop.get('category') == '_T1':
                        rel_data = loop.get('data', [])
                        for i in rel_data:
                            residue = i[5] + i[6]
                            output[residue] = {
                                'value': float(i[10]),
                                'error': float(i[11])
                                }
                        return output

                    if datatype == 'heteronucl_T2_relaxation' and loop.get('category') == '_T2':
                        rel_data = loop.get('data', [])
                        for i in rel_data:
                            residue = i[5] + i[6]
                            output[residue] = {
                                'value': float(i[10]),
                                'error': float(i[11])
                                }
                        return output
                        
    return []        

def get_conditions_from_BMRB(BMRBid):
    data = []
    x = requests.get("http://api.bmrb.io/v2/entry/" + BMRBid + "/experiments")
    data.append (x.json())

    exp_data_path = '../../Data/Experiments/spin_relaxation/BMRBid' + BMRBid
    if (not os.path.isdir(exp_data_path)):
        execStr = (f"mkdir {exp_data_path}")
        os.system(execStr)

    T1names = ['Het. Nuc. T1 relaxation', '2D 1H-15N HSQC-T1']
    T2names = ['Het. Nuc. T2 relaxation', '2D 1H-15N HSQC -T2']
    netNOEnames = ['15N-(1H) NOE', '2D 1H-15N HSQC-NOE']
    
    for i in data[0]:
        print(i['Name'])
        if i['Name'] in  T1names:
            metadata_file = exp_data_path + '/T1_metadata.yaml'
            with open(metadata_file, 'w') as file:
                yaml.dump(i, file, sort_keys=True, default_flow_style=False, indent=4)
            print('T1 metadata stored in ', metadata_file)

        if i['Name'] in T2names:
            metadata_file = exp_data_path + '/T2_metadata.yaml'
            with open(metadata_file, 'w') as file:
                yaml.dump(i, file, sort_keys=True, default_flow_style=False, indent=4)
            print('T2 metadata stored in ', metadata_file)

        if i['Name'] in netNOEnames:
            metadata_file = exp_data_path + '/hetNOE_metadata.yaml'
            with open(metadata_file, 'w') as file:
                yaml.dump(i, file, sort_keys=True, default_flow_style=False, indent=4)
            print('hetNOE metadata stored in ', metadata_file)



#    print(data[0]['Het. Nuc. T1 relaxation'])
        


def extract_bmrb_fasta(entry_id):
    """
    Extracts the FASTA sequence for a given BMRB entry ID.

    Args:
        entry_id (str or int): BMRB entry ID (e.g., '50115')

    Returns:
        str: FASTA formatted sequence
    """
    entry_id = str(entry_id)
    url = f"https://bmrb.io/ftp/pub/bmrb/entry_directories/bmr{entry_id}/bmr{entry_id}_3.str"

    response = requests.get(url)
    if not response.ok:
        raise Exception(f"Failed to download NMR-STAR file for BMRB entry {entry_id}")

    star_text = response.text

    # Extract Entity_poly_seq loop
    match = re.search(r'loop_.*?_Entity_poly_seq.*?(?=loop_|stop_)', star_text, re.DOTALL)
    if not match:
        raise Exception("Entity_poly_seq section not found in the NMR-STAR file")

    loop_block = match.group(0)
    lines = loop_block.strip().splitlines()

    # Only include lines that look like sequence rows (starting with an integer)
    data_lines = [line for line in lines if line and re.match(r'^\d+', line)]

    residues = []
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 3:
            residues.append(parts[2].upper())

    # 3-letter to 1-letter amino acid code conversion
    aa_dict = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'SEC': 'U', 'PYL': 'O', 'ASX': 'B', 'GLX': 'Z', 'UNK': 'X'
    }

    sequence = ''
    for res in residues:
        aa = aa_dict.get(res)
        if aa:
            sequence += aa
        else:
            print(f"Warning: Unrecognized residue '{res}' — replaced with 'X'")
            sequence += 'X'

    fasta = f">BMRB_{entry_id}\n{sequence}"
    return fasta




def extract_fasta_from_bmrb(bmrb_id: str) -> List[str]:
    """
    Extract FASTA sequences from a BMRB entry.
    
    Args:
        bmrb_id (str): BMRB entry ID (e.g., '15000', 'bmr15000')
    
    Returns:
        List[str]: List of FASTA formatted sequences
    
    Raises:
        requests.RequestException: If unable to fetch BMRB data
        ValueError: If no sequence data found or invalid BMRB ID
    """
    
    # Clean up BMRB ID - remove 'bmr' prefix if present
    clean_id = bmrb_id.lower().replace('bmr', '')
    
    # BMRB REST API URL for NMR-STAR format
    url = f"https://bmrb.io/ftp/pub/bmrb/entry_directories/bmr{clean_id}/bmr{clean_id}_3.str"
    
    try:
        # Fetch the NMR-STAR file
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        star_data = response.text
        
        # Extract sequences from the NMR-STAR format
        sequences = _parse_sequences_from_star(star_data, clean_id)

        if not sequences:
            raise ValueError(f"No sequence data found for BMRB ID: {bmrb_id}")

        exp_data_path = '../../Data/Experiments/spin_relaxation/BMRBid' + bmrb_id
        if (not os.path.isdir(exp_data_path)):
            execStr = (f"mkdir {exp_data_path}")
            os.system(execStr)

        fasta_file = exp_data_path + '/fasta.yaml'
        sequence_dict = {'sequence': sequences}
        with open(fasta_file, 'w') as file:
            yaml.dump(sequence_dict, file, sort_keys=True, default_flow_style=False, indent=4)

        print('Sequence stored in ', fasta_file)
            
        return sequences
        
    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to fetch BMRB entry {bmrb_id}: {str(e)}")

def _parse_sequences_from_star(star_data: str, bmrb_id: str) -> List[str]:
    """
    Parse sequence information from NMR-STAR format data.
    
    Args:
        star_data (str): Raw NMR-STAR format data
        bmrb_id (str): BMRB ID for FASTA header
    
    Returns:
        List[str]: List of FASTA formatted sequences
    """
    
    sequences = []
    
    # Look for entity sequences in the NMR-STAR file
    # Pattern to match sequence data in NMR-STAR format
    entity_pattern = r'_Entity\.ID\s+(\d+).*?_Entity\.Polymer_seq_one_letter_code\s*\n([^_\n]*(?:\n[^_\n]*)*)'
    
    matches = re.findall(entity_pattern, star_data, re.DOTALL | re.IGNORECASE)
    
    for entity_id, seq_block in matches:
        # Clean up the sequence - remove whitespace, quotes, and newlines
        sequence = re.sub(r'[\'"\s\n\r]', '', seq_block.strip())
        
        # Remove any remaining non-amino acid characters (except standard amino acids)
        sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())
        
        if sequence and len(sequence) > 5:  # Only include sequences with reasonable length
            # Get entity name if available
            entity_name_pattern = rf'_Entity\.ID\s+{entity_id}.*?_Entity\.Name\s*\n([^\n]*)'
            name_match = re.search(entity_name_pattern, star_data, re.DOTALL | re.IGNORECASE)
            
            entity_name = ""
            if name_match:
                entity_name = name_match.group(1).strip().strip('\'"')
            
            # Create FASTA header
            header = f">BMRB:{bmrb_id}_Entity:{entity_id}"
            if entity_name:
                header += f" {entity_name}"
            
            # Format sequence with 80 characters per line (standard FASTA format)
            formatted_seq = '\n'.join([sequence[i:i+80] for i in range(0, len(sequence), 80)])
            
            #fasta_entry = f"{header}\n{formatted_seq}"
            fasta_entry = f"{formatted_seq}"
            sequences.append(fasta_entry)
    
    # Alternative pattern for older BMRB formats
    if not sequences:
        alt_pattern = r'_Entity_poly_seq\.Residue_label\s+(\d+)\s+(\d+)\s+([A-Z]{1,3})'
        residue_matches = re.findall(alt_pattern, star_data)
        
        if residue_matches:
            # Group by entity ID
            entities = {}
            for comp_id, entity_id, residue in residue_matches:
                if entity_id not in entities:
                    entities[entity_id] = []
                # Convert 3-letter to 1-letter amino acid codes
                aa_code = _three_to_one_letter(residue)
                if aa_code:
                    entities[entity_id].append(aa_code)
            
            for entity_id, residues in entities.items():
                sequence = ''.join(residues)
                if len(sequence) > 5:
                    header = f">BMRB:{bmrb_id}_Entity:{entity_id}"
                    formatted_seq = '\n'.join([sequence[i:i+80] for i in range(0, len(sequence), 80)])
                    fasta_entry = f"{header}\n{formatted_seq}"
                    sequences.append(fasta_entry)
    
    return sequences

def _three_to_one_letter(three_letter: str) -> Optional[str]:
    """Convert 3-letter amino acid code to 1-letter code."""
    
    aa_dict = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    return aa_dict.get(three_letter.upper())

def save_fasta_sequences(sequences: List[str], filename: str) -> None:
    """
    Save FASTA sequences to a file.
    
    Args:
        sequences (List[str]): List of FASTA formatted sequences
        filename (str): Output filename
    """
    
    with open(filename, 'w') as f:
        for i, seq in enumerate(sequences):
            if i > 0:
                f.write('\n')
            f.write(seq)
            f.write('\n')



        


def get_data_from_BMRB(BMRBid):

    print('Getting experimental data from BMRBid: ', BMRBid)
    experimental_data_tmp = {
        'T1': extract_data_from_BMRB(BMRBid, 'heteronucl_T1_relaxation'),
        'T2': extract_data_from_BMRB(BMRBid, 'heteronucl_T2_relaxation'),
        'hetNOE': extract_data_from_BMRB(BMRBid, 'heteronucl_NOEs')
    }
    
    magnetic_field = {
        'T1': extract_magnetic_field(BMRBid, 'heteronucl_T1_relaxation'),
        'T2': extract_magnetic_field(BMRBid, 'heteronucl_T2_relaxation'),
        'hetNOE': extract_magnetic_field(BMRBid, 'heteronucl_NOEs')
    }

    if not magnetic_field['T1'] ==  magnetic_field['T1'] ==  magnetic_field['T1']:
        print('WARNING: magnetic fields of T1, T2, and hetNOE are not equal')

    #print(magnetic_field['T1'], magnetic_field['T1'], magnetic_field['T1'])
    
    units = {
        'T1': extract_units(BMRBid, 'heteronucl_T1_relaxation'),
        'T2': extract_units(BMRBid, 'heteronucl_T2_relaxation')
    }

    experimental_data = {}
    for residue in experimental_data_tmp['T1']:
        if not residue in experimental_data:
            experimental_data[residue] = {magnetic_field['T1']: {}}
            #print(experimental_data)
            if units['T1'] == 'ms':
                experimental_data_tmp['T1'][residue]['value'] = 0.001 * experimental_data_tmp['T1'][residue]['value']
                experimental_data_tmp['T1'][residue]['error'] = 0.001 * experimental_data_tmp['T1'][residue]['error']
            experimental_data[residue][magnetic_field['T1']]['T1'] = {
                'value' : experimental_data_tmp['T1'][residue]['value'],
                'error' : experimental_data_tmp['T1'][residue]['error'],
            }

    for residue in experimental_data_tmp['T2']:
        if not residue in experimental_data:
            experimental_data[residue] = {magnetic_field['T2']: {}}
        if units['T2'] == 'ms':
            experimental_data_tmp['T2'][residue]['value'] = 0.001 * experimental_data_tmp['T2'][residue]['value']
            experimental_data_tmp['T2'][residue]['error'] = 0.001 * experimental_data_tmp['T2'][residue]['error']
        experimental_data[residue][magnetic_field['T2']]['T2'] = {
            'value' : experimental_data_tmp['T2'][residue]['value'],
            'error' : experimental_data_tmp['T2'][residue]['error'],
        }

    for residue in experimental_data_tmp['hetNOE']:
        if not residue in experimental_data:
            experimental_data[residue] = {magnetic_field['hetNOE']: {}}
        experimental_data[residue][magnetic_field['hetNOE']]['hetNOE'] = {
            'value' : experimental_data_tmp['hetNOE'][residue]['value'],
            'error' : experimental_data_tmp['hetNOE'][residue]['error'],
        }

    exp_data_path = '../../Data/Experiments/spin_relaxation/BMRBid' + BMRBid
    if (not os.path.isdir(exp_data_path)):
        execStr = (f"mkdir {exp_data_path}")
        os.system(execStr)


    experimental_spin_relaxation_times_file = exp_data_path + '/spin_relaxation_times.yaml'
    with open(experimental_spin_relaxation_times_file, 'w') as file:
        yaml.dump(experimental_data, file, sort_keys=True, default_flow_style=False, indent=4)

    print('Experimental data stored in ', experimental_spin_relaxation_times_file)



def extract_magnetic_field(ID, datatype):
    data = []
    x = requests.get("http://api.bmrb.io/v2/entry/" + ID + "?saveframe_category=" + datatype)
    data.append (x.json())

    first_entry = data[0]
    key = next(iter(first_entry))  # e.g., '19993'
    relaxation_list = first_entry[key][datatype][0]
    tags = relaxation_list['tags']
    # Extract values from tags
    spectrometer_freq = None
    t1_units = None

    for tag in tags:
        if tag[0] == 'Spectrometer_frequency_1H':
            spectrometer_freq = float(tag[1])
#        elif tag[0] == 'T1_val_units':
#            t1_units = tag[1]
    
    return float(spectrometer_freq)


def extract_units(ID, datatype):
    # Navigate to the 'tags' list
    data = []
    x = requests.get("http://api.bmrb.io/v2/entry/" + ID + "?saveframe_category=" + datatype)
    data.append (x.json())
    #print(data)
    #frequency, units = extract_spectrometer_info(data)
    #output['frequency'] = frequency
    #output['units'] = units
    #print(output['frequency'],frequency,output['units'],units)

    first_entry = data[0]
    key = next(iter(first_entry))  # e.g., '19993'
    relaxation_list = first_entry[key][datatype][0]
    tags = relaxation_list['tags']
    # Extract values from tags
    spectrometer_freq = None
    t1_units = None

    for tag in tags:
        #print(tag[0])
        if datatype == 'heteronucl_T1_relaxation' and tag[0] == 'T1_val_units':
            return tag[1]
        if datatype == 'heteronucl_T2_relaxation' and tag[0] == 'T2_val_units':
            return tag[1]
        
    return ''
        
    #    return units




# Mapping from 3-letter to 1-letter amino acid codes
AA_THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'SEC': 'U', 'PYL': 'O', 'ASX': 'B', 'GLX': 'Z', 'UNK': 'X'
}

def extract_sequence_from_pdb(pdb_file):
    sequence = []
    seen_residues = set()

    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                res_name = line[17:20].strip()
                chain_id = line[21].strip()
                res_seq = line[22:26].strip()
                uid = (chain_id, res_seq)
                if uid not in seen_residues:
                    seen_residues.add(uid)
                    aa = AA_THREE_TO_ONE.get(res_name.upper(), 'X')
                    sequence.append(aa)
    return ''.join(sequence)


def extract_sequence_from_gro(gro_file):
    sequence = []
    seen_residues = set()

    with open(gro_file, 'r') as file:
        lines = file.readlines()[2:-1]  # Skip title and atom count, last line is box vectors

        for line in lines:
            if len(line) < 8:
                continue  # skip malformed lines
            res_num = line[0:5].strip()
            res_name = line[5:8].strip().upper()
            if res_num not in seen_residues:
                seen_residues.add(res_num)
                aa = AA_THREE_TO_ONE.get(res_name, 'X')
                sequence.append(aa)
    return ''.join(sequence)


def make_fasta(input_file, output_file=None):
    input_path = Path(input_file)
    file_ext = input_path.suffix.lower()

    if file_ext == '.pdb':
        sequence = extract_sequence_from_pdb(input_file)
    elif file_ext == '.gro':
        sequence = extract_sequence_from_gro(input_file)
    else:
        raise ValueError("Unsupported file format. Use .pdb or .gro")

    if not output_file:
        output_file = input_path.with_suffix('.fasta')

    with open(output_file, 'w') as fasta:
        fasta.write(f">{input_path.stem}\n")
        for i in range(0, len(sequence), 80):
            fasta.write(sequence[i:i+80] + "\n")

    return output_file
    
    
# 17.06.2025
# added during NMRLipids meeting in Bergen
# by Tobi R
# FUNCTION TO CALCULATE THE SAXS SCATTERING PROFILE OF THE TRJ
#	needs an external version of crysol 
#		https://www.embl-hamburg.de/biosaxs/crysol.html
#		tested with version: crysol, ATSAS 3.1.3 (r14636)
#	INPUT: 
#		- gro_file; xtc_file: gro file and xtc file (nojump) for MDAnalysis
#	OUTPUT:
#		-averaged SAXS profile (q-space) with standard deviation
#		-pandas dataframe
#		-3 columns (q in 1/A; mean in a.u.; sd in a.u.)

def calculate_SAXS_profile_crysol(gro_file, xtc_file,dt_analysis_ps=1000):
    # Load structure and trajectory / xtc contains inly protein and is no jump
    u = mda.Universe(gro_file, xtc_file)
    
    # Timesteps between analyzed frames (dt_analysis_ps) should be roughly 100 ps
    # 	find out the timestep in the trajectory 
    dt_trj_ps = u.trajectory.dt
    # get ab inzterval for frame analysis
    analysis_frame_interval = round(dt_analysis_ps/dt_trj_ps,0)
    
    # create pandas dataframe which stores the calculated SAXS profiles
    profiles = pd.DataFrame()
    
    # iterate over individul frames
    for ts in u.trajectory:
        # CHECK, IF CONSIDERED - only when frame_idx%analysis_frame_interval = 0
        frame_idx = ts.frame  # Get current frame index
        
        if frame_idx%analysis_frame_interval == 0:
            # PDB OUT
            # write out single PDB files
            filName_PDB = "frame_"+str(frame_idx)+".pdb"
            u.atoms.write(filName_PDB)
            # for some reason crysol expects a different naming of some atoms in pdb
            #	so fox this. I use sed from command line, since this is much faster than 
            #		any pthon tool
            #	use subprocess for that
            # ILE fix
            subprocess.run("sed -i 's/CD  ILE/CD1 ILE/' "+ filName_PDB, shell=True)
            subprocess.run("sed -i 's/HD1 ILE/HD11 ILE/'" + filName_PDB, shell=True)
            subprocess.run("sed -i 's/HD2 ILE/HD12 ILE/'"+ filName_PDB, shell=True)
            subprocess.run("sed -i 's/HD3 ILE/HD13 ILE/'"+ filName_PDB, shell=True)
            # TERMINI fix
            subprocess.run("sed -i 's/OC1/OXT/' "+ filName_PDB, shell=True)
            subprocess.run("sed -i 's/OT1/OXT/' "+ filName_PDB, shell=True)
            subprocess.run("sed -i '/.*OC2*/d' "+ filName_PDB, shell=True)
            subprocess.run("sed -i '/.*OT2*/d' "+ filName_PDB, shell=True)
            
            # RUN CRYSOL
            OUT = subprocess.run("crysol "+ filName_PDB +" -lm 50 -fb 18 -ns 101 -p profile_"+str(frame_idx), shell=True)
            
            # READ SAXS PROFILE FILE
            data = np.loadtxt("profile_"+str(frame_idx)+".abs",skiprows=1)
            tab = pd.DataFrame(data)
            tab.columns = ["q","Inten"]
            
            # SAVE FILES IN DATAFRAME
            prof_I = tab["Inten"]
            profiles = pd.concat([profiles,prof_I],axis=1)
            
            # CLEAN UP 
            os.remove("frame_"+str(frame_idx)+".pdb")
            os.remove("profile_"+str(frame_idx)+".alm")
            os.remove("profile_"+str(frame_idx)+".log")
            os.remove("profile_"+str(frame_idx)+".int")
            os.remove("profile_"+str(frame_idx)+".abs")
    
    # GET AVERAGE AND SD SAXS PROFILE 
    # calculate the mean profile and the corresponding sd
    profile_mean = profiles.mean(axis=1)
    profile_sd = profiles.std(axis=1)
    
    print(profiles)
    
    # merge data and set column names
    res = pd.DataFrame([tab["q"],profile_mean,profile_sd]).transpose()
    res.columns = ["q[1/A]","mean_Inten[a.u.]","sd_Inten[a.u.]"]
    
    return(res)


# 18.06.2025
# added during NMRLipids meeting in Bergen
# by Tobi R
# FUNCTION TO CALCULATE THE SAXS SCATTERING PROFILE OF THE TRJ
#	needs the python package MAICoS
#		https://gitlab.com/maicos-devel/maicos
#		https://maicos.readthedocs.io/en/main/analysis-modules/saxs.html
#	INPUT: 
#		- gro_file; xtc_file: gro file and xtc file (nojump) for MDAnalysis
#	OUTPUT:
#		-SAXS profile (q-space) of the trajectory
#		-pandas dataframe
#		-2 columns (q in 1/A; mean in a.u.)

def calculate_SAXS_profile_maicos(gro_file, xtc_file):
    # Load structure and trajectory / xtc contains inly protein and is no jump
    u = mda.Universe(gro_file, xtc_file)
    
    # run the program:
    #	-since we use nojump simulations, we can set unwrap to false:
    #	- since most scattering profiles of IDPs are not larger than q=0.5, adjust:
    #		- qmax = 0.5 / qmin = 0 / dq = 0.05
    #	-see MAICoS documentation: https://maicos.readthedocs.io/en/main/analysis-modules/saxs.html
    # create the object
    SAXS = maicos.Saxs(atomgroup=u.atoms,unwrap=False,qmin=0,qmax=0.5005,dq=0.0025)
    
    # run the analzsis
    profile = SAXS.run()
    
    # extract the profile:
    scattering_vectors = profile.results.scattering_vectors
    intensity = profile.results.scattering_intensities
    
    #merge data for return and set column names
    res = pd.DataFrame([scattering_vectors,intensity]).transpose()
    res.columns = ["q[1/A]","mean_Inten[a.u.]"]

    return(res)
