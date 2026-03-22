import os
import subprocess
import numpy as np
import pandas as pd
import MDAnalysis as mda
import mdtraj
from MDAnalysis.analysis import distances
from scipy import optimize
import requests
from pathlib import Path
import yaml
import requests
import re
import maicos
from typing import List, Dict, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import math

import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


from pymol import cmd
import pymol
import glob

import matplotlib.image as mpimg


def calculate_contact_probabilities(gro_file, xtc_file, cutoff):
    u = mda.Universe(gro_file, xtc_file)
    CAatoms = u.select_atoms("name CA")
    num_residues = len(CAatoms)

    within_cutoff_count = np.zeros((num_residues, num_residues))
    num_frames = 0

    for ts in u.trajectory:
        distances_array = distances.distance_array(CAatoms.positions, CAatoms.positions)
        within_cutoff_count += distances_array <= cutoff
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
    color_map = plt.imshow(
        probabilities_df.values, cmap="jet", vmin=0, vmax=1, origin="lower"
    )

    cbar = plt.colorbar(color_map)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(
        "Probability of contact (≤ 15.0 Å)", rotation=270, labelpad=25, fontsize=18
    )

    plt.tick_params(axis="both", which="major", labelsize=12)
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

    color_map = plt.imshow(
        probabilities_df.values, vmin=vmin, vmax=vmax, cmap="jet", origin="lower"
    )

    cbar = plt.colorbar(color_map)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("Distance (Å)", rotation=270, labelpad=25, fontsize=18)

    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.xlabel("Residue", fontsize=16)
    plt.ylabel("Residue", fontsize=16)

    plt.tight_layout()

    plt.savefig(output_file, dpi=600)
    plt.close()
    print(f"Distance map saved as {output_file}.")


def calculate_backbone_correlations(gro_file, xtc_file, output_file):

    outfile_png = output_file
    outfile_csv = output_file.replace(".png", ".csv")

    u = mda.Universe(gro_file, xtc_file)
    CAatoms = u.select_atoms("name CA")

    num_atoms = len(CAatoms.ix)
    num_vectors = num_atoms - 1
    vec = np.zeros((num_vectors, 3))
    matrix = np.zeros((num_vectors, num_vectors))

    for frame in u.trajectory:
        for i in range(num_vectors):
            vec[i] = CAatoms[i + 1].position - CAatoms[i].position
            vec[i] /= np.linalg.norm(vec[i])
        matrix += np.dot(vec, vec.T)
    matrix /= len(u.trajectory)

    np.savetxt(outfile_csv, matrix, delimiter=",")

    color_map = plt.imshow(matrix, vmin=-1, vmax=1, cmap="seismic", origin="lower")

    cbar = plt.colorbar(color_map)
    cbar.ax.tick_params(labelsize=16)

    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.xlabel(r"C$_{\alpha}$ Pair Index", fontsize=16)
    plt.ylabel(r"C$_{\alpha}$ Pair Index", fontsize=16)

    plt.tight_layout()

    plt.savefig(outfile_png, dpi=600)
    plt.close("all")


def calculate_backbone_NH_correlation_functions(
    gro_file, trj_fname, top_fname, sim_folder
):
    NHindex_file = sim_folder + "NH.ndx"
    correlation_function_folder = sim_folder + "/correlation_functions/"
    if not os.path.isdir(correlation_function_folder):
        execStr = f"mkdir {correlation_function_folder}"
    try:
        os.system(execStr)
    except:
        pass
    #print(gro_file, NHindex_file)
    resids = make_index_file_for_backbone_nh_pairs(gro_file, NHindex_file)
    list_of_correlation_functions = []
    for resname in resids:
        correlation_function = (
            sim_folder + "/correlation_functions/NHrotaCF_" + resname + ".xvg"
        )
        list_of_correlation_functions.append(correlation_function)
        if not os.path.isfile(correlation_function):
            execStr = f"echo {resids[resname]} | gmx rotacf -f {trj_fname} -s {top_fname} -n {NHindex_file} -o {correlation_function} -P 2 -d -xvg none"
            os.system(execStr)
    return list_of_correlation_functions


def make_index_file_for_backbone_nh_pairs(gro_file, output_ndx):
    # Load the structure
    u = mda.Universe(gro_file)

    # Ensure hydrogens are present
    if not any(atom.name == "H" for atom in u.atoms) and not any(atom.name == "HN" for atom in u.atoms):
        raise ValueError(
            "Hydrogen atoms not found. Ensure the gro file includes hydrogens."
        )

    resids_with_nh = {}

    with open(output_ndx, "w") as ndx:
        pair_count = 0


        for res in u.select_atoms("protein").residues:
            try:
                N_atom = res.atoms.select_atoms("name N")[0]
                H_atom = res.atoms.select_atoms("name H")[0]
            except IndexError:
                #pass #continue  # Skip residues without N or H (e.g., termini or incomplete)
                try:
                    N_atom = res.atoms.select_atoms("name N")[0]
                    H_atom = res.atoms.select_atoms("name HN")[0]
                except IndexError:
                    continue  # Skip residues without N or H (e.g., termini or incomplete)

            resid = res.resid
            resname = str(res.resid) + res.resname
            resids_with_nh[resname] = pair_count  # .append(resname)

            # group_name = f"N_H_pair_{pair_count}"
            ndx.write(f"[ {resname} ]\n")
            ndx.write(
                f"{N_atom.index + 1} {H_atom.index + 1}\n\n"
            )  # GROMACS uses 1-based indexing
            pair_count += 1

    return resids_with_nh


def read_correlation_function(input_data):
    # for reading the correlation function data
    opf = open(input_data, "r")
    lines = opf.readlines()
    data_times = []
    data_F = []
    for i, line in enumerate(lines):
        if "#" in line:
            continue
        if "&" in line:
            continue
        if "@" in line:
            continue
        if "label" in line:
            continue
        if line == "":
            continue
        parts = line.split()
        if np.shape(parts)[0] == 2:
            try:
                data_F.append(float(parts[1]))
                data_times.append(float(parts[0]))
            except:
                print(i)
                break

    data_Fout = np.array(data_F)
    times_out = np.array(data_times)
    return data_Fout, times_out


def calc_relax_time(
    correlation_function,
    magnetic_field,
    biggest_corr_time,
    smallest_corr_time,
    N_exp_to_fit,
):

    # normalized correlation fuction
    ## NcorrF = (self.org_corrF - self.OP ** 2) / (1 - self.OP ** 2);

    correlation_function_data = read_correlation_function(correlation_function)

    # Create correlation times from the times and number of exponential specified by the user
    step_exp = (biggest_corr_time - smallest_corr_time) / N_exp_to_fit
    Ctimes = 10 ** np.arange(smallest_corr_time, biggest_corr_time, step_exp)

    # First, no forcing the plateou
    # create exponential functions and put them into a matrix, individual exponentials in columns
    # the lengthe of correlationd data to be used is specified by the user
    NcorrF, times_out = correlation_function_data
    n = len(times_out)
    m = len(Ctimes)
    Cexp_mat = np.zeros((n, m))

    for i in range(0, n):
        for j in range(0, m):
            Cexp_mat[i, j] = np.exp(-times_out[i] / Ctimes[j])

    # least square solution
    Coeffs, res = optimize.nnls(Cexp_mat, NcorrF[0:n])

    # Effective correlation time from components, in units of sec

    Teff = sum(Coeffs * Ctimes * 0.001 * 10 ** (-9))

    # calculate t_eff from area
    dt = times_out[2] - times_out[1]
    pos = np.argmax(NcorrF[0:n] < 0)

    if pos > 0:
        tau_eff_area = sum(NcorrF[0:pos]) * dt * 0.001 * 10 ** (-9)
        conv = 1
    else:
        tau_eff_area = sum(NcorrF[0:n]) * dt * 0.001 * 10 ** (-9)
        conv = 0

    # changin the unit of time permanently
    Ctimes = Ctimes * 0.001 * 10 ** (-9)
    # self.Coeffs=Coeffs
    # self.Ctimes=Ctimes
    Ctimes_ns = Ctimes * 10 ** (9)

    # Calculate the relaxation times for chosen nuclei
    T1, T2, NOE = get_relaxation_N(magnetic_field, Coeffs, Ctimes)

    # get the reconstucted correlation function
    # self.rec_corrF=Cexp_mat.dot(Coeffs)
    # self.plot_fit(self.rec_corrF)
    # self.plot_exp_hist(Ctimes,Coeffs)

    return Teff, tau_eff_area, T1, T2, NOE, Coeffs, Ctimes_ns


def calc_dynamic_landscape(
    correlation_function, biggest_corr_time, smallest_corr_time, N_exp_to_fit
):

    # normalized correlation fuction
    ## NcorrF = (self.org_corrF - self.OP ** 2) / (1 - self.OP ** 2);

    correlation_function_data = read_correlation_function(correlation_function)

    # Create correlation times from the times and number of exponential specified by the user
    step_exp = (biggest_corr_time - smallest_corr_time) / N_exp_to_fit
    Ctimes = 10 ** np.arange(smallest_corr_time, biggest_corr_time, step_exp)

    # First, no forcing the plateou
    # create exponential functions and put them into a matrix, individual exponentials in columns
    # the lengthe of correlationd data to be used is specified by the user
    NcorrF, times_out = correlation_function_data
    n = len(times_out)
    m = len(Ctimes)
    Cexp_mat = np.zeros((n, m))

    for i in range(0, n):
        for j in range(0, m):
            Cexp_mat[i, j] = np.exp(-times_out[i] / Ctimes[j])

    # least square solution
    Coeffs, res = optimize.nnls(Cexp_mat, NcorrF[0:n])

    # Effective correlation time from components, in units of sec

    Teff = sum(Coeffs * Ctimes * 0.001 * 10 ** (-9))

    # calculate t_eff from area
    dt = times_out[2] - times_out[1]
    pos = np.argmax(NcorrF[0:n] < 0)

    if pos > 0:
        tau_eff_area = sum(NcorrF[0:pos]) * dt * 0.001 * 10 ** (-9)
        conv = 1
    else:
        tau_eff_area = sum(NcorrF[0:n]) * dt * 0.001 * 10 ** (-9)
        conv = 0

    # changin the unit of time permanently
    Ctimes = Ctimes * 0.001 * 10 ** (-9)
    # self.Coeffs=Coeffs
    # self.Ctimes=Ctimes
    Ctimes_ns = Ctimes * 10 ** (9)

    return Teff, tau_eff_area, Coeffs, Ctimes_ns


def get_relaxation_N(magnetic_field, Coeffs, Ctimes):

    gammaH = 267.513 * 10**6
    gammaN = -27.166 * 10**6

    # wh = gammaH * magnetic_field
    wh = magnetic_field * 10**6 * 2 * np.pi
    wn = wh * gammaN / gammaH

    # initiate spectral densities
    J0 = 0
    JhMn = 0
    JhPn = 0
    Jh = 0
    Jn = 0

    m = len(Ctimes)
    CtimesSeconds = np.zeros(m)
    for i in range(0, m):
        CtimesSeconds[i] = Ctimes[i] * 10 ** (-9)
        # print(Ctimes[i],Coeffs[i])
        w = 0

        J0 = J0 + 2 * Coeffs[i] * CtimesSeconds[i] / (1.0 + w * w * CtimesSeconds[i] * CtimesSeconds[i])

        w = wh - wn
        JhMn = JhMn + 2 * Coeffs[i] * CtimesSeconds[i] / (1.0 + w * w * CtimesSeconds[i] * CtimesSeconds[i])

        w = wn
        Jn = Jn + 2 * Coeffs[i] * CtimesSeconds[i] / (1.0 + w * w * CtimesSeconds[i] * CtimesSeconds[i])

        w = wh
        Jh = Jh + 2 * Coeffs[i] * CtimesSeconds[i] / (1.0 + w * w * CtimesSeconds[i] * CtimesSeconds[i])

        w = wn + wh
        JhPn = JhPn + 2 * Coeffs[i] * CtimesSeconds[i] / (1.0 + w * w * CtimesSeconds[i] * CtimesSeconds[i])

    mu = 4 * np.pi * 10 ** (-7)  # magnetic constant of vacuum permeability
    h_planck = 1.055 * 10 ** (-34)
    # reduced Planck constant
    rN = 0.101 * 10 ** (-9)
    # average cubic length of N-H bond
    d = 1 * (mu * gammaN * gammaH * h_planck) / (4 * np.pi * rN**3)
    # dipolar coupling constant

    # units were corrected by S.Ollila and E.Mantzari, removed 2*pi from R1 and R2
    R1 = (d**2 / 20) * (1 * JhMn + 3 * Jn + 6 * JhPn) + Jn * (
        wn * 160 * 10 ** (-6)
    ) ** 2 / 15
    R2 = 0.5 * (d**2 / 20) * (4 * J0 + 3 * Jn + 1 * JhMn + 6 * Jh + 6 * JhPn) + (
        wn * 160 * 10 ** (-6)
    ) ** 2 / 90 * (4 * J0 + 3 * Jn)
    NOE = 1 + (d**2 / 20) * (6 * JhPn - 1 * JhMn) * gammaH / (gammaN * R1)

    # print("T1: {}, T2: {}, NOE: {}".format(1/R1, 1/R2, NOE))

    return 1 / R1, 1 / R2, NOE


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
            for noe in entry_content.get("heteronucl_NOEs", []):
                for loop in noe.get("loops", []):
                    if loop.get("category") == "_Heteronucl_NOE":
                        return loop.get("data", [])
    return []


def extract_data_from_BMRB(ID, datatype):

    output = {}
    data = []

    x = requests.get(
      "http://api.bmrb.io/v2/entry/" + ID + "?saveframe_category=" + datatype
    )
    data.append (x.json())
    for entry in data:
        for entry_id, entry_content in entry.items():
            for noe in entry_content.get(datatype, []): # this line gives error in 2 BMRB entries. ex. AttributeError: 'str' object has no attribute 'get'"
                for loop in noe.get('loops', []):


                    if (
                        datatype == "heteronucl_NOEs"
                        and loop.get("category") == "_Heteronucl_NOE"
                    ):
                        rel_data = loop.get("data", [])
                        for i in rel_data:
                            residue = i[5] + i[6]
                            # Parse the “value” field (should always be numeric), but handle errors just in case
                            try: # Add error handling

                                value = float(i[19])
                            except (ValueError, TypeError):
                                value = None
                            # Now safely parse the “error” field
                            try:
                                error_value = float(i[20])
                            except (ValueError, TypeError):
                                # '.' or other non-numeric → treat as missing
                                error_value = None # or 0.0 or 'NA' depending on how you want to handle it.... not sure what is best 

                            output[residue] = {

                                'value': value,
                                'error': error_value
                            }


                        return output

                    if (
                        datatype == "heteronucl_T1_relaxation"
                        and loop.get("category") == "_T1"
                    ):
                        rel_data = loop.get("data", [])
                        for i in rel_data:
                            residue = i[5] + i[6]
                            # The value may have entires of '.'
                            try:
                                value = float(i[10])
                            except (ValueError, TypeError):
                                value = None
                            # Now safely parse the “error” field
                            try:
                                error_value = float(i[11])
                            except (ValueError, TypeError):
                                error_value = None

                            output[residue] = {

                                'value': value,
                                'error': error_value
                                }

                        return output

                    if (
                        datatype == "heteronucl_T2_relaxation"
                        and loop.get("category") == "_T2"
                    ):
                        rel_data = loop.get("data", [])
                        for i in rel_data:
                            residue = i[5] + i[6]
                            # The value may have entires of '.'
                            try:
                                value = float(i[10])
                            except (ValueError, TypeError):
                                value = None
                            # Now safely parse the “error” field
                            try:
                                error_value = float(i[11])
                            except (ValueError, TypeError):
                                error_value = None
                            output[residue] = {

                                'value': value,
                                'error': error_value
                                }

                        return output

    return []


def get_spin_relaxation_conditions_from_BMRB(BMRBid):
    data = []
    x = requests.get("http://api.bmrb.io/v2/entry/" + BMRBid + "/experiments")
    data.append(x.json())

    exp_data_path = "../../Data/Experiments/spin_relaxation/BMRBid" + BMRBid
    if not os.path.isdir(exp_data_path):
        execStr = f"mkdir {exp_data_path}"
        os.system(execStr)


    T1names = ['Het. Nuc. T1 relaxation', '2D 1H-15N HSQC-T1', 'T1/R1 relaxation', '15N R1', 'sqct1etf3gpsitc3d', 'Het. Nuc. T1 relaxation', '2D 1H-15N HSQC-T1', '15N T1', 'hsqct1etf3gpsitc3d.nlf', 'T1_relaxation_800', 'R1-measurement', 'T1', '3D 1H-15N t1 interleaved', '2D 15N HSQC T1', '2D 1H-15N R1 relaxation', 'T2 (H[n[T2(N)]])', '2D 1H-15N HSQC T1', '2D 1H-15N HSQC R1', '2D 15N-T1', '15N T1 relaxation', '15N T1 experiment', '"T1, T2, NOE"', '2D 1H-15N T1-HSQC', 'T1 experiments', '2D 1H-15N HSQC R1 edited', '2D R1 15','1H correlation']
    T2names = ['Het. Nuc. T2 relaxation', '2D 1H-15N HSQC -T2', 'T2/R2 relaxation', '15N R2', 'hsqct2etf3gpsitc3d', 'Het. Nuc. T2 relaxation', '2D 1H-15N HSQC -T2', '15N T2', 'hsqct2etf3gpsitc3d.ac', 'T2_relaxation_800', 'R2 measurement', 'T2', '3D 1H-15N t2 interleaved', '2D 15N HSQC T2', '2D 1H-15N R2 relaxation', 'T1 (H[n[T1(N)]])', '2D 1H-15N HSQC T2', '2D 1H-15N HSQC R2', '2D 15N-T2', '15N T2 relaxation', '15N T2 experiment', '"T1, T2, NOE"', '2D 1H-15N T2-HSQC', 'T2 experiments', '2D 1H-15N HSQC R2 edited', '2D R2 15N', '1H correlation']
    netNOEnames = ['15N-(1H) NOE', '2D 1H-15N HSQC-NOE','15N-(1H) NOE', 'hsqcnoef3gpsi', '1H 15N het NOE', '1H-15N heteronoe', '2D 1H-15N HSQC-NOE', '{1H}-15N NOE', 'HetNOE_relaxation_800', 'hetNOE measurement', 'HTNOE 1', '2D 1H-15N NOE with saturation', 'HETERONOE', '2D 15N HSQC Heteronuclear NOE', '2D 1H-15N heteronuclear NOE', '"1H,15N NOE"', '2D 1H-15N HSQC NOE', '2D 1H-15N HSQC hNOE', '2D 15N-HET-NOE', '2D 15N- HET-NOE', 'heteronuclear 1H-15N NOE', '"T1, T2, NOE"', '15N-1H NOE', '2D 1H-15N Het NOE', 'Heteronuclear NOE ratio', '2D NOE 15N', '1H correlation']
    

    for i in data[0]:
        print(i["Name"])
        if i["Name"] in T1names:
            metadata_file = exp_data_path + "/T1_metadata.yaml"
            with open(metadata_file, "w") as file:
                yaml.dump(i, file, sort_keys=True, default_flow_style=False, indent=4)
            print("T1 metadata stored in ", metadata_file)

        if i["Name"] in T2names:
            metadata_file = exp_data_path + "/T2_metadata.yaml"
            with open(metadata_file, "w") as file:
                yaml.dump(i, file, sort_keys=True, default_flow_style=False, indent=4)
            print("T2 metadata stored in ", metadata_file)

        if i["Name"] in netNOEnames:
            metadata_file = exp_data_path + "/hetNOE_metadata.yaml"
            with open(metadata_file, "w") as file:
                yaml.dump(i, file, sort_keys=True, default_flow_style=False, indent=4)
            print("hetNOE metadata stored in ", metadata_file)


#    print(data[0]['Het. Nuc. T1 relaxation'])


def get_chemical_shift_conditions_from_BMRB(BMRBid,chemical_shift_experiment):
    data = []
    x = requests.get("http://api.bmrb.io/v2/entry/" + BMRBid + "/experiments")
    data.append(x.json())

    exp_data_path = "../../Data/Experiments/chemical_shift/BMRBid" + BMRBid
    if not os.path.isdir(exp_data_path):
        execStr = f"mkdir {exp_data_path}"
        os.system(execStr)

    for i in data[0]:
        print(i["Name"], chemical_shift_experiment)
        if i["Name"] == chemical_shift_experiment:
            metadata_file = exp_data_path + "/chemical_shift_metadata.yaml"
            with open(metadata_file, "w") as file:
                yaml.dump(i, file, sort_keys=True, default_flow_style=False, indent=4)
            print("Chemical shift metadata stored in ", metadata_file)


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
    match = re.search(
        r"loop_.*?_Entity_poly_seq.*?(?=loop_|stop_)", star_text, re.DOTALL
    )
    if not match:
        raise Exception("Entity_poly_seq section not found in the NMR-STAR file")

    loop_block = match.group(0)
    lines = loop_block.strip().splitlines()

    # Only include lines that look like sequence rows (starting with an integer)
    data_lines = [line for line in lines if line and re.match(r"^\d+", line)]

    residues = []
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 3:
            residues.append(parts[2].upper())

    # 3-letter to 1-letter amino acid code conversion
    aa_dict = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
        "SEC": "U",
        "PYL": "O",
        "ASX": "B",
        "GLX": "Z",
        "UNK": "X",
    }

    sequence = ""
    for res in residues:
        aa = aa_dict.get(res)
        if aa:
            sequence += aa
        else:
            print(f"Warning: Unrecognized residue '{res}' — replaced with 'X'")
            sequence += "X"

    fasta = f">BMRB_{entry_id}\n{sequence}"
    return fasta


def extract_fasta_from_bmrb(bmrb_id: str, experiment_type) -> List[str]:
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
    clean_id = bmrb_id.lower().replace("bmr", "")

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

        exp_data_path = "../../Data/Experiments/" + experiment_type + "/BMRBid" + bmrb_id
        if not os.path.isdir(exp_data_path):
            execStr = f"mkdir {exp_data_path}"
            os.system(execStr)

        fasta_file = exp_data_path + "/fasta.yaml"
        sequence_dict = {"sequence": sequences}
        with open(fasta_file, "w") as file:
            yaml.dump(
                sequence_dict, file, sort_keys=True, default_flow_style=False, indent=4
            )

        print("Sequence stored in ", fasta_file)

        return sequences

    except requests.RequestException as e:
        raise requests.RequestException(
            f"Failed to fetch BMRB entry {bmrb_id}: {str(e)}"
        )


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
    entity_pattern = r"_Entity\.ID\s+(\d+).*?_Entity\.Polymer_seq_one_letter_code\s*\n([^_\n]*(?:\n[^_\n]*)*)"

    matches = re.findall(entity_pattern, star_data, re.DOTALL | re.IGNORECASE)

    for entity_id, seq_block in matches:
        # Clean up the sequence - remove whitespace, quotes, and newlines
        sequence = re.sub(r'[\'"\s\n\r]', "", seq_block.strip())

        # Remove any remaining non-amino acid characters (except standard amino acids)
        sequence = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", sequence.upper())

        if (
            sequence and len(sequence) > 5
        ):  # Only include sequences with reasonable length
            # Get entity name if available
            entity_name_pattern = (
                rf"_Entity\.ID\s+{entity_id}.*?_Entity\.Name\s*\n([^\n]*)"
            )
            name_match = re.search(
                entity_name_pattern, star_data, re.DOTALL | re.IGNORECASE
            )

            entity_name = ""
            if name_match:
                entity_name = name_match.group(1).strip().strip("'\"")

            # Create FASTA header
            header = f">BMRB:{bmrb_id}_Entity:{entity_id}"
            if entity_name:
                header += f" {entity_name}"

            # Format sequence with 80 characters per line (standard FASTA format)
            #formatted_seq = "\n".join(
            #    [sequence[i : i + 80] for i in range(0, len(sequence), 80)]
            #)

            # fasta_entry = f"{header}\n{formatted_seq}"
            #fasta_entry = f"{formatted_seq}"
            fasta_entry = f"{sequence}"
            sequences.append(fasta_entry)

    # Alternative pattern for older BMRB formats
    if not sequences:
        alt_pattern = r"_Entity_poly_seq\.Residue_label\s+(\d+)\s+(\d+)\s+([A-Z]{1,3})"
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
                sequence = "".join(residues)
                if len(sequence) > 5:
                    header = f">BMRB:{bmrb_id}_Entity:{entity_id}"
                    #formatted_seq = "\n".join(
                    #    [sequence[i : i + 80] for i in range(0, len(sequence), 80)]
                    #)
                    #fasta_entry = f"{header}\n{formatted_seq}"
                    fasta_entry = f"{sequence}"
                    sequences.append(fasta_entry)

    return sequences


def _three_to_one_letter(three_letter: str) -> Optional[str]:
    """Convert 3-letter amino acid code to 1-letter code."""

    aa_dict = {
        "ALA": "A",
        "CYS": "C",
        "ASP": "D",
        "GLU": "E",
        "PHE": "F",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LYS": "K",
        "LEU": "L",
        "MET": "M",
        "ASN": "N",
        "PRO": "P",
        "GLN": "Q",
        "ARG": "R",
        "SER": "S",
        "THR": "T",
        "VAL": "V",
        "TRP": "W",
        "TYR": "Y",
    }

    return aa_dict.get(three_letter.upper())


def save_fasta_sequences(sequences: List[str], filename: str) -> None:
    """
    Save FASTA sequences to a file.

    Args:
        sequences (List[str]): List of FASTA formatted sequences
        filename (str): Output filename
    """

    with open(filename, "w") as f:
        for i, seq in enumerate(sequences):
            if i > 0:
                f.write("\n")
            f.write(seq)
            f.write("\n")


def get_spin_relaxations_from_BMRB(BMRBid):

    print("Getting experimental data from BMRBid: ", BMRBid)
    experimental_data_tmp = {
        "T1": extract_data_from_BMRB(BMRBid, "heteronucl_T1_relaxation"),
        "T2": extract_data_from_BMRB(BMRBid, "heteronucl_T2_relaxation"),
        "hetNOE": extract_data_from_BMRB(BMRBid, "heteronucl_NOEs"),
    }

    magnetic_field = {
        "T1": extract_magnetic_field(BMRBid, "heteronucl_T1_relaxation"),
        "T2": extract_magnetic_field(BMRBid, "heteronucl_T2_relaxation"),
        "hetNOE": extract_magnetic_field(BMRBid, "heteronucl_NOEs"),
    }

    if not magnetic_field["T1"] == magnetic_field["T1"] == magnetic_field["T1"]:
        print("WARNING: magnetic fields of T1, T2, and hetNOE are not equal")

    # print(magnetic_field['T1'], magnetic_field['T1'], magnetic_field['T1'])

    units = {
        "T1": extract_units(BMRBid, "heteronucl_T1_relaxation"),
        "T2": extract_units(BMRBid, "heteronucl_T2_relaxation"),
    }

    experimental_data = {}
    for residue in experimental_data_tmp["T1"]:
        if not residue in experimental_data:

            experimental_data[residue] = {magnetic_field['T1']: {}}
            #print(experimental_data)
            if units['T1'] == 'ms':
                #Handle errors for value and error if missing in data
                if experimental_data_tmp['T1'][residue]['value'] is not None:
                    experimental_data_tmp['T1'][residue]['value'] *= 0.001
                else:
                    print(f"[WARNING] 'value' is None for residue {residue} in T1 data ({BMRBid})")
                    experimental_data_tmp['T1'][residue]['value'] = None  # or 0.0, or skip?
                
                if experimental_data_tmp['T1'][residue]['error'] is not None:
                    experimental_data_tmp['T1'][residue]['error'] *= 0.001
                else:
                    print(f"[WARNING] 'error' is None for residue {residue} in T1 data ({BMRBid})")
                    experimental_data_tmp['T1'][residue]['error'] = None  # or 0.0, or skip?

            experimental_data[residue][magnetic_field['T1']]['T1'] = {
                'value' : experimental_data_tmp['T1'][residue]['value'],
                'error' : experimental_data_tmp['T1'][residue]['error'],
            }


    for residue in experimental_data_tmp['T2']:
        if residue not in experimental_data:
            experimental_data[residue] = {}

        if magnetic_field['T2'] not in experimental_data[residue]:
            experimental_data[residue][magnetic_field['T2']] = {}

        if units['T2'] == 'ms':
            #Handle errors for value and error if missing in data
            if experimental_data_tmp['T2'][residue]['value'] is not None:
                experimental_data_tmp['T2'][residue]['value'] *= 0.001
            else:
                print(f"[WARNING] 'value' is None for residue {residue} in T2 data ({BMRBid})")
                experimental_data_tmp['T2'][residue]['value'] = None  # or 0.0, or skip?
            if experimental_data_tmp['T2'][residue]['error'] is not None:
                experimental_data_tmp['T2'][residue]['error'] *= 0.001
            else:
                print(f"[WARNING] 'error' is None for residue {residue} in T2 data ({BMRBid})")
                experimental_data_tmp['T2'][residue]['error'] = None  # or 0.0, or skip?

        experimental_data[residue][magnetic_field['T2']]['T2'] = {
            'value': experimental_data_tmp['T2'][residue]['value'],
            'error': experimental_data_tmp['T2'][residue]['error'],
        }


    for residue in experimental_data_tmp['hetNOE']:
        if residue not in experimental_data:
            experimental_data[residue] = {}

        if magnetic_field['hetNOE'] not in experimental_data[residue]:
            experimental_data[residue][magnetic_field['hetNOE']] = {}

        experimental_data[residue][magnetic_field['hetNOE']]['hetNOE'] = {
            'value': experimental_data_tmp['hetNOE'][residue]['value'],
            'error': experimental_data_tmp['hetNOE'][residue]['error'],
        }


    exp_data_path = '../../Data/Experiments/spin_relaxation/BMRBid' + BMRBid
    if (not os.path.isdir(exp_data_path)):
        execStr = (f"mkdir {exp_data_path}")

        os.system(execStr)

    experimental_spin_relaxation_times_file = (
        exp_data_path + "/spin_relaxation_times.yaml"
    )
    with open(experimental_spin_relaxation_times_file, "w") as file:
        yaml.dump(
            experimental_data, file, sort_keys=True, default_flow_style=False, indent=4
        )

    print("Experimental data stored in ", experimental_spin_relaxation_times_file)


def extract_magnetic_field(ID, datatype):
    data = []

    #Include error handling so that if the data does not exist the code does not fail with relaxation_list = first_entry[key][datatype][0]\nIndexError: list index out of range
    try:
        x = requests.get("http://api.bmrb.io/v2/entry/" + ID + "?saveframe_category=" + datatype)
        if x.status_code != 200: #if HTTP status code is not "OK"
            print(f"[WARNING] Request failed for BMRB {ID} with status code {x.status_code}")
            return None
        data.append(x.json())

        first_entry = data[0]
        key = next(iter(first_entry), None)  # e.g., '19993'
        if not key:
            print(f"[WARNING] No top-level key in JSON for BMRB {ID}")
            return None

        if datatype not in first_entry[key] or not first_entry[key][datatype]:
            print(f"[WARNING] No data found for datatype '{datatype}' in BMRB {ID}")
            return None
        relaxation_list = first_entry[key][datatype][0]
        tags = relaxation_list['tags']
        # Extract values from tags
        spectrometer_freq = None
        t1_units = None

        for tag in tags:
            if tag[0] == 'Spectrometer_frequency_1H':
                try:
                    spectrometer_freq = float(tag[1])
                except (ValueError, TypeError):
                    print(f"[WARNING] Invalid spectrometer frequency in BMRB {ID}")
                    return None

        if spectrometer_freq is None:
            print(f"[WARNING] Spectrometer_frequency_1H not found in BMRB {ID}")
            return None

        return float(spectrometer_freq)

    except Exception as e:
        print(f"[ERROR] Failed to extract magnetic field for BMRB {ID}: {e}")
        return None
    


def extract_units(ID, datatype):
    # Navigate to the 'tags' list
    data = []
    x = requests.get(
        "http://api.bmrb.io/v2/entry/" + ID + "?saveframe_category=" + datatype
    )
    data.append(x.json())
    # print(data)
    # frequency, units = extract_spectrometer_info(data)
    # output['frequency'] = frequency
    # output['units'] = units
    # print(output['frequency'],frequency,output['units'],units)

    first_entry = data[0]
    key = next(iter(first_entry))  # e.g., '19993'
    relaxation_list = first_entry[key][datatype][0]
    tags = relaxation_list["tags"]
    # Extract values from tags
    spectrometer_freq = None
    t1_units = None

    for tag in tags:
        # print(tag[0])
        if datatype == "heteronucl_T1_relaxation" and tag[0] == "T1_val_units":
            return tag[1]
        if datatype == "heteronucl_T2_relaxation" and tag[0] == "T2_val_units":
            return tag[1]

    return ""

    #    return units


# Mapping from 3-letter to 1-letter amino acid codes
AA_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    "ASX": "B",
    "GLX": "Z",
    "UNK": "X",
}


def extract_sequence_from_pdb(pdb_file):
    sequence = []
    seen_residues = set()

    with open(pdb_file, "r") as file:
        for line in file:
            if line.startswith("ATOM"):
                res_name = line[17:20].strip()
                chain_id = line[21].strip()
                res_seq = line[22:26].strip()
                uid = (chain_id, res_seq)
                if uid not in seen_residues:
                    seen_residues.add(uid)
                    aa = AA_THREE_TO_ONE.get(res_name.upper(), "X")
                    sequence.append(aa)
    return "".join(sequence)



def extract_sequence_from_gro(gro_file):
    sequence = []
    seen_residues = set()

    with open(gro_file, "r") as file:
        lines = file.readlines()[
            2:-1
        ]  # Skip title and atom count, last line is box vectors

        for line in lines:
            if len(line) < 8:
                continue  # skip malformed lines
            res_num = line[0:5].strip()
            res_name = line[5:8].strip().upper()
            if res_num not in seen_residues:
                seen_residues.add(res_num)
                aa = AA_THREE_TO_ONE.get(res_name, "X")
                sequence.append(aa)
    return "".join(sequence)



def make_fasta(input_file, output_file=None):
    input_path = Path(input_file)
    file_ext = input_path.suffix.lower()

    if file_ext == ".pdb":
        sequence = extract_sequence_from_pdb(input_file)
    elif file_ext == ".gro":
        sequence = extract_sequence_from_gro(input_file)
    else:
        raise ValueError("Unsupported file format. Use .pdb or .gro")

    if not output_file:
        output_file = input_path.with_suffix(".fasta")

    with open(output_file, "w") as fasta:
        fasta.write(f">{input_path.stem}\n")
        for i in range(0, len(sequence), 80):
            fasta.write(sequence[i : i + 80] + "\n")

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
#		- dt_analysis_ps: sets a value in ps. This will be (roughlz) the 
#						  time steps for analzsis
#	OUTPUT:
#		-averaged SAXS profile (q-space) with standard deviation
#		-pandas dataframe
#		-3 columns (q in 1/A; mean in a.u.; sd in a.u.)

def calculate_SAXS_profile_crysol(gro_file, xtc_file,dt_analysis_ps=100):
    # Load structure and trajectory (PBC: nojump)

    u = mda.Universe(gro_file, xtc_file)
    
    # only select protein atoms
    protein_atoms = u.select_atoms("protein")
    
    # timesteps between analyzed frames (dt_analysis_ps) should be roughly 100 ps
    # 	find out the timestep in the trj 
    dt_trj_ps = u.trajectory.dt
    # get interval for frame analysis
    analysis_frame_interval = round(dt_analysis_ps/dt_trj_ps,0)
    
    # create pandas dataframe which stores the calculated SAXS profiles
    profiles = pd.DataFrame()
    
    # iterate over individual frames
    for ts in u.trajectory:
        # CHECK, IF CONSIDERED - only when frame_idx%analysis_frame_interval = 0
        # get current frame index
        frame_idx = ts.frame
        
        if frame_idx%analysis_frame_interval == 0:
            # PDB OUT
            # write out single PDB files
            filName_PDB = "frame_"+str(frame_idx)+".pdb"
            protein_atoms.write(filName_PDB)
            # for some reason, crysol expects a different naming of some atoms in pdb
            #	known issues for
            #		ILE and termini
            #	so for this use sed from command line, since this is much faster than 
            #		any python tool
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
    
    # GET AVERAGE AND SD FOR SAXS PROFILE 
    # calculate the mean profile and the corresponding sd
    profile_mean = profiles.mean(axis=1)
    profile_sd = profiles.std(axis=1)
    
    # merge data and set column names
    res = pd.DataFrame([tab["q"],profile_mean,profile_sd]).transpose()
    res.columns = ["q[1/A]","mean_I(q)[a.u.]","sd_I(q)[a.u.]"]
    
    return(res)


# 18.06.2025
# added during NMRLipids meeting in Bergen
# by Tobi R
# FUNCTION TO CALCULATE THE SAXS SCATTERING PROFILE OF THE TRJ
#	needs the python package MAICoS
#		https://gitlab.com/maicos-devel/maicos
#		https://maicos.readthedocs.io/en/main/analysis-modules/saxs.html
#	INPUT: 
#		- gro_file; xtc_file: gro file and xtc file (nojump!!!) for MDAnalysis
#		- water_shell: if submitted as interger or float, all atoms within this
#					   value (distance to protein) will be considered for the
#					   hydration shell; otherwise only protein is considered 
#		- output_file: if true, file of calculated data will be saved in MAICoS format
#	OUTPUT:
#		-SAXS profile (q-space) of the trajectory
#		-pandas dataframe
#		-2 columns (q in 1/A; mean in a.u.)

def calculate_SAXS_profile_maicos(gro_file, xtc_file,water_shell=None,output_file=False):
    # Load structure and trajectory (PBC: nojump)
    u = mda.Universe(gro_file, xtc_file)
    
    # consider hydration shell or not
    #	considered: if hydration_shell is an integer or float
    #				this value will be used for cutoff (in A)
    #	not considered: any other datatype
    if type(water_shell) == int or type(water_shell)==float:
        protein = u.select_atoms("protein")
        int_water = u.select_atoms("(around "+str(water_shell)+" protein) and (not type DUMMY)", updating=True)
        sel_atoms = protein + int_water
        print("  NOTE: Water shell is considered.")
    else:
        sel_atoms = u.select_atoms("protein")
        print("  NOTE: Water shell is NOT considered.")
    
    # run the program:
    #	-since we use nojump simulations, we can set unwrap to false:
    #	-since most scattering profiles of IDPs are not larger than q=0.5, adjust:
    #		qmax = 0.5 / qmin = 0 / dq = 0.0025
    #	-see MAICoS documentation: https://maicos.readthedocs.io/en/main/analysis-modules/saxs.html
    # create the maicos object
    #print(sel_atoms)
    SAXS = maicos.Saxs(atomgroup=sel_atoms,unwrap=False,qmin=0,qmax=0.5005,dq=0.0025,jitter=10**(-3))
    
    # run the analysis
    profile = SAXS.run()
    
    # extract the calculated SAXS profile:
    scattering_vectors = profile.results.scattering_vectors
    intensity = profile.results.scattering_intensities
    
    # check, if data should be saved in maicos format
    if output_file:
        SAXS.save()
    
    # merge data for return and set column names
    res = pd.DataFrame([scattering_vectors,intensity]).transpose()
    res.columns = ["q[1/A]","I(q)[a.u.]"]
    
    return(res)
    
    
# 20.06.2025
# added during NMRLipids meeting in Bergen
# by Tobi R
# FUNCTION TO CALCULATE THE AVERAGED CHEMICAL SHIFTS OF THE TRJ
#	needs the external program sparta+
#		- https://spin.niddk.nih.gov/bax-apps/software/SPARTA+/
#		- path to sparta+ needs to be added to $PATH 
#		- after running install.com, the output needs to be added to ".cshrc"
#			in your "~/" path // 
#			if it does not exist, create it and add lines
#	INPUT: 
#		- gro_file; xtc_file: gro file and xtc file (nojump!!!) for MDAnalysis
#		- dt_analysis_ps: sets a value in ps. This will be (roughly) the 
#						  time steps for analysis
#		- be_quiet: can be True or False: If True, sparta+ output 
#					(including error messages) will be suppressed
#	OUTPUT:
#		- table of averaged chemical shifts with standard deviation over trj
#		- pandas dataframe
#		- 6 columns (residue ID; residue name; atom name; 
#			mean chemical shift in ppm, standard deviation(sd) of chemical shift in ppm
#	COMMENT:
#		- replaces the function: calculate_ChemShifts_sparta_OUTDATED
#		- here it writes out only a single PDB file for analysis
#		- solves disk space problem
#		- might be slover than the OUTDATED function

def calculate_ChemShifts_sparta(gro_file, xtc_file, dt_analysis_ps=100000,be_quiet=True):
    # Load structure and trajectory (PBC: nojump)
    u = mda.Universe(gro_file, xtc_file)
    
    # only select protein atoms
    protein_atoms = u.select_atoms("protein")
    
    # timesteps between analyzed frames (dt_analysis_ps) should be roughly 100 ps
    # 	find out the timestep in the trajectory 
    dt_trj_ps = u.trajectory.dt
    # get an interval for frame analysis
    analysis_frame_interval = round(dt_analysis_ps/dt_trj_ps,0)
    
    # create pandas dataframe which stores the calculated SAXS profiles
    all_shifts = pd.DataFrame()
    
    # iterate over individul frames
    for ts in u.trajectory:
        # CHECK, IF CONSIDERED - only when frame_idx%analysis_frame_interval = 0
        # get current frame index
        frame_idx = ts.frame
        
        if frame_idx%analysis_frame_interval == 0:
            # PDB OUT
            # write out single PDB files
            filName_PDB = "frame_"+str(frame_idx)+".pdb"
            protein_atoms.write(filName_PDB)
            # RUN SPARTA+
            if be_quiet:
                shell_command = "sparta+ -in " +filName_PDB+ " -out pred_"+ str(frame_idx) +".tab > /dev/null 2>&1"
            else:
                shell_command = "sparta+ -in " +filName_PDB+ " -out pred_"+ str(frame_idx) +".tab"
            OUT = subprocess.run(shell_command, shell=True)
            
            # READ SAXS PROFILE FILE
            tab = pd.read_csv("pred_"+ str(frame_idx) +".tab",delim_whitespace=True, skiprows=28, header=None)
            tab.columns=['RESID', 'RESNAME', 'ATOMNAME', 'SS_SHIFT', 'SHIFT', 'RC_SHIFT', 'HM_SHIFT', 'EF_SHIFT', 'SIGMA']
            
            # SAVE FILES IN DATAFRAME
            shifts = tab["SHIFT"]
            all_shifts = pd.concat([all_shifts,shifts],axis=1)
            
            # CLEAN UP 
            os.remove("frame_"+str(frame_idx)+".pdb")
            os.remove("pred_"+str(frame_idx)+".tab")
            os.remove("struct.tab")
            
    # GET AVERAGE AND SD SAXS PROFILE 
    # calculate the mean profile and the corresponding sd
    shifts_mean = all_shifts.mean(axis=1)
    shifts_sd = all_shifts.std(axis=1)
    
    # merge data and set column names
    res = pd.DataFrame([tab["RESID"],tab["RESNAME"],tab["ATOMNAME"],shifts_mean,shifts_sd]).transpose()
    res.columns = ["ResID","ResName","AtomName","ShiftMean[ppm]","ShiftSD[ppm]"]
    
    return(res)
    
    
# 19.06.2025
# added during NMRLipids meeting in Bergen
# by Tobi R
# FUNCTION TO CALCULATE THE AVERAGED CHEMICAL SHIFTS OF THE TRJ
#	needs the python package mdtraj
#		https://www.mdtraj.org/1.9.8.dev0/installation.html
#	needs the external program sparta+
#		- https://spin.niddk.nih.gov/bax-apps/software/SPARTA+/
#		- path to sparta+ needs to be added to $PATH 
#		- after running install.com, the output needs to be added to ".cshrc"
#			in your "~/" path // 
#			if it does not exist, create it and add lines
#	INPUT: 
#		- gro_file; xtc_file: gro file and xtc file (nojump!!!) for MDAnalysis
#		- dt_analysis_ps: sets a value in ps. This will be (roughly) the 
#						  time steps for analzsis
#	OUTPUT:
#		-table of averaged chemical shifts with standard deviation over trj
#		-pandas dataframe
#		-3 columns (atom/nuclei name; mean chemical shift, standard deviation(sd) of chemical shift
#	KNOWN ISSUES:
#		- seems that the implementation in mdtraj writes out the whole trj
#			in PDB files and then handles this files with sparta+
#		- this results in the issue, that we can out of disk space; 
#			analysis of a single trj requieres easily >100 GB
#		- files are saved in an unknown location, so if this code crashes, 
#			they will not be deleted
#		- SOLUTION: Use a different implementation: calculate_ChemShifts_sparta

def calculate_ChemShifts_sparta(gro_file, xtc_file, dt_analysis_ps=1000):

    # in this function we use mdtraj and NOT MDAnalysis
    
    # LOAD DATA
    # load trj with  mdtraj
    traj = mdtraj.load(xtc_file, top=gro_file)

    # GET PROTEINE ONLY TRJ
    idx_prot_atoms = traj.topology.select('protein')
    # slice trajectory to keep only protein atoms
    traj_just_prot = traj.atom_slice(idx_prot_atoms)
    
    # ADJUST TIME STEPS FOR ANALYSIS
    # timesteps between analyzed frames (dt_analysis_ps) should be roughly 100 ps
    # 	find out the timestep in the trajectory 
    dt_trj_ps = traj_just_prot.time[1]-traj_just_prot.time[0]
    # get an interval for frame analysis
    analysis_frame_interval = int(round(dt_analysis_ps/dt_trj_ps,0))
    # apply to trajectory
    traj_just_prot = traj_just_prot[::analysis_frame_interval]

    
    print('loaded')
    
    
    
    # RUN SPARTA
    ChemShifts_all_frames = mdtraj.chemical_shifts_spartaplus(traj_just_prot, rename_HN=True)
    
    # AVERAGE RESULTS
    # shift_all_frames is a pandas dataframe:
    #	rows are the individual atoms/nuclei in the protein
    #	columns are the individual frames of the traj
    # so calculate the mean/sd over each row
    mean_ChemShifts = ChemShifts_all_frames.mean(axis=1)
    # get SD
    sd_ChemShifts = ChemShifts_all_frames.std(axis=1)
    
    # PREPARE DATA FOR RETURN
    res = pd.DataFrame([mean_ChemShifts,sd_ChemShifts]).transpose()
    res.columns = ["meanChemShifts[ppm]","SDChemShifts[ppm]"]
    
    return(res)

def calculate_spin_relaxation_time_RMSD(spin_relaxation_time_file,experimental_data_file):

    with open(spin_relaxation_time_file, "r") as file:
        spin_relaxation_times = yaml.safe_load(file)

    with open(experimental_data_file, "r") as file:
        experimental_data = yaml.safe_load(file)

    rounded_data = {}
    for residue, freq_dict in experimental_data.items():
        new_freq_dict = {}
        for freq, values in freq_dict.items():
            rounded_freq = round(float(freq))
            new_freq_dict[rounded_freq] = values
        rounded_data[residue] = new_freq_dict

    experimental_data = rounded_data
    #print(experimental_data)
        
    differences = {}
    for residue in spin_relaxation_times:
        for magnetic_field in spin_relaxation_times[residue]:
            #print(residue,magnetic_field)
            try:
                differences[residue] = {
                    'R1': 1/spin_relaxation_times[residue][magnetic_field]['T1']['value'] - 1/experimental_data[residue][magnetic_field]['T1']['value'],
                    'R2': 1/spin_relaxation_times[residue][magnetic_field]['T2']['value'] - 1/experimental_data[residue][magnetic_field]['T2']['value'],
                    'hetNOE': spin_relaxation_times[residue][magnetic_field]['hetNOE']['value'] - experimental_data[residue][magnetic_field]['hetNOE']['value']
                }
                continue
            except:
                pass
            #print(experimental_data.keys())
            #print(experimental_data[int(re.sub(r"\D", "", residue))])
            try:
                #print(residue + " " + re.sub(r"\D", "", residue))
                #print(experimental_data[int(re.sub(r"\D", "", residue))])
                differences[residue] = {
                    'R1': 1/spin_relaxation_times[residue][magnetic_field]['T1']['value'] - 1/experimental_data[int(re.sub(r"\D", "", residue))][magnetic_field]['T1']['value'],
                    'R2': 1/spin_relaxation_times[residue][magnetic_field]['T2']['value'] - 1/experimental_data[int(re.sub(r"\D", "", residue))][magnetic_field]['T2']['value'],
                    'hetNOE': spin_relaxation_times[residue][magnetic_field]['hetNOE']['value'] - experimental_data[int(re.sub(r"\D", "", residue))][magnetic_field]['hetNOE']['value']
                }
                continue
            except:
                pass

            print('Calculation of difference failed for ' + residue + " " + re.sub(r"\D", "", residue))
            continue

    RMSDs = {}

    values = []
    for residue in differences:
        values.append(differences[residue]['R1']**2)
    RMSDs['R1'] = np.sqrt(sum(values) / len(values))

    values = []
    for residue in differences:
        values.append(differences[residue]['R2']**2)
    RMSDs['R2'] =  np.sqrt(sum(values) / len(values))

    values = []
    for residue in differences:
        values.append(differences[residue]['hetNOE']**2)
    RMSDs['hetNOE'] =  np.sqrt(sum(values) / len(values))

    RMSDs['differences'] = differences
    
    return(RMSDs)
    


def convert_original_to_nested_dict(data: dict) -> dict:
    """
    Convert the original nested dict:
    {'meanChemShifts[ppm]': {(1, 'C'): 174.59, (1, 'CA'): 58.19, ...}}

    directly into:
    {
      1: {"C": 174.59, "CA": 58.19, "CB": 63.19},
      2: {"C": 174.76, "N": 120.67}
    }
    """
    inner_dict = next(iter(data.values()))
    nested_dict = defaultdict(dict)
    for (residue, atom), value in inner_dict.items():
        nested_dict[residue][atom] = value
    return dict(nested_dict)



def parse_star_file(filename):
    """
    Parses chemical shift data from a NMR-star file
    """
    atom_map = {
        "C": "C",
        "CA": "CA",
        "CB": "CB",
        "HA": "HA",
        "H": "H",
        "HA2": "HA",
        "HA3": "HA",
        "Hα": "HA",
        "N":"N",
    }
    shifts = {}
    with open(filename, "r") as f:
        lines = f.readlines()
    current_loop = []
    in_loop = False
    tags = []
    for line in lines:
        line = line.strip()
        if line.startswith("loop_"):
            in_loop = True
            current_loop = []
            tags = []
        elif in_loop and line.startswith("_Atom_chem_shift"):
            tags.append(line)
        elif in_loop and line == "stop_":
            # Parse collected loop
            for row in current_loop:
                if len(row) != len(tags):
                    continue
                try:
                    row_dict = dict(zip(tags, row))
                    resname = row_dict["_Atom_chem_shift.Comp_ID"]
                    res = str(int(row_dict["_Atom_chem_shift.Seq_ID"])) + resname
                    atom = row_dict["_Atom_chem_shift.Atom_ID"]
                    val = float(row_dict["_Atom_chem_shift.Val"])
                    mapped_atom = atom_map.get(atom.upper(), None)
                    if mapped_atom:
                        shifts.setdefault(res, {})[mapped_atom] = val
                except Exception:
                    continue
            in_loop = False
        elif in_loop:
            # Split line into fields
            row = line.split()
            if row:
                current_loop.append(row)
    return shifts


#def compute_rmsd_chemical_shift(sim_data, exp_data, nuclei, residues):
#    rmsd_per_nucleus = {}
#    for nucleus in nuclei:
#        diff_sq = []
#        for res in residues:
#            sim_val = sim_data.get(res, {}).get(nucleus, None)
#            exp_val = exp_data.get(res, {}).get(nucleus, None)
#            if sim_val is not None and exp_val is not None:
#                diff_sq.append((sim_val - exp_val)**2)
#        if diff_sq:
#            rmsd_per_nucleus[nucleus] = math.sqrt(np.mean(diff_sq))
#        else:
#            rmsd_per_nucleus[nucleus] = None
#    return rmsd_per_nucleus

def compute_rmsd_chemical_shift(sim_data, exp_data, nuclei, residues):
    """
    Compute RMSD values of chemical shifts per nucleus, along with per-residue differences.

    Parameters
    ----------
    sim_data : dict
        Dictionary with simulated chemical shifts.
        Format: {residue_number: {nucleus: value}}
    exp_data : dict
        Dictionary with experimental chemical shifts.
        Format: {residue_number: {nucleus: value}}
    nuclei : list of str
        List of nucleus types to evaluate (e.g. ["C", "CA", "CB"]).
    residues : list of int
        Residues to consider for the RMSD calculation.

    Returns
    -------
    dict
        Dictionary with the following structure:
        {
            "C": <RMSD value for nucleus C>,
            "CA": <RMSD value for nucleus CA>,
            ...
            "differences": {
                <residue_number>: {
                    "C": <sim - exp>,
                    "CA": <sim - exp>,
                    ...
                },
                ...
            }
        }

        Notes:
        - RMSDs are computed as sqrt(mean((sim - exp)^2)) across the selected residues.
        - If a simulated or experimental value is missing, the difference is stored as None
          and excluded from the RMSD calculation.
    """
    result = {"differences": {}}

    # Initialize empty difference dict for each residue
    for res in residues:
        result["differences"][res] = {}

    # Compute per-nucleus RMSD and per-residue differences
    for nucleus in nuclei:
        diff_sq = []
        for res in residues:
            sim_val = sim_data.get(res, {}).get(nucleus, None)
            exp_val = exp_data.get(res, {}).get(nucleus, None)

            print(res)
            
            if sim_val is not None and exp_val is not None:
                diff = sim_val - exp_val
                if diff < 5:
                    diff_sq.append(diff**2)
                    result["differences"][res][nucleus] = diff
                else:
                    result["differences"][res][nucleus] = None    
            else:
                result["differences"][res][nucleus] = None

        # Store RMSD for this nucleus at the top level
        if diff_sq:
            result[nucleus] = math.sqrt(np.mean(diff_sq))
        else:
            result[nucleus] = None

    return result

def download_NMR_star_file(BMRBid):
    bmrb_url = "https://bmrb.io/ftp/pub/bmrb/entry_directories/bmr" + BMRBid +"/bmr" + BMRBid + "_3.str"
    exp_data_path = '../../Data/Experiments/chemical_shift/BMRBid' + BMRBid
    os.makedirs(exp_data_path, exist_ok=True)
    bmrb_local_file = exp_data_path + "/bmr" + BMRBid + ".str"

    if not os.path.exists(bmrb_local_file):
        print("Downloading BMRB NMR-STAR file to ",  bmrb_local_file)
        r = requests.get(bmrb_url)
        r.raise_for_status()
        with open(bmrb_local_file, "wb") as f:
            f.write(r.content)
                
        print("NMR-star file downloaded to ", bmrb_local_file)

    return bmrb_local_file


def extract_chemical_shift_experiment_name_from_star(star_file_path):
    """
    Extracts all experiment names from the _Chem_shift_experiment loop of an NMR-STAR file.
    Handles both quoted and unquoted experiment names.

    Parameters
    ----------
    star_file_path : str
        Path to the NMR-STAR file.

    Returns
    -------
    list[str]
        List of extracted experiment names (empty list if none found).
    """
    with open(star_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_chem_shift_loop = False
    experiment_names = []

    for line in lines:
        line = line.strip()

        # Activate only after finding the Chem_shift_experiment section
        if line.startswith("_Chem_shift_experiment.Experiment_ID"):
            in_chem_shift_loop = True
            continue

        # Stop reading once we reach the end of the loop
        if in_chem_shift_loop and line.startswith("stop_"):
            break

        # Skip headers
        if in_chem_shift_loop and line.startswith("_Chem_shift_experiment."):
            continue

        # Match data lines: first column = ID (integer)
        if in_chem_shift_loop and re.match(r"^\d+\s+", line):
            # Split safely by whitespace but preserve quoted strings if any
            parts = re.findall(r"(?:'[^']*'|\"[^\"]*\"|\S+)", line)
            if len(parts) >= 2:
                name = parts[1].strip("'\"")  # remove quotes if present
                experiment_names.append(name)

    return experiment_names


from Bio import SeqIO

# One-letter to three-letter amino acid code mapping
AA_MAP = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


from io import StringIO

def fasta_string_to_residue_dict(fasta_str):
    """
    Converts a raw FASTA sequence (with or without header) into
    {residue_number: three_letter_code}.
    """
    fasta_str = fasta_str.strip().upper()

    # Remove FASTA header if present
    if fasta_str.startswith(">"):
        lines = fasta_str.splitlines()[1:]
        seq = "".join(lines)
    else:
        seq = fasta_str.replace("\n", "")

    if not seq:
        raise ValueError("No sequence data found in FASTA string.")

    # Map residues
    return {i + 1: AA_MAP.get(res, "UNK") for i, res in enumerate(seq)}



def fasta_to_residue_dict(fasta_path):
    """
    Reads a FASTA file and returns a dictionary:
    {residue_number: three_letter_code}
    """
    # Parse the FASTA file (takes first sequence if multiple)
    record = next(SeqIO.parse(fasta_path, "fasta"))
    seq = str(record.seq).upper()

    # Build residue number → 3-letter code dictionary
    res_dict = {i + 1: AA_MAP.get(res, "UNK") for i, res in enumerate(seq)}

    return res_dict



def percentage_over_accuracy(accuracy,value):
    ratio = abs(value)/accuracy
    if ratio > 1:
        return ratio-1
    else:
        return 0

def plot_quality_heatmap(data_dict, save_path, vmax=None):
    """
    Create a traffic-light style heatmap showing quality metrics for each residue and nucleus.

    Parameters
    ----------
    data_dict : dict
        Dictionary where keys are residues (e.g., "99GLN") or "Averages",
        and values are dicts mapping nuclei (e.g., "CA", "N", "Sum") to scores.

    save_path : str
        Path where the generated heatmap image will be saved.

    vmax : float, optional
        Maximum value for the color scale. If None, scale automatically to data range.
        Useful for comparing multiple heatmaps with consistent color limits.
    """

    # --- Prepare data ---
    df = pd.DataFrame(data_dict).T

    # Remove "Averages" row temporarily for sorting residues numerically
    if "Averages" in df.index:
        averages = df.loc["Averages"]
        df = df.drop("Averages")
    else:
        averages = None

    # Sort residues numerically if possible
    def residue_sort_key(name):
        if isinstance(name, str):
            match = re.match(r"(\d+)", name)
            if match:
                return int(match.group(1))
        return float("inf")

    df = df.loc[sorted(df.index, key=residue_sort_key)]

    # Add Averages back to the end
    if averages is not None:
        df.loc["Averages"] = averages

    # --- Transpose for plotting (residues on x-axis, nuclei on y-axis) ---
    df_t = df.T

    df_t.to_csv(save_path.replace(".png", ".csv"))
    
    # --- Plot ---
    plt.figure(figsize=(10, 5))
    ax = sns.heatmap(
        df_t,
        cmap="RdYlGn_r",        # traffic-light color scheme (green = good, red = bad)
        cbar=True,
        linewidths=0.5,
        vmax=vmax               # 👈 fixed color scale if provided
    )
    ax.set_xlabel("Residue", fontsize=20)
    ax.set_ylabel("Nucleus", fontsize=20)
    ax.set_title("Chemical shift quality", fontsize=20)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=18)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def evaluate_chemical_shift_quality(system, atom_accuracies):
    """
    Evaluate chemical shift quality for a given system.

    Parameters
    ----------
    system : dict
        Dictionary containing simulation info with keys:
        - 'TRJ': trajectory identifier
        - 'ID': system ID
        - 'path': relative path to data folder
        - 'EXPERIMENT': includes 'chemical_shift' -> 'alignment_score'

    atom_accuracies : dict
        Mapping from nucleus (e.g. 'C', 'CA', 'N', ...) to accuracy thresholds.

    percentage_over_accuracy : callable
        Function taking (accuracy, rmsd_value) -> quality percentage.

    plot_quality_heatmap : callable
        Function taking (data_dict, save_path) to create and save a heatmap.

    Returns
    -------
    str
        Path to the generated quality heatmap file.
    """

    print(system["TRJ"], system["ID"], system["EXPERIMENT"]["chemical_shift"]["alignment_score"])

    # --- Set up file paths ---
    data_folder = os.path.join("../../Data/Simulations", system["path"])
    chemical_shifts_file_rmsd = os.path.join(data_folder, "chemical_shift_rmsd.yaml")

    if not os.path.exists(chemical_shifts_file_rmsd):
        print(f"⚠️ Missing file: {chemical_shifts_file_rmsd}")
        return None

    quality_evaluation_file = os.path.join(data_folder, "chemical_shift_quality.yaml")
    quality_heatmap_file = os.path.join(data_folder, "chemical_shift_quality_heatmap.png")

    # --- Initialize quality dict ---
    quality = {"Averages": {}}

    # --- Load RMSD data ---
    with open(chemical_shifts_file_rmsd) as f:
        rmsds = yaml.safe_load(f)

    # --- Compute quality measures ---
    for key in rmsds.keys():
        if key == "differences":
            for residue, values in rmsds[key].items():
                if residue not in quality:
                    quality[residue] = {}
                res_sum = 0
                for nuclei, value in values.items():
                    if value is None:
                        continue
                    accuracy = atom_accuracies.get(nuclei)
                    if accuracy is None:
                        continue
                    score = percentage_over_accuracy(accuracy, value)
                    quality[residue][nuclei] = score
                    res_sum += score
                #quality[residue]["Sum"] = res_sum
        else:
            value = rmsds[key]
            if value is None:
                continue
            accuracy = atom_accuracies.get(key)
            if accuracy is None:
                continue
            quality["Averages"][key] = percentage_over_accuracy(accuracy, value)

    # --- Write YAML output ---
    with open(quality_evaluation_file, "w") as f:
        yaml.dump(quality, f, sort_keys=True)

    # --- Generate heatmap ---
    print(f"🧩 Generating heatmap: {quality_heatmap_file}")
    plot_quality_heatmap(quality, quality_heatmap_file,2)
    print("✅ Heatmap saved successfully.")

    return quality_heatmap_file

def evaluate_spin_relaxation_quality(system, relaxation_accuracies):
    """
    Evaluate spin relaxation quality (R1, R2, hetNOE) for a given system.

    Parameters
    ----------
    system : dict
        Dictionary containing simulation info with keys:
        - 'TRJ': trajectory identifier
        - 'ID': system ID
        - 'path': relative path to data folder
        - 'EXPERIMENT': includes 'relaxation' -> 'alignment_score'

    relaxation_accuracies : dict
        Mapping from relaxation type ('R1', 'R2', 'hetNOE') to accuracy thresholds.

    percentage_over_accuracy : callable
        Function taking (accuracy, rmsd_value) -> quality percentage.

    plot_quality_heatmap : callable
        Function taking (data_dict, save_path) to create and save a heatmap.

    Returns
    -------
    str
        Path to the generated relaxation quality heatmap file.
    """

    #print(system["TRJ"], system["ID"], system["EXPERIMENT"]["spirelaxation"]["alignment_score"])

    # --- Define paths ---
    data_folder = os.path.join("../../Data/Simulations", system["path"])
    relaxation_file_rmsd = os.path.join(data_folder, "spin_relaxation_rmsd.yaml")

    if not os.path.exists(relaxation_file_rmsd):
        print(f"⚠️ Missing file: {relaxation_file_rmsd}")
        return None

    quality_file = os.path.join(data_folder, "spin_relaxation_quality.yaml")
    heatmap_file = os.path.join(data_folder, "spin_relaxation_quality_heatmap.png")

    # --- Initialize output dict ---
    quality = {"Averages": {}}

    # --- Load RMSD data ---
    with open(relaxation_file_rmsd) as f:
        rmsds = yaml.safe_load(f)

    # --- Compute quality measures ---
    for key in rmsds.keys():
        if key == "differences":
            for residue, values in rmsds[key].items():
                if residue not in quality:
                    quality[residue] = {}
                res_sum = 0
                for relaxation_type, value in values.items():
                    if value is None:
                        continue
                    accuracy = relaxation_accuracies.get(relaxation_type)
                    if accuracy is None:
                        continue
                    score = percentage_over_accuracy(accuracy, value)
                    quality[residue][relaxation_type] = score
                    res_sum += score
                #quality[residue]["Sum"] = res_sum
        else:
            # Average metrics (like R1, R2, hetNOE) outside of "differences"
            value = rmsds[key]
            if value is None:
                continue
            accuracy = relaxation_accuracies.get(key)
            if accuracy is None:
                continue
            quality["Averages"][key] = percentage_over_accuracy(accuracy, value)

    # --- Write YAML output ---
    with open(quality_file, "w") as f:
        yaml.dump(quality, f, sort_keys=True)

    # --- Generate heatmap ---
    print(f"🧩 Generating relaxation heatmap: {heatmap_file}")
    plot_quality_heatmap(quality, heatmap_file,3)
    print("✅ Relaxation heatmap saved successfully.")

    return heatmap_file


import yaml
from numbers import Number

def compute_residue_nonzero_percentages(input_yaml: str, output_yaml: str) -> None:
    """
    Adds `non_zero_percentage` into each residue block of the original YAML.
    """

    with open(input_yaml, "r") as f:
        data = yaml.safe_load(f)

    all_percentages = []

    for residue, atoms in data.items():

        # Case 1: residue is a single number
        if isinstance(atoms, Number):
            percentage = 100.0 if atoms != 0 else 0.0
            data[residue] = {
                "value": atoms,
                "non_zero_percentage": percentage
            }

        # Case 2: residue is a dict
        elif isinstance(atoms, dict):
            numeric_values = []

            for key, value in atoms.items():
                if key == "non_zero_percentage":
                    continue  # avoid double counting if rerun

                if isinstance(value, Number):
                    numeric_values.append(value)
                elif isinstance(value, dict):
                    for v in value.values():
                        if isinstance(v, Number):
                            numeric_values.append(v)

            if numeric_values:
                nonzero_count = sum(1 for v in numeric_values if v != 0)
                percentage = 100.0 * nonzero_count / len(numeric_values)
            else:
                percentage = 0.0

            atoms["non_zero_percentage"] = percentage

        # Case 3: unexpected type
        else:
            percentage = 0.0

        all_percentages.append(percentage)

    overall_percentage = (
        sum(all_percentages) / len(all_percentages)
        if all_percentages else 0.0
    )

    # Optional: store global value at top level
    data["overall_non_zero_percentage"] = overall_percentage

    with open(output_yaml, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)




    plt.tight_layout()
    
    plt.savefig("combined_ensemble_secondary.png", dpi=300)
    #plt.show()


def calculate_secondary_structures(gro_file, xtc_file):

    import mdtraj
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from pymol import cmd

    print(xtc_file, gro_file)

    # -------------------------
    # LOAD TRAJECTORY
    # -------------------------
    traj = mdtraj.load(xtc_file, top=gro_file)

    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }

    residue_names = [three_to_one[res.name] for res in traj.topology.residues]

    # -------------------------
    # DSSP
    # -------------------------
    dssp = mdtraj.compute_dssp(traj, simplified=True)

    initial_ss = dssp[0]
    final_ss = dssp[-1]

    color_map = {'C': 'grey', 'H': 'red', 'E': 'blue'}

    initial_colors = [color_map[s] for s in initial_ss]
    final_colors = [color_map[s] for s in final_ss]

    # -------------------------
    # PYMOL ENSEMBLE IMAGE
    # -------------------------
    cmd.set("ray_opaque_background", 1)

    obj_name = "traj"
    cmd.load(gro_file, obj_name)
    cmd.load_traj(xtc_file, obj_name, state=1, interval=2000)

    cmd.color("blue", obj_name)
    cmd.set("all_states", "on")
    cmd.orient()
    cmd.ray(600, 600)

    pymol_img = "ensemble.png"
    cmd.png(pymol_img)
    cmd.delete("all")

    # Convert PyMOL image → matplotlib fig
    img = mpimg.imread(pymol_img)
    fig_ensemble, ax_e = plt.subplots(figsize=(6, 6))
    ax_e.imshow(img)
    ax_e.axis("off")
    ax_e.set_title("Conformational Ensemble")

    # -------------------------
    # COMBINED FIGURE
    # -------------------------
    fig_combined = plt.figure(figsize=(18, 7))
    gs = fig_combined.add_gridspec(3, 1, height_ratios=[2, 1, 1])

    ax_img = fig_combined.add_subplot(gs[0])
    ax_init = fig_combined.add_subplot(gs[1])
    ax_final = fig_combined.add_subplot(gs[2])

    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title("Conformational Ensemble")

    for i, c in enumerate(initial_colors):
        ax_init.bar(i, 1, color=c)

    for i, c in enumerate(final_colors):
        ax_final.bar(i, 1, color=c)

    for ax in [ax_init, ax_final]:
        ax.set_xlim(0, len(dssp[0]))
        ax.set_ylim(0, 1)
        ax.set_yticks([])

    xticks = list(range(0, len(dssp[0]), 10))

    ax_final.set_xticks(xticks)
    ax_final.set_xticklabels([f"{i+1}" for i in xticks], rotation=90)
    ax_init.set_xticks([])

    ax_final.set_xlabel("Residue Index")

    ax_init.set_title("Initial Secondary Structure")
    ax_final.set_title("Final Secondary Structure")

    legend_labels = ['Coil (C)', 'Helix (H)', 'Extended (E)']
    handles = [plt.Line2D([0], [0], color=color_map[k], lw=4) for k in color_map]
    ax_final.legend(handles, legend_labels, loc='upper right')

    plt.tight_layout()
    fig_combined.savefig("combined_ensemble_secondary.png", dpi=300)

    # -------------------------
    # COIL PROBABILITIES
    # -------------------------
    coil_counts = np.sum(dssp == 'C', axis=0)
    total_frames = dssp.shape[0]
    coil_probabilities = coil_counts / total_frames

    ### dictionary output
    ##coil_dict = {
    ##    f"{i+1}_{residue_names[i]}": float(prob)
    ##    for i, prob in enumerate(coil_probabilities)
    ##}


    structures = ['H', 'B', 'E', 'G', 'I', 'T', 'S', 'C']

    total_frames = dssp.shape[0]

    ss_dict = {}

    for i in range(dssp.shape[1]):  # safer than enumerate(residue_names)
        res_key = f"{i+1}_{residue_names[i]}"
    
        ss_dict[res_key] = {
            ss: float(np.sum(dssp[:, i] == ss) / total_frames)
            for ss in structures
        }

    coil_dict = ss_dict


    
    # -------------------------
    # ORDER PROBABILITY FIG
    # -------------------------
    fig_order = plt.figure(figsize=(18, 4))
    plt.bar(range(len(coil_probabilities)), coil_probabilities, color='grey')
    plt.xlabel('Residue Index')
    plt.ylabel('Coil Probability')
    plt.title('Coil Probability for Each Residue')

    plt.xticks(
        ticks=range(len(coil_probabilities)),
        labels=[f"{i+1} {residue_names[i]}" for i in range(len(coil_probabilities))],
        rotation=90
    )

    plt.tight_layout()
    fig_order.savefig("order_probability.png", dpi=300)

    # -------------------------
    # RETURN EVERYTHING
    # -------------------------
    return fig_combined, fig_order, fig_ensemble, coil_dict




