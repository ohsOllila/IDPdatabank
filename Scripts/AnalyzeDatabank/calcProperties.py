## python3 create_contact.py 
## python3 create_distance.py
## #python3 $secondary
## python3 correlationCALC.py
## python3 ${path}/Old_Relaxations_for_Samuli.py

from DatabankLib.protein_functions import *
import yaml

databankPath = "/home/sosamuli/work/NMRlipids/IDPdatabank/"  # this is the local path for the cloned Databank
os.environ["NMLDB_ROOT_PATH"] = "/home/sosamuli/work/NMRlipids/IDPdatabank/"

#databankPath = "/home/sosamuli/work/NMRlipids/IDPsimBank/"  # this is the local path for the cloned Databank
#os.environ["NMLDB_ROOT_PATH"] = "/home/sosamuli/work/NMRlipids/IDPsimBank/"



# These two lines include core Databank routines and Databank API
from DatabankLib.core import *
from DatabankLib.databankLibrary import *
# This is for plotting
from DatabankLib.databankio import *
from DatabankLib.plottings import plotSimulation
#from IPython.display import display, Markdown

# This initializes the databank and stores the information of all simulations into a list.
# Each list item contains the information from README.yaml file of the given simulation.
systems = initialize_databank()


trjconvCOMMAND = 'gmx trjconv'

for system in systems:

    if system['TYPEOFSYSTEM'] != 'protein':
        continue

    print(system['path'])

    dataFolder = databankPath + 'Data/Simulations/' + system['path']

    print(system['TRJ'][0])
    trj_fname = dataFolder + system['TRJ'][0][0] 
    top_fname = dataFolder + system['TPR'][0][0]
    gro_fname = dataFolder + 'protein_centered.gro'

    trj_fname_noPBC = dataFolder + 'traj_noPBC.xtc'

    
    SAXS_file = dataFolder + 'SAXS.yaml'
    SAXS_file_MAICoS = dataFolder + 'SAXS_MAICoS.yaml'
    chemical_shift_file = dataFolder + 'chemical_shifts_sparta.yaml'
    Contact_map_file = dataFolder + "Contact_map.png"
    distance_map_file = dataFolder + "Distance_map.png"
    backbone_correlation_file = dataFolder + "Backbone_correlations.png" 
    rog_file = dataFolder + "gyrate.xvg"
    dynamic_landscape_file = dataFolder + 'dynamic_landscape_Coeffs.yaml'
    spin_relaxation_time_file = dataFolder + 'spin_relaxation_times.yaml'
    
    files = {
        "SAXS_file": SAXS_file,
#        "SAXS_file_MAICoS": SAXS_file_MAICoS,
        "chemical_shift_file": chemical_shift_file,
        "Contact_map_file": Contact_map_file,
        "distance_map_file": distance_map_file,
        "backbone_correlation_file": backbone_correlation_file,
        "rog_file": rog_file,
        "dynamic_landscape_file": dynamic_landscape_file,
        "spin_relaxation_time_file": spin_relaxation_time_file,
    }

    # Check for missing files
    missing_files = [name for name, path in files.items() if not os.path.exists(path)]

    print(missing_files)
    
    if missing_files:
        try:
            MDAuni = system2MDanalysisUniverse(system)
        except:
            pass

        if (not os.path.isfile(gro_fname)):
            execStr = (
                f"echo Protein Protein | {trjconvCOMMAND} -f {trj_fname} "
                f"-s {top_fname} -o {gro_fname} -dump 0 -center -pbc mol"
            )
            os.system(execStr)


    ## Calculate SAXS

    if (not os.path.isfile(SAXS_file)):
        SAXS = calculate_SAXS_profile_crysol(gro_fname, trj_fname,1000)
        #print(SAXS)

        SAXS_data = SAXS.to_dict(orient='records')

        # Save as YAML
        with open(SAXS_file, 'w') as file:
            yaml.dump(SAXS_data, file, sort_keys=False)

            
    if (not os.path.isfile(SAXS_file_MAICoS)):
        try:
            SAXS_MAICoS = calculate_SAXS_profile_maicos(gro_fname, trj_fname)
            #print(SAXS)
    
            data_MAICoS = SAXS_MAICoS.to_dict(orient='records')
    
            # Save as YAML
            with open(SAXS_file_MAICoS, 'w') as file:
                yaml.dump(data_MAICoS, file, sort_keys=False)
        except:
            print("MAICOS CALCULATION FAILED")

    print('Calculate chemical shifts')
    if (not os.path.isfile(chemical_shift_file)):
        chemical_shifts = calculate_ChemShifts_sparta(gro_fname, trj_fname)
        #print(chemical_shifts)

        chemical_shift_data = chemical_shifts.to_dict()

        #print(chemical_shift_data)
        # Save as YAML

        chemical_shift_data_dict = convert_original_to_nested_dict(chemical_shift_data)

        #print(chemical_shift_data_dict)
        
        with open(chemical_shift_file, 'w') as file:
            yaml.dump( chemical_shift_data_dict, file, sort_keys=False)
       
    ## Calculate contact probailities

    print('Calculate contact maps')
    # Define cutoff distance (in Angstroms)
    cutoff = 15.0  # Adjust as needed
    if (not os.path.isfile(Contact_map_file)):
        probabilities_df = calculate_contact_probabilities(gro_fname, trj_fname, cutoff)
        create_contact_map_plot(probabilities_df, Contact_map_file)

    
    print('Calculate distance maps')
    # Calculate distance maps
    if (not os.path.isfile(distance_map_file)):
        probabilities_df = calculate_distances(gro_fname, trj_fname)
        create_distance_map_plot(probabilities_df, distance_map_file)

    print('Calculate backbone correlations')
    ## Calculate backbone correlations
    if (not os.path.isfile(backbone_correlation_file)):
        probabilities_df = calculate_backbone_correlations(gro_fname, trj_fname,backbone_correlation_file)

    print('Calculate radius of gyration')
    ## Calculate radius of gyration
    if (not os.path.isfile(rog_file)):

        if (not os.path.isfile(trj_fname_noPBC)):
            execStr = (
                f"echo Protein Protein | {trjconvCOMMAND} -f {trj_fname} "
                f"-s {top_fname} -o {trj_fname_noPBC} -center -pbc mol"
            )
            os.system(execStr)
        
        execStr = (
            f"echo Protein | gmx gyrate -s {top_fname} -f {trj_fname_noPBC} -o {rog_file}"
        )
        os.system(execStr)

            
    #correlation_function = dataFolder + 'correlation_functions/NHrotaCF_11HIP.xvg'
    #correlation_function_data = read_correlation_function(correlation_function)
    #print(correlation_function_data[1])

    gammaH=267.513*10**6;

    smallest_corr_time=0 # enter in log scale -3 fs; 0 ps; 3 ns; 6 us;
    biggest_corr_time=5 # same as above
    N_exp_to_fit=100 # number of exponential functions to be fitted between the samlles and biggest corr time
    analyze=1/50 # the proportin of correlation data to be used for fitting, ex. 1/2 uses first half of the data
    #magnetic_field=2.35 # 5.99 # 8.49 T (values used in SDS paper, J.Chem. Soc.,, Faraday Trans. 1, 1988, 84(12), 4475-4486)
    ## magn_field=magnetic_field
    ## magnetic_field=magn_field*2*np.pi/gammaH*10**6
    #print(magn_field*2*np.pi/gammaH*10**6)


    #tst = calc_relax_time(correlation_function,magnetic_field,biggest_corr_time,smallest_corr_time,N_exp_to_fit)
    #print(tst[0])

    print('Calculate dynamic landscape')
    if not os.path.exists(dynamic_landscape_file):

        dynamic_landscape = {}


        list_of_correlation_functions = calculate_backbone_NH_correlation_functions(gro_fname,trj_fname,top_fname,dataFolder)        

        
        for correlation_function in list_of_correlation_functions:
            print(correlation_function)
            residue = correlation_function[-10:-4]
            #correlation_function[-9:-4]
            #print(residue[0])
            if residue[0] == "F":
                residue = residue[2:]
            if residue[0] == "_":
                residue = residue[1:]
            print(residue)
            relaxation_results = calc_dynamic_landscape(
                correlation_function,
                biggest_corr_time,
                smallest_corr_time,
                N_exp_to_fit
            )

            # Populate dynamic landscape
            dynamic_landscape[residue] = {
                'tau_eff': relaxation_results[0],
                'weights': relaxation_results[2],
                'timescales': relaxation_results[3]
            }

            #print(dynamic_landscape)


        # Convert NumPy objects to Python-native types
        clean_dynamic_landscape = convert_numpy(dynamic_landscape)
        

        # Dump as readable YAML
        with open(dynamic_landscape_file, 'w') as file:
            yaml.dump(clean_dynamic_landscape, file, sort_keys=True, default_flow_style=False, indent=4)


    if not os.path.exists(spin_relaxation_time_file):
            
        with open(dynamic_landscape_file, "r") as file:
            dynamic_landscape = yaml.safe_load(file)

        
        magnetic_fields = [800, 600]
        #magnetic_field=magnetic_field*2*np.pi/gammaH*10**6
        print('Calculating for ', magnetic_fields, ' MHz magnetic fields')
        spin_relaxation_times = {}    

        #print(dynamic_landscape)

        for magnetic_field in magnetic_fields:
            ### Populate spin relaxation times
            print(magnetic_field)
            print(dynamic_landscape)
            for residue in dynamic_landscape:
                T1, T2, NOE = get_relaxation_N(magnetic_field,dynamic_landscape[residue]['weights'],dynamic_landscape[residue]['timescales'])
                print(T1, T2, NOE)
                if residue not in spin_relaxation_times:
                    spin_relaxation_times[residue] = {}
                spin_relaxation_times[residue][magnetic_field] = {
                    'T1': {
                        'value': T1
                    },
                    'T2': {
                        'value': T2
                    },
                    'hetNOE': {
                        'value': NOE
                    }
                }

        clean_spin_relaxation_times = convert_numpy(spin_relaxation_times)
        print(clean_spin_relaxation_times)


        with open(spin_relaxation_time_file, 'w') as file:
            yaml.dump(clean_spin_relaxation_times, file, sort_keys=True, default_flow_style=False, indent=4)


            
    print('Calculate spin relaxation data RMSD between simulations and experiments')
    spin_relaxation_rmsd_file =  dataFolder + 'spin_relaxation_rmsd.yaml'
    try:
        exp_spin_relax_file = databankPath + '/Data/Experiments/spin_relaxation/' + system['EXPERIMENT']['spin_relaxation']['path'][0] + '/spin_relaxation_times.yaml'
        ExperimentalFile = True
    except:
        print('Experimental spin relaxation data file not found')
        ExperimentalFile = False

    if not os.path.exists(spin_relaxation_rmsd_file) and ExperimentalFile:
        print(spin_relaxation_time_file,exp_spin_relax_file)
        rmsd = calculate_spin_relaxation_time_RMSD(spin_relaxation_time_file,exp_spin_relax_file)
        clean_rmsd = convert_numpy(rmsd)
        with open(spin_relaxation_rmsd_file, 'w') as file:
            yaml.dump(clean_rmsd, file, sort_keys=False, default_flow_style=False, indent=4)

    #continue

    print('Calculate chemical shift RMSD between simulations and experiments')
    chemical_shift_rmsd_file =  dataFolder + 'chemical_shift_rmsd.yaml'

    #print(system['EXPERIMENT'])
    chemical_shift_paths = system['EXPERIMENT']['chemical_shift']['path']
    if len(chemical_shift_paths) > 0:
        for BMRBid in chemical_shift_paths:
            bmrb_local_file = download_NMR_star_file(BMRBid)
            print(bmrb_local_file)
    else:
        print("Trying to get chemical shifts from the spin relaxation data source")
        try:
            BMRBid = system['EXPERIMENT']['spin_relaxation']['path'][0].replace("BMRBid", "", 1)
            bmrb_local_file = download_NMR_star_file(BMRBid)
            ExperimentalFile = True
        except:
            print('Experimental chemical shift data file not found')
            ExperimentalFile = False

    #print('test',  BMRBid)
    if not os.path.exists(chemical_shift_rmsd_file) and ExperimentalFile:
        sim_residues_all = set()
        sim_file = os.path.join(dataFolder, 'chemical_shifts_sparta.yaml')
        with open(sim_file) as f:
            raw = yaml.safe_load(f)
            sim_residues_all.update(int(k) for k in raw.keys())
        

        # ------------------------
        # Parse NMR-STAR file
        # ------------------------

        exp_data = parse_star_file(bmrb_local_file)
        # Only residues present in simulation
        exp_data_filtered = {res: atoms for res, atoms in exp_data.items() if res in sim_residues_all}
        exp_residues = sorted(exp_data_filtered.keys())

        #print(exp_data)
        
        # ------------------------
        # Calculate RMSDs
        # ------------------------
        nuclei = ["C", "CA", "CB", "HA", "H", "N"]

        with open(sim_file) as f:
            raw = yaml.safe_load(f)

        # convert string keys to integers
        sim_data = {int(k): v for k, v in raw.items()}

        # compute RMSD
        chemical_shift_rmsd_vals = compute_rmsd_chemical_shift(sim_data, exp_data_filtered, nuclei, exp_residues)

        #print(rmsd_vals)

        #clean_rmsd = convert_numpy(rmsd)
        with open(chemical_shift_rmsd_file, 'w') as file:
            yaml.dump(chemical_shift_rmsd_vals, file, sort_keys=False, default_flow_style=False, indent=4)
        
        #print(f"RMSDs for {sim_file}:")
        #for nuc, rmsd in rmsd_vals.items():
        #    if rmsd is not None:
        #        print(f"  {nuc}: {rmsd:.2f} ppm")
        #    else:
        #        print(f"  {nuc}: N/A")

        #avg_rmsd = np.mean([v for v in rmsd_vals.values() if v is not None])
        #print(f"  Average RMSD: {avg_rmsd:.2f} ppm\n")

        

        #print(rmsd_vals,avg_rmsd)
        #rmsd = calculate_spin_relaxation_time_RMSD(spin_relaxation_time_file,exp_spin_relax_file)


