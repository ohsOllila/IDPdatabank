## python3 create_contact.py 
## python3 create_distance.py
## #python3 $secondary
## python3 correlationCALC.py
## python3 ${path}/Old_Relaxations_for_Samuli.py

from DatabankLib.protein_functions import *
import yaml

databankPath = "/home/sosamuli/work/NMRlipids/IDPsimBank/"  # this is the local path for the cloned Databank
os.environ["NMLDB_ROOT_PATH"] = "/home/sosamuli/work/NMRlipids/IDPsimBank/"


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



    if (not os.path.isfile(gro_fname)):
        execStr = (
            f"echo Protein Protein | {trjconvCOMMAND} -f {trj_fname} "
            f"-s {top_fname} -o {gro_fname} -dump 0 -center -pbc mol"
        )
        os.system(execStr)

        if (not os.path.isfile(trj_fname_noPBC)):
            execStr = (
                f"echo Protein Protein | {trjconvCOMMAND} -f {trj_fname} "
                f"-s {top_fname} -o {trj_fname_noPBC} -center -pbc mol"
            )
            os.system(execStr)


    ## Calculate contact probailities
            
    # Define cutoff distance (in Angstroms)
    cutoff = 15.0  # Adjust as needed
    if (not os.path.isfile(dataFolder + "Contact_map.png")):
        probabilities_df = calculate_contact_probabilities(gro_fname, trj_fname, cutoff)
        create_contact_map_plot(probabilities_df, dataFolder + "Contact_map.png")

    ## Calculate distance maps
    if (not os.path.isfile(dataFolder + "Distance_map.png")):
        probabilities_df = calculate_distances(gro_fname, trj_fname)
        create_distance_map_plot(probabilities_df, dataFolder + "Distance_map.png")

    ## Calculate backbone correlations
    if (not os.path.isfile(dataFolder + "Backbone_correlations.png")):
        probabilities_df = calculate_backbone_correlations(gro_fname, trj_fname,dataFolder + "Backbone_correlations.png")

    ## Calculate radius of gyration
    rog_file = dataFolder + "gyrate.xvg"
    if (not os.path.isfile(rog_file)):
        execStr = (
            f"echo Protein | gmx gyrate -s {top_fname} -f {trj_fname_noPBC} -o {rog_file}"
        )
        os.system(execStr)

    ## Calculate backbone NH bond correlation fucntions


    #BMRBid = '50115'
    #experimental_data_tmp = {
    #    'T1': extract_data_from_BMRB(BMRBid, 'heteronucl_T1_relaxation'),
    #    'T2': extract_data_from_BMRB(BMRBid, 'heteronucl_T2_relaxation'),
    #    'hetNOE': extract_data_from_BMRB(BMRBid, 'heteronucl_NOEs')
    #}

    #magnetic_field = {
    #    'T1': extract_magnetic_field(BMRBid, 'heteronucl_T1_relaxation'),
    #    'T2': extract_magnetic_field(BMRBid, 'heteronucl_T2_relaxation'),
    #    'hetNOE': extract_magnetic_field(BMRBid, 'heteronucl_NOEs')
    #}

    #if not magnetic_field['T1'] ==  magnetic_field['T1'] ==  magnetic_field['T1']:
    #    print('WARNING: magnetic fields of T1, T2, and hetNOE are not equal')

    #print(magnetic_field['T1'], magnetic_field['T1'], magnetic_field['T1'])
    
    #units = {
    #    'T1': extract_units(BMRBid, 'heteronucl_T1_relaxation'),
    #    'T2': extract_units(BMRBid, 'heteronucl_T2_relaxation')
    #}

    #experimental_data = {}
    #for residue in experimental_data_tmp['T1']:
    #    if not residue in experimental_data:
    #        experimental_data[residue] = {magnetic_field['T1']: {}}
    #        print(experimental_data)
    #        if units['T1'] == 'ms':
    #            experimental_data_tmp['T1'][residue]['value'] = 0.001 * experimental_data_tmp['T1'][residue]['value']
    #            experimental_data_tmp['T1'][residue]['error'] = 0.001 * experimental_data_tmp['T1'][residue]['error']
    #            experimental_data[residue][magnetic_field['T1']]['T1'] = {
    #                'value' : experimental_data_tmp['T1'][residue]['value'],
    #                'error' : experimental_data_tmp['T1'][residue]['error'],
    #            }

    #for residue in experimental_data_tmp['T2']:
    #    if not residue in experimental_data:
    #        experimental_data[residue] = {magnetic_field['T2']: {}}
    #    if units['T2'] == 'ms':
    #        experimental_data_tmp['T2'][residue]['value'] = 0.001 * experimental_data_tmp['T2'][residue]['value']
    #        experimental_data_tmp['T2'][residue]['error'] = 0.001 * experimental_data_tmp['T2'][residue]['error']
    #    experimental_data[residue][magnetic_field['T2']]['T2'] = {
    #        'value' : experimental_data_tmp['T2'][residue]['value'],
    #        'error' : experimental_data_tmp['T2'][residue]['error'],
    #    }

    #for residue in experimental_data_tmp['hetNOE']:
    #    if not residue in experimental_data:
    #        experimental_data[residue] = {magnetic_field['hetNOE']: {}}
    #    experimental_data[residue][magnetic_field['T1']]['hetNOE'] = {
    #        'value' : experimental_data_tmp['hetNOE'][residue]['value'],
    #        'error' : experimental_data_tmp['hetNOE'][residue]['error'],
    #    }

#<
    #if units['T2'] == 'ms':
    #    for residue in experimental_data_tmp['T2']:
    #        experimental_data_tmp['T2'][residue]['value'] =  0.001 * experimental_data_tmp['T2'][residue]['value']

    
    #exp_data_path = '../../Data/Experiments/spin_relaxation/BMRBid' + BMRBid
    #if (not os.path.isdir(exp_data_path)):
    #    execStr = (f"mkdir {exp_data_path}")
    #    os.system(execStr)


    # experimental_spin_relaxation_times_file = exp_data_path + '/spin_relaxation_times.yaml'
    # with open(experimental_spin_relaxation_times_file, 'w') as file:
    #     yaml.dump(experimental_data, file, sort_keys=True, default_flow_style=False, indent=4)

        
    list_of_correlation_functions = calculate_backbone_NH_correlation_functions(gro_fname,trj_fname,top_fname,dataFolder)

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

    dynamic_landscape_file = dataFolder + 'dynamic_landscape_Coeffs.yaml'
    spin_relaxation_time_file = dataFolder + 'spin_relaxation_times.yaml'


    if not os.path.exists(dynamic_landscape_file):

        dynamic_landscape = {}

        for correlation_function in list_of_correlation_functions:
            print(correlation_function)
            residue = correlation_function[-9:-4]
            #print(residue[0])
            if residue[0] == "_":
                residue = residue[1:]

            relaxation_results = calc_relax_time(
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


    with open(dynamic_landscape_file, "r") as file:
        dynamic_landscape = yaml.safe_load(file)

    
    magnetic_field = 800
    magnetic_field=magnetic_field*2*np.pi/gammaH*10**6
    print(magnetic_field)
    spin_relaxation_times = {}    

    #print(dynamic_landscape)

    
    ### Populate spin relaxation times
    for residue in dynamic_landscape:
        T1, T2, NOE = get_relaxation_N(magnetic_field,dynamic_landscape[residue]['weights'],dynamic_landscape[residue]['timescales'])
        spin_relaxation_times[residue] = {
            'T1': T1,
            'T2': T2,
            'hetNOE': NOE
        }

    clean_spin_relaxation_times = convert_numpy(spin_relaxation_times)
    print(clean_spin_relaxation_times)


            
    with open(spin_relaxation_time_file, 'w') as file:
        yaml.dump(clean_spin_relaxation_times, file, sort_keys=True, default_flow_style=False, indent=4)


    with open(spin_relaxation_time_file, "r") as file:
        spin_relaxation_times = yaml.safe_load(file)

    differences = {}
    for residue in spin_relaxation_times:
        try:
            differences[residue] = {
                'R1': 1/spin_relaxation_times[residue]['T1'] - 1/experimental_data[residue]['T1']['value'],
                'R2': 1/spin_relaxation_times[residue]['T2'] - 1/experimental_data[residue]['T2']['value'],
                'hetNOE': spin_relaxation_times[residue]['hetNOE'] - experimental_data[residue]['hetNOE']['value']
            }
        except:
            print('Calculation of difference failed for ' + residue)

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

    #print(differences)
    
    print(RMSDs)
