from fairmd.idp.protein_functions import *
import yaml
from fairmd.idp import NMLDB_SIMU_PATH

# These two lines include core Databank routines and Databank API
from fairmd.idp.core import *
from fairmd.idp.databankLibrary import *
# This is for plotting
from fairmd.idp.databankio import *
from fairmd.idp.plottings import plotSimulation
#from IPython.display import display, Markdown

# This initializes the databank and stores the information of all simulations into a list.
# Each list item contains the information from README.yaml file of the given simulation.
systems = initialize_databank()


trjconvCOMMAND = 'gmx trjconv'

for system in systems:

    dataFolder = os.path.join(NMLDB_SIMU_PATH, system['path'])
    
    spin_relaxation_quality_file =  dataFolder + 'spin_relaxation_quality.yaml'
    spin_relaxation_res_quality_file =  dataFolder + 'spin_relaxation_res_quality.yaml'
    if os.path.exists(spin_relaxation_quality_file):
        compute_residue_nonzero_percentages(spin_relaxation_quality_file, spin_relaxation_res_quality_file)
        
    chemical_shift_quality_file =  dataFolder + 'chemical_shift_quality.yaml'
    chemical_shift_res_quality_file =  dataFolder + 'chemical_shift_res_quality.yaml'
    if os.path.exists(chemical_shift_quality_file):
        compute_residue_nonzero_percentages(chemical_shift_quality_file, chemical_shift_res_quality_file)
