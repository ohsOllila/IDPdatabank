from DatabankLib.protein_functions import *
import yaml
from pathlib import Path

#databankPath = "/home/sosamuli/work/NMRlipids/IDPdatabank/"  # this is the local path for the cloned Databank
#os.environ["NMLDB_ROOT_PATH"] = "/home/sosamuli/work/NMRlipids/IDPdatabank/"

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

#for system in systems:
#    #print(system['FF'])
#    if system['FF'] is not None:
#        continue

#    print(system['TPR'])
    


# Define a regex pattern to capture common FF names
# This assumes the FF name is one of these strings in the filename
ff_pattern = r"(AMBER99SBWS|AMBER99SB-DISP|DESAMBER|CHARMM36M|AMBER03WS|amber99SB-ILDN)"

for system in systems:

    yaml_path = '../../Data/Simulations/' + system['path'] + 'README.yaml'

    yaml_path = Path(yaml_path)
    
    print(yaml_path)
    # Load YAML file
    with yaml_path.open('r') as f:
        data = yaml.safe_load(f)

        
    if system['FF'] is not None and 'calmodulin' not in  system['FF']:
        continue  # skip if FF already exists
    
    # TPR is a nested list
    tpr_files = system['TPR'][0]  # get inner list
    if not tpr_files:
        continue
    
    tpr_filename = tpr_files[0]  # take first file
    match = re.search(ff_pattern, tpr_filename)
    if match:
        data['FF'] = match.group(0)
        #print(system['path'])
    else:
        print(tpr_filename)
        system['FF'] = "UNKNOWN"


    print(data)

    # Save back to YAML
    with yaml_path.open('w') as f:
        yaml.safe_dump(data, f)

    #exit()
        
#for system in systems:
#    print(system['FF'])
