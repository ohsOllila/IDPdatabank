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



from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML

def blast_sequence(sequence, top_n=5):
    """
    Run BLAST against UniProt (SwissProt) and PDB.
    
    Returns:
        dict with keys:
            - 'uniprot': list of hits
            - 'pdb': list of hits
    """

    def parse_hits(blast_record, seq_len):
        hits = []

        for alignment in blast_record.alignments[:top_n]:
            for hsp in alignment.hsps:
                identity = (hsp.identities / hsp.align_length) * 100
                similarity = (hsp.positives / hsp.align_length) * 100
                coverage = (hsp.align_length / seq_len) * 100

                hit = {
                    "title": alignment.title,
                    "length": alignment.length,
                    "identity_%": round(identity, 2),
                    "similarity_%": round(similarity, 2),
                    "coverage_%": round(coverage, 2),
                    "e_value": hsp.expect,
                    "bit_score": hsp.bits
                }

                hits.append(hit)

        return hits

    seq_len = len(sequence)

    # --- UniProt (SwissProt) ---
    sp_handle = NCBIWWW.qblast("blastp", "swissprot", sequence)
    sp_record = NCBIXML.read(sp_handle)
    sp_hits = parse_hits(sp_record, seq_len)

    # --- PDB ---
    pdb_handle = NCBIWWW.qblast("blastp", "pdb", sequence)
    pdb_record = NCBIXML.read(pdb_handle)
    pdb_hits = parse_hits(pdb_record, seq_len)

    return {
        "uniprot": sp_hits,
        "pdb": pdb_hits
    }

for system in systems:

    path = databankPath + '/Data/Simulations/' + (system['path'])
    output_file = path + 'cross_link_data.yaml'

    if os.path.isfile(output_file):
        print(output_file, ' found')
        continue

    print('generating: ', output_file)
    
    sequence = system['COMPOSITION']['PROTEIN']['SEQUENCE']
    results = blast_sequence(sequence)

    print("\n--- UniProt hits ---")
    for hit in results["uniprot"]:
        print(hit)

    print("\n--- PDB hits ---")
    for hit in results["pdb"]:
        print(hit)

    with open(output_file, "w") as f:
        yaml.dump(results, f, sort_keys=False)

    print(output_file, ' Saved')
