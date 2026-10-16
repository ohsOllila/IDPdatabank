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



#from Bio.Blast import NCBIWWW
#from Bio.Blast import NCBIXML



import time
import requests

EBI_BASE = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast"


def blast_sequence(sequence, top_n=5, email="samuli.ollila@vtt.fi",
                   low_complexity_filter=False, poll_interval=5, timeout=900):
    """
    Run BLAST against UniProtKB (Swiss-Prot + TrEMBL) and PDB via EMBL-EBI.

    Returns:
        dict with keys:
            - 'uniprot': list of hits
            - 'pdb': list of hits
    """
    seq_len = len(sequence)

    def submit(database):
        r = requests.post(f"{EBI_BASE}/run", data={
            "email": email,
            "program": "blastp",
            "stype": "protein",
            "sequence": sequence,
            "database": database,
            "alignments": 50,
            "scores": 50,
            "exp": "10",
            "filter": "T" if low_complexity_filter else "F",
        })
        if not r.ok:
            raise RuntimeError(f"EBI rejected the job ({r.status_code}): {r.text}")
        return r.text.strip()
    

    def wait(job_id):
        start = time.time()
        while True:
            status = requests.get(f"{EBI_BASE}/status/{job_id}").text.strip()
            if status == "FINISHED":
                return
            if status in ("FAILED", "ERROR", "NOT_FOUND"):
                raise RuntimeError(f"BLAST job {job_id} ended with status {status}")
            if time.time() - start > timeout:
                raise TimeoutError(f"BLAST job {job_id} timed out")
            time.sleep(poll_interval)

    def parse_hits(job_id):
        r = requests.get(f"{EBI_BASE}/result/{job_id}/json")
        r.raise_for_status()
        data = r.json()

        hits = []
        for hit in data.get("hits", [])[:top_n]:
            hsp = hit["hit_hsps"][0]             # best HSP per hit
            align_len = int(hsp["hsp_align_len"])
            q_span = abs(int(hsp["hsp_query_to"]) - int(hsp["hsp_query_from"])) + 1
            hits.append({
                "accession": hit.get("hit_acc"),
                "title": f'{hit.get("hit_id")} {hit.get("hit_desc", "")}'.strip(),
                "length": int(hit["hit_len"]),
                "identity_%": round(float(hsp["hsp_identity"]), 2),
                "similarity_%": round(float(hsp["hsp_positive"]), 2),
                "coverage_%": round(q_span / seq_len * 100, 2),
                "e_value": float(hsp["hsp_expect"]),
                "bit_score": float(hsp["hsp_bit_score"]),
                "align_length": align_len,
            })
        return hits

    print("Submitting UniProtKB and PDB searches")
    uniprot_job = submit("uniprotkb")
    pdb_job = submit("pdb")

    print("Analyzing UniProt")
    wait(uniprot_job)
    uniprot_hits = parse_hits(uniprot_job)

    print("Analyzing PDB")
    wait(pdb_job)
    pdb_hits = parse_hits(pdb_job)

    return {"uniprot": uniprot_hits, "pdb": pdb_hits}



for system in systems:

    path = os.path.join(NMLDB_SIMU_PATH, system['path'])
    output_file = path + 'cross_link_data.yaml'

    if os.path.isfile(output_file):
        print(output_file, ' found')
        continue

    print('generating: ', output_file)
    
    sequence = system['COMPOSITION']['PROTEIN']['SEQUENCE']
    print(sequence)
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
