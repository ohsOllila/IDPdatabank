from DatabankLib.protein_functions import *


#get_data_from_BMRB('50115')

import argparse

def main():
    parser = argparse.ArgumentParser(description="Process -BMRB argument.")
    
    # Add a -BMRB argument that expects a string value
    parser.add_argument('-BMRB', type=str, help='Specify the BMRB identifier or file.')

    args = parser.parse_args()

    if args.BMRB:
        print(f"Received BMRB input: {args.BMRB}")
        try:
            get_spin_relaxations_from_BMRB(args.BMRB)
            get_spin_relaxation_conditions_from_BMRB(args.BMRB)
            #fasta_seq = extract_bmrb_fasta(args.BMRB)
            sequences = extract_fasta_from_bmrb(args.BMRB, "spin_relaxation")
            print("Spin relaxations fetched")
        except:
            print("Spin relaxations fetching failed")
            pass

        try:
            sequences = extract_fasta_from_bmrb(args.BMRB, "chemical_shift")
            bmrb_local_file = download_NMR_star_file(args.BMRB)
            exp_data_path = '../../Data/Experiments/chemical_shift/BMRBid' + args.BMRB
            print("Analysing " + bmrb_local_file)
            exp_data = parse_star_file(bmrb_local_file)
            chemical_shift_file = exp_data_path + "/chemical_shifts.yaml"
            with open(chemical_shift_file, 'w') as file:
                yaml.dump(exp_data, file, sort_keys=False, default_flow_style=False, indent=4)
            #print(exp_data)
            chemical_shift_experiment = extract_chemical_shift_experiment_name_from_star(bmrb_local_file)
            print("chemical_shift_experiment", chemical_shift_experiment[0])
            get_chemical_shift_conditions_from_BMRB(args.BMRB,chemical_shift_experiment[0])
            print("Chemical shifts fetched")
        except Exception as e:
            print(e)
            print("Chemical shift fetching failed")
            pass        
    else:
        print("No BMRB input provided.")

if __name__ == "__main__":
    main()
