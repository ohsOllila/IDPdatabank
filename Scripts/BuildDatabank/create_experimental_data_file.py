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
        get_data_from_BMRB(args.BMRB)
        get_conditions_from_BMRB(args.BMRB)
        #fasta_seq = extract_bmrb_fasta(args.BMRB)
        sequences = extract_fasta_from_bmrb(args.BMRB)
        #print(sequences)

    else:
        print("No BMRB input provided.")

if __name__ == "__main__":
    main()
