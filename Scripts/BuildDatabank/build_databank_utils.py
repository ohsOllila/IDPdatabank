from typing import Dict, List, Optional, Union, Any

from Bio import Align
from Bio.Align import substitution_matrices


def align_and_evaluate_protein_sequences(
    seq1_str: str,
    seq2_str: str,
    open_gap_score: int = -10,
    extend_gap_score: int = -0.5,
):
    """
    Aligns two protein sequences and provides details about the alignment quality.

    Args:
        seq1_str (str): The first protein sequence as a string.
        seq2_str (str): The second protein sequence as a string.
    """

    # 1. Create an aligner object
    # For protein alignment, it's best to use a substitution matrix.
    # BLOSUM62 is a common choice for proteins.
    # We'll also define gap penalties.
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")

    # Set gap penalties (these values are typical, but you can adjust them)
    # - open_gap_score: Penalty for opening a new gap
    # - extend_gap_score: Penalty for extending an existing gap
    aligner.open_gap_score = open_gap_score
    aligner.extend_gap_score = extend_gap_score

    # You can choose between global (Needleman-Wunsch) and local (Smith-Waterman) alignment.
    # Global alignment (default for PairwiseAligner) attempts to align the entire sequences.
    # Local alignment finds the best matching sub-regions.
    aligner.mode = "local"

    # 2. Perform the alignment
    # The align method returns an iterator of Alignment objects.
    # There might be multiple equally good alignments. We'll just take the first one.
    alignments = list(aligner.align(seq1_str, seq2_str))

    if not alignments:
        warnings.warn(f"No alignment found for sequences {seq1_str} and {seq2_str}.")
        return None

    # Get the best (first) alignment
    best_alignment = alignments[0]

    return best_alignment
