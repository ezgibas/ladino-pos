"""
various functions to load data saved locally
so that POS tagging/alignment only has to be run once
and the outputs are preserved across sessions
"""

import ast

def load_parallel_data(file_path):
    ladino_tokens = []
    spanish_tokens = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            ladino, spanish = line.strip().split(" ||| ")  # Split at delimiter
            ladino_tokens.append(ladino.split())  # Convert to token list
            spanish_tokens.append(spanish.split())  # Convert to token list
    return ladino_tokens, spanish_tokens

def load_alignments(file_path):
    """Reads a FastAlign alignment file and converts each line into a list of tuples."""
    alignments = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # Split the line by spaces, then split each alignment pair by '-'
            sentence_alignments = [tuple(map(int, pair.split('-'))) for pair in line.strip().split()]
            alignments.append(sentence_alignments)
    
    return alignments

def load_spanish_pos(file_path):
    spanish_pos = []
    with open(file_path, "r") as f:
        spanish_pos = [ast.literal_eval(line.strip()) for line in f]
    return spanish_pos

def load_ladino_pos(file_path):
    return "" # TODO