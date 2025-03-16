"""
various functions to load data saved locally
so that POS tagging/alignment only has to be run once
and the outputs are preserved across sessions
"""

import ast
import re
from collections import Counter, defaultdict

class Token:
    def __init__(self, word, pos):
        self.word = word
        self.pos = pos
    
    def get_word(self):
        return self.word
    
    def get_pos(self):
        return self.pos
    
    def correct_pos(self, pos):
        self.pos = pos
    
    def __str__(self):
        return self.word + " (" + self.pos + ")"

def load_parallel_data(file_path):
    ladino_tokens = []
    spanish_tokens = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            ladino, spanish = line.strip().split(" ||| ")  # split at delimiter
            ladino_tokens.append(ladino.split())  # convert to token list
            spanish_tokens.append(spanish.split())  # convert to token list
    return ladino_tokens, spanish_tokens

def load_alignments(file_path):
    """Reads a FastAlign alignment file and converts each line into a list of tuples."""
    alignments = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # split the line by spaces, then split each alignment pair by '-'
            sentence_alignments = [tuple(map(int, pair.split('-'))) for pair in line.strip().split()]
            alignments.append(sentence_alignments)
    
    return alignments

def load_spanish_pos(file_path):
    spanish_pos = []
    with open(file_path, "r") as f:
        spanish_pos = [ast.literal_eval(line.strip()) for line in f]
    return spanish_pos

def load_ladino_pos(file_path):
    print(file_path)
    ladino_pos_sentences = []
    word_tags = defaultdict(Counter)  # {word: {tag1: count, tag2: count, ...}}

    with open(file_path, "r") as f:
        for line in f:
            sentence = []  # list to store current sentence's words
            # regex to find word (tag)
            matches = re.findall(r"(\S+)\s*\((\S+)\)", line.strip())

            for word, tag in matches:
                sentence.append(Token(word, tag))
                if tag != "UNK":  # don't store UNK tag in the dictionary
                    word_tags[word][tag] += 1  # count occurrences of each tag
        
            ladino_pos_sentences.append(sentence)  
    return ladino_pos_sentences, word_tags
