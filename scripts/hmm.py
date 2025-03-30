import numpy as np
from collections import defaultdict
import re

rules = [
    (r'.*(ando|endo)$', 'VERB'), # verbs in gerund
    (r'.*(ido|ado|ida|ada)$', 'VERB'), # verbs in continuous
    (r'.*(er|ir|ar)$', 'VERB'), # verbs in infinitive
    (r'.*(erse|irse|arse)$', 'VERB'), # verbs in infinitive reflexive
    (r'.*mente$', 'ADV'), # -mente suffix is for adverbs 
    (r'^-?[0-9]+(.[0-9]+)?\.*$', 'NUM'), # numbers
    (r'(un|uno|una)$', 'DET'), # determiners
    (r'(el|El|eya|Eya|Yo|yo)$', 'PRON'), # pronouns
    (r'[!\"#\$%&\'\(\)\*\+,\-.\/:;<=>\?@\[\\\]\^_`{\|}~]', 'PUNCT'), # punctuation   
    (r'\b[A-Z].*?\b', 'PROPN') # proper nouns (capitalized)  
]

class HMMTagger:
    def __init__(self, tags, vocab, smoothing=1.0):
        self.states = tags
        self.vocab = vocab
        self.num_tags = len(self.states)  
        self.vocab_size = len(self.vocab)  
        self.smoothing = smoothing 

        # initialize these variables to 0. should be initialized using initialize_probabilities()
        self.transition_probs = np.zeros((self.num_tags, self.num_tags))
        self.emission_probs = np.zeros((self.vocab_size, self.num_tags))
        self.initial_probs = np.zeros(self.num_tags)
        
    def initialize_probabilities(self, transition, emission, initial):
        """initialize transition, emission, and initial probabilities"""
        self.transition_probs = transition
        self.emission_probs = emission
        self.initial_probs = initial

        # normalize if needed
        self.transition_probs /= self.transition_probs.sum(axis=1, keepdims=True)
        self.emission_probs /= self.emission_probs.sum(axis=1, keepdims=True)
        self.initial_probs /= self.initial_probs.sum()

    def apply_laplace_smoothing(self):
        """apply Laplace smoothing to handle OOV words better"""
        self.transition_probs += self.smoothing
        self.transition_probs /= self.transition_probs.sum(axis=1, keepdims=True)
        
        self.emission_probs += self.smoothing
        self.emission_probs /= self.emission_probs.sum(axis=1, keepdims=True)

    def train_em(self, sequences, num_iterations=10):
        """Train HMM using EM algorithm"""
        for _ in range(num_iterations):
            # expected counts
            expected_transitions = np.zeros((self.num_tags, self.num_tags))
            expected_emissions = np.zeros((self.vocab_size, self.num_tags))
            expected_initials = np.zeros(self.num_tags)

            for sequence in sequences:
                alpha = self.forward(sequence)  # forward probabilities
                beta = self.backward(sequence)  # backward probabilities
                xi, gamma = self.compute_expectations(sequence, alpha, beta)

                expected_transitions += xi.sum(axis=0)
                for t, word in enumerate(sequence):
                    word_idx = self.vocab.index(word) if word in self.vocab else -1
                    if word_idx != -1:
                        expected_emissions[word_idx] += gamma[t]

                expected_initials += gamma[0]

            # normalize to update parameters
            self.transition_probs = expected_transitions / expected_transitions.sum(axis=1, keepdims=True)
            self.emission_probs = expected_emissions / expected_emissions.sum(axis=1, keepdims=True)
            self.initial_probs = expected_initials / expected_initials.sum()

        self.apply_laplace_smoothing()


    def forward(self, sequence):
        """Compute forward probabilities (alpha)"""
        sent_length = len(sequence)
        alpha = np.zeros((sent_length, self.num_tags))

        # initialization step
        word_idx = self.vocab.index(sequence[0]) if sequence[0] in self.vocab else -1
        if word_idx != -1:
            alpha[0] = self.initial_probs * self.emission_probs[word_idx]
        else:
            alpha[0] = self.initial_probs * (1 / self.vocab_size)  #  for handling oov words

        # recusrsion
        for t in range(1, sent_length):
            word_idx = self.vocab.index(sequence[t]) if sequence[t] in self.vocab else -1
            if word_idx != -1:
                emission = self.emission_probs[word_idx]
            else:
                emission = 1 / self.vocab_size  # for handling oov

            alpha[t] = (alpha[t-1] @ self.transition_probs) * emission

        return alpha

    def backward(self, sequence):
        """Compute backward probabilities (beta)"""
        sent_length = len(sequence)
        beta = np.zeros((sent_length, self.num_tags))

        # initialize
        beta[-1] = 1  

        # recursion
        for t in range(sent_length - 2, -1, -1):
            word_idx = self.vocab.index(sequence[t+1]) if sequence[t+1] in self.vocab else -1
            if word_idx != -1:
                emission = self.emission_probs[word_idx]
            else:
                emission = 1 / self.vocab_size  # handle oov

            beta[t] = self.transition_probs @ (beta[t+1] * emission)

        return beta

    def compute_expectations(self, sequence, alpha, beta):
        """
        Compute the expected counts of transitions (xi) and states (gamma)
        for a given sequence using Forward-Backward probabilities.
        """

        sent_length = len(sequence)  # lengh
        num_tags = self.num_tags 

        xi = np.zeros((sent_length - 1, num_tags, num_tags))  # transition expectations
        gamma = np.zeros((sent_length, num_tags))  # state expectations

        # probability of whole sequence
        prob_sequence = np.sum(alpha[-1]) 

        # gamma
        for t in range(sent_length):
            gamma[t] = (alpha[t] * beta[t]) / prob_sequence  # normlaize


        # xi
        for t in range(sent_length - 1):
            word_index = self.vocab.index(sequence[t + 1]) if sequence[t + 1] in self.vocab else -1
            for i in range(num_tags):
                for j in range(num_tags):
                    # add smoothing for unseen words
                    # TODO use rules 
                    emission_prob = (self.emission_probs[word_index, j] 
                                     if word_index != -1 
                                     else self.smoothing / (self.vocab_size + self.smoothing))
                    xi[t, i, j] = (
                        alpha[t, i] 
                        * self.transition_probs[i, j] 
                        * emission_prob  
                        * beta[t + 1, j]
                    )

            # normalize xi
            xi[t] /= np.sum(xi[t])

        return xi, gamma

    
    def viterbi(self, sequence):
        """Predict best set of tags for a given sentence"""
        sent_length = len(sequence)  
        N = self.num_tags

        V = np.zeros((sent_length, N))
        B = np.zeros((sent_length, N), dtype=int)

        # initialize
        for i in range(N):
            word = sequence[0]
            emission_prob = self.emission_probs[self.vocab.index(word), i] if word in self.vocab else 1e-4
            V[0, i] = self.initial_probs[i] * emission_prob
            B[0, i] = 0

        # recursion
        for t in range(1, sent_length):
            for j in range(N):
                word = sequence[t]
                emission_prob = (
                    self.emission_probs[self.vocab.index(word), j]
                    if word in self.vocab
                    else 1e-4 # Small probability for OOV
                )
                probabilities = V[t-1] * self.transition_probs[:, j] * emission_prob
                V[t, j] = np.max(probabilities)
                B[t, j] = np.argmax(probabilities)

        # backtracking + termination
        best_path = np.zeros(sent_length, dtype=int)
        best_path[-1] = np.argmax(V[-1])

        for t in range(sent_length-2, -1, -1):
            best_path[t] = B[t+1, best_path[t+1]]

        best_state_sequence = [
            self.states[i] if sequence[t] in self.vocab else self.handle_oov(sequence[t])
            for t, i in enumerate(best_path)
        ]
        
        return best_state_sequence

    def handle_oov(self, word):
        """ Assign a POS tag to an OOV word based on regex rules"""
        for pattern, tag in rules:
            if re.fullmatch(pattern, word):  # check if word matches regex pattern
                return tag
        return "NOUN"  # Default fallback if no rule matches





