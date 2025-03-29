import numpy as np
from collections import defaultdict
import re

# TODO how do i even incorporate this
rules = [
    (r'.*(and'), # pronounso|endo)$', 'VERB'), # verbs in gerund
    (r'.*(ido|ado|ida|ada)$', 'VERB'), # verbs in continuous
    (r'.*(er|ir|ar)$', 'VERB'), # verbs in infinitive
    (r'.*(erse|irse|arse)$', 'VERB'), # verbs in infinitive reflexive
    (r'.*mente$', 'ADV'), # -mente suffix is for adverbs 
    (r'^-?[0-9]+(.[0-9]+)?\.*$', 'NUM'), # numbers
    (r'(el|El|eya|Eya|Yo|yo)$', 'PRON'), # pronouns
    (r'[!\"#\$%&\'\(\)\*\+,\-.\/:;<=>\?@\[\\\]\^_`{\|}~]', 'PUNCT'), # punctuation   
    (r'\b[A-Z].*?\b', 'PROPN') # proper nouns (capitalized)  
]

class HMMTagger:
    def __init__(self, num_tags, vocab_size, smoothing=1.0):
        self.num_tags = num_tags  
        self.vocab_size = vocab_size  
        self.smoothing = smoothing 

        # initialize these variables to 0. should be initialized using initialize_probabilities()
        self.transition_probs = np.zeros(num_tags, num_tags)
        self.emission_probs = np.zeros(num_tags, vocab_size)
        self.initial_probs = np.zeros(num_tags)
        
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
        """Train HMM using EM for unsupervised learning."""
        for _ in range(num_iterations):
            expected_transitions = np.zeros_like(self.transition_probs)
            expected_emissions = np.zeros_like(self.emission_probs)
            expected_initials = np.zeros_like(self.initial_probs)

            for sequence in sequences:
                alpha = self.forward(sequence)  # Forward probabilities
                beta = self.backward(sequence)  # Backward probabilities
                xi, gamma = self.compute_expectations(sequence, alpha, beta)

                expected_transitions += xi.sum(axis=0)
                for t in range(len(sequence)):
                    expected_emissions[:, sequence[t]] += gamma[t]

                expected_initials += gamma[0]

            # Normalize to update parameters
            self.transition_probs = expected_transitions / expected_transitions.sum(axis=1, keepdims=True)
            self.emission_probs = expected_emissions / expected_emissions.sum(axis=1, keepdims=True)
            self.initial_probs = expected_initials / expected_initials.sum()

        self.apply_laplace_smoothing()

    def forward(self, sequence):
        """Computer forward probabilities (alpha) for EM algorithm"""
        T = len(sequence)
        N = self.num_tags
        alpha = np.zeros((T, N))

        # initialize
        for i in range(N):
            alpha[0, i] = self.initial_probs[i] * self.emission_probs[i, sequence[0]]

        # recursion
        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = np.sum(alpha[t-1] * self.transition_probs[:, j]) * self.emission_probs[j, sequence[t]]

        return alpha


    def backward(self, sequence):
        """Compute backward probabilities (beta) for EM algorithm"""
        T = len(sequence)
        N = len(self.states)
        beta = np.zeros((T, N))

        # initialize
        beta[T-1] = 1  

        # recursion
        for t in range(T-2, -1, -1):
            for i in range(N):
                beta[t, i] = np.sum(self.transition_probs[i] * self.emission_probs[:, sequence[t+1]] * beta[t+1])

        return beta

    def compute_expectations(self, sequence, alpha, beta):
        """Compute expectations using forward and backward probabilities."""
        T = len(sequence)
        N = len(self.states)

        xi = np.zeros((T-1, N, N))  # Expected transition counts
        gamma = np.zeros((T, N))    # Expected state probabilities

        for t in range(T-1):
            denominator = np.sum(alpha[t] * self.transition_probs * self.emission_probs[:, sequence[t+1]] * beta[t+1])
            for i in range(N):
                for j in range(N):
                    xi[t, i, j] = (alpha[t, i] * self.transition_probs[i, j] * 
                                self.emission_probs[j, sequence[t+1]] * beta[t+1, j]) / denominator
            gamma[t] = np.sum(xi[t], axis=1)

        gamma[T-1] = alpha[T-1] / np.sum(alpha[T-1])  # Last state

        return xi, gamma

    
    def viterbi(self, sequence):
        """Tag a given sequence using viterbi"""
        T = len(sequence)
        V = np.zeros((T, self.num_tags))  # Viterbi table
        backpointers = np.zeros((T, self.num_tags), dtype=int)

        # initialization
        V[0] = np.log(self.initial_probs) + np.log(self.emission_probs[:, sequence[0]])

        # recursion
        for t in range(1, T):
            for s in range(self.num_tags):
                prob = V[t-1] + np.log(self.transition_probs[:, s]) + np.log(self.emission_probs[s, sequence[t]])
                V[t, s] = np.max(prob)
                backpointers[t, s] = np.argmax(prob)

        # backtrack
        best_path = [np.argmax(V[-1])]
        for t in range(T-1, 0, -1):
            best_path.insert(0, backpointers[t, best_path[0]])

        return best_path
    

    # TODO how do i even incorporate this
    def handle_oov(word):
        """ Assign a POS tag to an OOV word based on regex rules. """
        for pattern, tag in rules:
            if re.fullmatch(pattern, word):  # check if word matches regex pattern
                return tag
        return "NOUN"  # Default fallback if no rule matches





