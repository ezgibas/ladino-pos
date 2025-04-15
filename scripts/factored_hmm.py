from baum_welch import *
from collections import defaultdict
from viterbi import *

class FactoredHMM(BaumWelch):
    def __init__(self, num_pos_tags, states_per_tag, words):
        """
        num_pos_tags: Number of actual POS tags in data
        states_per_tag: How many HMM states to allocate per tag
        words: List of possible emissions (words)
        """
        self.num_pos_tags = num_pos_tags
        self.states_per_tag = states_per_tag
        total_states = num_pos_tags * states_per_tag
        
        super().__init__(range(total_states), words)
        
        # initial mapping
        self.state_to_tag_mapping = {
            state: state // states_per_tag 
            for state in range(total_states)
        }

    # TODO override save_hmm to save with actual tags and not
    # dummy states
    
    def initialize_from_tagged_data(self, tagged_sequences):
        # First create matrices with small values
        epsilon = 0.01
        self.transition_probs = np.log2(np.full((self.num_tags, self.num_tags), epsilon))
        self.emission_probs = np.log2(np.full((self.num_tags, len(self.vocab)), epsilon))
        self.initial_probs = np.log2(np.full(self.num_tags, epsilon))
                
        # For each expanded state, determine which original tag it corresponds to
        state_to_orig_tag = {state: state // self.states_per_tag for state in range(self.num_tags)}
        
        # Set higher values for transitions between states of the same original tag
        for i in range(self.num_tags):
            orig_tag_i = state_to_orig_tag[i]
            for j in range(self.num_tags):
                orig_tag_j = state_to_orig_tag[j]
                
                # If these states correspond to tags that commonly follow each other
                # set higher transition probability
                if orig_tag_i == orig_tag_j:
                    self.transition_probs[i, j] = np.log2(0.5)  # Higher probability for same tag
                else:
                    self.transition_probs[i, j] = np.log2(0.1)  # Lower probability for different tags
        
        # Normalize all matrices (in log space)
        # For transitions
        for i in range(self.num_tags):
            row_sum = self.logsumexp2(self.transition_probs[i])
            self.transition_probs[i] -= row_sum
        
        # For emissions - just use uniform distribution for now
        for i in range(self.num_tags):
            row_sum = self.logsumexp2(self.emission_probs[i])
            self.emission_probs[i] -= row_sum
        
        # For initial probabilities
        initial_sum = self.logsumexp2(self.initial_probs)
        self.initial_probs -= initial_sum
    
    def _normalize_counts(self, counts_dict):
        """Helper to normalize counts to probabilities"""
        probs_dict = defaultdict(dict)
        for key1, inner_dict in counts_dict.items():
            total = sum(inner_dict.values())
            if total > 0:
                for key2, count in inner_dict.items():
                    probs_dict[key1][key2] = count / total
        return probs_dict
    
    def _normalize_initial_counts(self, counts_dict):
        """Helper to normalize initial counts"""
        total = sum(counts_dict.values())
        if total > 0:
            return {key: count / total for key, count in counts_dict.items()}
        return {}
    
    def update_state_to_tag_mapping(self, tagged_sequences, state_sequences):
        """
        Find the best mapping from HMM states to POS tags
        based on training data frequencies
        """
        
        # Count co-occurrences of states and true tags
        state_tag_counts = defaultdict(lambda: defaultdict(int))
        for state_seq, sequence in zip(state_sequences, tagged_sequences):
            for state, (_, true_tag) in zip(state_seq, sequence):
                state_tag_counts[state][true_tag] += 1
        
        # Assign each state to its most frequent tag
        for state in self.states:
            if state in state_tag_counts:
                tag_counts = state_tag_counts[state]
                self.state_to_tag_mapping[state] = max(
                    tag_counts.keys(), 
                    key=lambda tag: tag_counts[tag]
                )