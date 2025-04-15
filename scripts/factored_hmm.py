from baum_welch import *
from itertools import product

class FactoredHMM(BaumWelch):
    def __init__(self, factors, possible_values_per_factor, words):
        """
        factors: list of factor names (e.g., ['pos', 'number', 'person'])
        possible_values_per_factor: dict mapping factor names to possible values
        words: list of possible emissions (words)
        """
        self.factors = factors
        self.possible_values = possible_values_per_factor
        
        # Generate all possible states (combinations of factor values)
        self.states = self._generate_all_states()
        super().__init__(self.states, words)
        
    def _generate_all_states(self):
        """generate all possible state combinations from factors"""
        
        # get all possible values for each factor
        factor_values = [self.possible_values[f] for f in self.factors]
        
        # gereate the cartesian product
        all_combinations = product(*factor_values)
        
        # convert to tuple representation
        states = [tuple(comb) for comb in all_combinations]
        return states
    
    def factorize_transition_probs(self):
        """reduce parameter space by factorizing transition probabilities"""
        return
    
    def train_with_em(self, observations, max_iterations=100):
        """Override the EM algorithm to handle factored states"""
        return 
    
    def map_to_simple_tags(self, state_sequence):
        """Map a sequence of factored states to simple POS tags"""
        return [state[0] for state in state_sequence]  