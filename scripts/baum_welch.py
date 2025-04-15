import numpy as np
import pickle

class BaumWelch:
    def __init__(self, tags, words):
        self.states = tags
        self.vocab = words
        self.vocab_lookup = {word: i for i, word in enumerate(words)}
        self.num_tags = len(self.states)  
        self.vocab_size = len(words)  

        # initialize uniform, can be initialized using initialize_probabilities() w/ other values
        transition_probs = np.random.rand(self.num_tags, self.num_tags)
        emission_probs = np.random.rand(self.num_tags, self.vocab_size)
        initial_probs = np.random.rand(self.num_tags)

        # normalize
        transition_probs /= transition_probs.sum(axis=1, keepdims=True)  
        emission_probs /= emission_probs.sum(axis=1, keepdims=True) 
        initial_probs /= initial_probs.sum() 

        self.transition_probs = np.log2(transition_probs)
        self.emission_probs = np.log2(emission_probs)
        self.initial_probs = np.log2(initial_probs)
    
    def save_hmm(hmm, filename="../results/hmm_tagger-BW.pkl"):
        """Save HMM to a file"""
        with open(filename, "wb") as f:
            pickle.dump({
                "transition_probs": np.exp2(hmm.transition_probs),
                "emission_probs": np.exp2(hmm.emission_probs),
                "initial_probs": np.exp2(hmm.initial_probs),
                "vocab": hmm.vocab,
                "states": hmm.states
            }, f)

    def initialize_probabilities(self, transition, emission, initial, log=False):
        """initialize transition, emission, and initial probabilities.
        log: boolean - if the probabilities fed in are in log space or not
        """
        if not log:
            self.transition_probs = np.log2(transition)
            self.emission_probs = np.log2(emission)
            self.initial_probs = np.log2(initial)
        else: 
            self.transition_probs = transition
            self.emission_probs = emission
            self.initial_probs = initial

    def logsumexp2(self, arr):
        arr = np.asarray(arr)  # ensure  it's a numpy array
        if np.all(np.isinf(arr)):
            return -np.inf  # handle case where all values are -inf
        max_ = np.max(arr[~np.isinf(arr)]) if np.any(np.isinf(arr)) else np.max(arr)
        return np.log2(np.sum(2 ** (arr - max_))) + max_


    def train_em(self, sequences, max_iterations=100, learning_rate=0.8, decay_rate=0.9):
        """Train the HMM using the Expectation-Maximization algorithm."""
        # track convergence by looking at log probability
        converged = False
        prev_log_likelihood = None
        iteration = 0
        epsilon = 0.001*len(sequences)
        logprobs = []

        # compute expectations over all training sequences
        # one loop = one forward-backward pass
        while not converged and iteration < max_iterations:
            # initialize probability tables to modify
            acc_transitions_num = np.full((self.num_tags, self.num_tags), -np.inf) 
            acc_emissions_num = np.full((self.num_tags, self.vocab_size), -np.inf)
            acc_initial_num = np.full(self.num_tags, -np.inf)

            # track denominators for normalization
            acc_transition_denom = np.full(self.num_tags, -np.inf)
            acc_emission_denom = np.full(self.num_tags, -np.inf)
            acc_initial_denom = -np.inf

            log_likelihood = 0
            for sequence in sequences:
                if len(sequence) <= 1:
                    log_prob_sequence = 1 # random value so that it doesn't falsely converge
                    continue
                # values for this sequence
                (log_prob_sequence, 
                 seq_acc_transitions_num, 
                 seq_acc_emissions_num, 
                 seq_acc_transition_denom, 
                 seq_acc_emission_denom,
                 seq_acc_initial_num, 
                 seq_acc_initial_denom) = self.baum_welch(sequence)
                

                for i in range(self.num_tags):
                    # print(f"{seq_acc_emissions_num[i]} - {log_prob_sequence}")
                    acc_transitions_num[i] = np.logaddexp2(acc_transitions_num[i], seq_acc_transitions_num[i] - log_prob_sequence)
                    acc_emissions_num[i] = np.logaddexp2(acc_emissions_num[i], seq_acc_emissions_num[i] - log_prob_sequence)
                
                acc_transition_denom = np.logaddexp2(acc_transition_denom, seq_acc_transition_denom - log_prob_sequence)
                acc_emission_denom = np.logaddexp2(acc_emission_denom, seq_acc_emission_denom - log_prob_sequence)

                acc_initial_num = np.logaddexp2(acc_initial_num, seq_acc_initial_num - log_prob_sequence)
                acc_initial_denom = np.logaddexp2(acc_initial_denom, seq_acc_initial_denom - log_prob_sequence)

                log_likelihood += log_prob_sequence
                
            # update the transition and output probability values
            for i in range(self.num_tags):
                # print(f"transition: {acc_transitions_num[i]} - {acc_transition_denom[i]}")
                # print(f"emission: {acc_emissions_num[i]} - {acc_emission_denom[i]}")
                logprob_trans_i = acc_transitions_num[i] - acc_transition_denom[i]
                logprob_ems_i = acc_emissions_num[i] - acc_emission_denom[i]

                # replace any -inf with a value for stability
                logprob_ems_i[np.isinf(logprob_ems_i) & (logprob_ems_i < 0)] = np.log2(1e-10)
                logprob_trans_i[np.isinf(logprob_trans_i) & (logprob_trans_i < 0)] = np.log2(1e-10)

                logprob_trans_i -= self.logsumexp2(logprob_trans_i)
                logprob_ems_i -= self.logsumexp2(logprob_ems_i)
            

                # transition probabilities
                for j in range(self.num_tags):
                    prob_old = 2 ** self.transition_probs[i, j]
                    prob_new = 2 ** logprob_trans_i[j]
                    interpolated_prob = (1 - learning_rate) * prob_old + learning_rate * prob_new
                    self.transition_probs[i, j] = np.log2(interpolated_prob)
                # emission probabilities
                for k in range(self.vocab_size):
                    # self.emission_probs[i, k] = logprob_ems_i[k]
                    prob_old = 2 ** (self.emission_probs[i, k])
                    prob_new = 2 ** logprob_ems_i[k]
                    interpolated_prob =  (1 - learning_rate) * prob_old + learning_rate * prob_new
                    self.emission_probs[i, k] = np.log2(interpolated_prob)
            # initial probabilities
            logprob_initial = acc_initial_num - acc_initial_denom
            # replace -inf for numerical stability
            logprob_initial[np.isinf(logprob_initial) & (logprob_initial < 0)] = np.log2(1e-10)
            # normalize
            logprob_initial -= self.logsumexp2(logprob_initial)

            # update initial probabilities
            for i in range(self.num_tags):
                prob_old = 2 ** (self.initial_probs[i])
                prob_new = 2 ** logprob_initial[i]
                interpolated_prob =  (1 - learning_rate) * prob_old + learning_rate * prob_new
                self.initial_probs[i] = np.log2(interpolated_prob)

            
            # test for convergence
            if iteration > 0 and abs(log_likelihood - prev_log_likelihood) < epsilon:
                converged = True

            print("iteration", iteration, "logprob", log_likelihood/len(sequences))
            # Sanity check that matrix rows sum up to 1
            # # For transition probabilities
            # trans_row_sums = np.array([2 ** self.logsumexp2(self.transition_probs[i]) for i in range(self.num_tags)])

            # # For emission probabilities
            # emissions_row_sums = np.array([2 ** self.logsumexp2(self.emission_probs[i]) for i in range(self.num_tags)])

            # # For initial probabilities
            # initials_sum = 2 ** self.logsumexp2(self.initial_probs)
            # print(trans_row_sums)
            # print(emissions_row_sums)
            # print(initials_sum)
            iteration += 1
            prev_log_likelihood = log_likelihood
            logprobs.append(log_likelihood)
            learning_rate = learning_rate*decay_rate

        return (self, logprobs)

    def baum_welch(self, sequence):  
        """One forward-backward pass"""    
        # forward and backward probabilities
        alpha = self.forward(sequence)
        beta = self.backward(sequence)

        log_prob_sequence = self.logsumexp2(alpha[-1][:])

        # initialize probability tables to modify
        acc_transition_num = np.full((self.num_tags, self.num_tags), -np.inf) 
        acc_emission_num = np.full((self.num_tags, self.vocab_size), -np.inf)
        acc_inital_num = np.full(self.num_tags, -np.inf)

        # track denominators for normalization
        acc_transition_denom = np.full(self.num_tags, -np.inf)
        acc_emission_denom = np.full(self.num_tags, -np.inf)
        acc_initial_denom = -np.inf

        for t in range(len(sequence)):
            word = sequence[t]
            next_word = None
            if t < len(sequence) - 1:
                next_word = sequence[t+1]
                next_word_idx = self.vocab_lookup.get(next_word)
                next_emission_prob = self.emission_probs[:, next_word_idx:next_word_idx+1]
            word_idx = self.vocab_lookup.get(word)

            gamma = alpha[t] + beta[t]    

            if t == 0:
                acc_inital_num = gamma
                acc_initial_denom = self.logsumexp2(gamma)   

            if t < len(sequence) - 1:
                xi = self.transition_probs + next_emission_prob + beta[t+1] + alpha[t].reshape(self.num_tags, 1)
                acc_transition_num = np.logaddexp2(acc_transition_num, xi)
                acc_transition_denom = np.logaddexp2(acc_transition_denom, gamma)
            else:
                acc_emission_denom = np.logaddexp2(acc_transition_denom, gamma)
            
            acc_emission_num[:, word_idx] = np.logaddexp2(acc_emission_num[:, word_idx], gamma)
        
        return (log_prob_sequence,
                acc_transition_num,
                acc_emission_num, 
                acc_transition_denom,
                acc_emission_denom,
                acc_inital_num,
                acc_initial_denom)
    
    def forward(self, sequence):
        """Compute forward probabilities (alpha)"""
        sent_length = len(sequence)
        alpha = np.full((sent_length, self.num_tags), -np.inf)

        # initialization step
        first_word = sequence[0]
        word_idx = self.vocab_lookup.get(first_word) if first_word in self.vocab else -1

        if word_idx >= 0: # if word is in vocabulary
            alpha[0, :] = self.initial_probs + self.emission_probs[:, word_idx]
        else:
            alpha[0, :] = self.initial_probs + np.log2(1e-6)

        # Recursion step
        for t in range(1, sent_length):
            word = sequence[t]
            word_idx = self.vocab_lookup.get(word) if word in self.vocab else -1
            for j in range(self.num_tags):
                output_prob = 0
                alpha_sum = self.logsumexp2(alpha[t-1] + self.transition_probs[:, j])
                if word_idx >= 0: # if word is in vocabulary
                    output_prob = self.emission_probs[j, word_idx]
                else:
                    output_prob = np.log2(1e-6)

                alpha[t, j] = alpha_sum + output_prob
        # replace any -inf for numerical stability
        alpha[np.isinf(alpha) & (alpha < 0)] = np.log2(1e-10)
        return alpha
    
    def backward(self, sequence):
        """Compute backward probabilities (beta)"""
        sent_length = len(sequence)
        beta = np.full((sent_length, self.num_tags), -np.inf)

        # initialize
        last_word = sequence[-1]
        word_idx = self.vocab_lookup.get(last_word) if last_word in self.vocab else -1

        beta[-1, :] = self.initial_probs + np.log2(1e-6)

        # recursion
        for t in range(sent_length - 2, -1, -1):
            word = sequence[t+1]  # emission (word) at t
            word_idx =  self.vocab_lookup.get(word) if word in self.vocab else -1
            for j in range(self.num_tags):
                # P(transition from j -> any) + beta(t + 1 -> any)
                # note: multiplication is addition in log space
                beta_sum = self.transition_probs[j, :] + beta[t + 1] 
                # add P(next evidence | any)
                if word_idx >= 0: # if word is in vocabulary
                   beta_sum += self.emission_probs[j, word_idx]
                else:
                    beta_sum += np.log2(1e-6)
                    
                beta[t, j] = self.logsumexp2(beta_sum)

        beta[np.isinf(beta) & (beta < 0)] = np.log2(1e-10)
        return beta