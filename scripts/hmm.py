import numpy as np
import re
import pickle
from load_preprocessed_data import strip_punctuation
from scipy.special import logsumexp

# rules = [
#     (r'.*(ando|endo)$', 'VERB'), # verbs in gerund
#     (r'.*(ido|ado|ida|ada)$', 'VERB'), # verbs in continuous
#     (r'.*(er|ir|ar)$', 'VERB'), # verbs in infinitive
#     (r'.*(erse|irse|arse)$', 'VERB'), # verbs in infinitive reflexive
#     (r'.*mente$', 'ADV'), # -mente suffix is for adverbs 
#     (r'^-?[0-9]+(.[0-9]+)?\.*$', 'NUM'), # numbers
#     (r'(el|la|un|uno|una)$', 'DET'), # determiners
#     (r'(eya|yo)$', 'PRON'), # pronouns
#     (r'[!\"#\$%&\'\(\)\*\+,\-.\/:;<=>\?@\[\\\]\^_`{\|}~]', 'PUNCT'), # punctuation   
# ]

class HMMTagger:
    def __init__(self, tags, vocab):
        self.states = tags
        self.vocab = vocab
        self.num_tags = len(self.states)  
        self.vocab_size = len(self.vocab)  

        # initialize these variables to 0. should be initialized using initialize_probabilities()
        self.transition_probs = np.zeros((self.num_tags, self.num_tags))
        self.emission_probs = np.zeros((self.num_tags, self.vocab_size))
        self.initial_probs = np.zeros(self.num_tags)

        # initialize hyperparameters
        self.learning_rate = 0.1
        self.decay_rate = 0.9

    def load_hmm(filename="../results/hmm_tagger.pkl"):
        """Load HMM from a file"""
        with open(filename, "rb") as f:
            data = pickle.load(f)
            
        hmm = HMMTagger(
            tags=data["states"],
            vocab=data["vocab"],
        )
        hmm.initialize_probabilities(            
            transition=data["transition_probs"],
            emission=data["emission_probs"],
            initial=data["initial_probs"])
        return hmm
    
    def save_hmm(hmm, filename="../results/hmm_tagger.pkl"):
        """Save HMM to a file"""
        with open(filename, "wb") as f:
            pickle.dump({
                "transition_probs": hmm.transition_probs,
                "emission_probs": hmm.emission_probs,
                "initial_probs": hmm.initial_probs,
                "vocab": hmm.vocab,
                "states": hmm.states
            }, f)
        
    def initialize_probabilities(self, transition, emission, initial):
        """initialize transition, emission, and initial probabilities.
        MUST feed in log probabilities
        """
        self.transition_probs = np.array(transition)
        self.emission_probs = np.array(emission)
        self.initial_probs = np.array(initial)

    def initialize_hyperparameters(self, learning_rate=0.1, decay_rate=0.9):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate



    def train_em(self, sequences, smoothing=1e-3, iterations=10):
        learning_rate = self.learning_rate
        decay_rate = self.decay_rate
        """Train the HMM using the Expectation-Maximization algorithm."""
        # compute expectations over all training sequences
        # one loop = one forward-backward pass
        for iteration in range(iterations):
            print("iteration: " + str(iteration))
            # initialize expectations
            expected_transitions = np.zeros((self.num_tags, self.num_tags)) 
            expected_emissions = np.zeros((self.num_tags, self.vocab_size))
            expected_initials = np.zeros(self.num_tags)
            for sequence in sequences:
                sequence = np.array([strip_punctuation(word.lower()) for word in sequence])
                # forward and backward probabilities
                alpha = self.forward(sequence)
                beta = self.backward(sequence)

                log_prob_sequence =  np.logaddexp.reduce(alpha[-1][:])
                avg_log_prob = log_prob_sequence / len(sequence)

                print("Log probability: " + str(avg_log_prob))

                xis = [] # xi (hidden transitions)
                gammas = [] # gamma (posterior probabilities for states)

                for t in range(len(sequence) - 1):
                    xis.append(self.xi(t, sequence, alpha, beta))
                for t in range(len(sequence)):
                    gammas.append(self.gamma(t, sequence, alpha, beta, xis))

                gammas = np.array(gammas)

                # CALCULATE EXPECTATIONS FOR MATRICES
                # initial probabilities
                for i in range(self.num_tags):
                    expected_initials[i] = gammas[0][i]

                # hidden - hidden transitions
                transitions_num = np.logaddexp.reduce(xis, axis=0)
                transitions_denom = np.logaddexp.reduce(gammas[:-1], axis=0)

                # apply Laplace smoothing early - otherwise values underflow
                transitions_num = np.logaddexp(transitions_num, smoothing)  
                transitions_denom = np.logaddexp(transitions_denom, smoothing) 

                # normalize
                expected_transitions = transitions_num - transitions_denom

                # emissions
                for i in range(self.num_tags):
                    emissions_denom = np.logaddexp.reduce(gammas[:, i])  
                    emissions_denom = np.logaddexp(emissions_denom, smoothing) # smoothing

                    # Loop over unique words in the sequence
                    unique_words = set(sequence)
                    for k in unique_words:
                        mask = sequence == k  # mask for words matching k
                        k_idx = self.vocab.index(k)
                        if np.any(mask):  # if word in sequence
                            emissions_num = np.logaddexp.reduce(gammas[:, i][mask])  
                        else:
                            emissions_num = np.log(1e-6)  

                        # smoothing
                        emissions_num = np.logaddexp(emissions_num, smoothing)
                        # normalize
                        expected_emissions[i, k_idx] = emissions_num - emissions_denom

            # UPDATE 
            # exponentiate 
            self.transition_probs = np.exp(self.transition_probs)
            self.initial_probs = np.exp(self.initial_probs)
            self.emission_probs = np.exp(self.emission_probs)

            expected_emissions = np.exp(expected_emissions)
            expected_initials = np.exp(expected_initials)
            expected_transitions = np.exp(expected_transitions)

            # learn
            self.transition_probs = (1 - learning_rate) * self.transition_probs + learning_rate * expected_transitions
            self.initial_probs = (1 - learning_rate) * self.initial_probs + learning_rate * expected_initials
            self.emission_probs = (1 - learning_rate) * self.emission_probs + learning_rate * expected_emissions

            # normalize
            self.transition_probs = np.log(self.transition_probs / self.transition_probs.sum(axis = 1, keepdims=True))
            self.initial_probs = np.log(self.initial_probs / self.initial_probs.sum())
            self.emission_probs = np.log(self.emission_probs / self.emission_probs.sum(axis = 1, keepdims=True))

            # decay learning rate
            learning_rate = learning_rate * decay_rate

    def forward(self, sequence):
        """Compute forward probabilities (alpha)"""
        sent_length = len(sequence)
        alpha = np.full((sent_length, self.num_tags), -np.inf)

        # initialization step
        first_word = sequence[0]
        word_idx = self.vocab.index(first_word) if first_word in self.vocab else -1

        if word_idx >= 0: # if word is in vocabulary
            alpha[0, :] = self.initial_probs + self.emission_probs[:, word_idx]
        else:
            alpha[0, :] = self.initial_probs + np.log(1e-6)

        # Recursion step
        for t in range(1, sent_length):
            word = sequence[t]
            word_idx = self.vocab.index(word) if word in self.vocab else -1
            for j in range(self.num_tags):
                alpha_sum = alpha[t-1] + self.transition_probs[:, j]
                if word_idx >= 0: # if word is in vocabulary
                    alpha_sum += self.emission_probs[j, word_idx]
                else:
                    alpha_sum += np.log(1e-6)

                alpha[t, j] = np.logaddexp.reduce(alpha_sum)
        return alpha
    
    def backward(self, sequence):
        """Compute backward probabilities (beta)"""
        sent_length = len(sequence)
        beta = np.full((sent_length, self.num_tags), -np.inf)

        # initialize
        last_word = sequence[-1]
        word_idx = self.vocab.index(last_word) if last_word in self.vocab else -1

        if word_idx >= 0: # if word is in vocabulary
            beta[-1, :] = self.initial_probs + self.emission_probs[:, word_idx]
        else:
            beta[-1, :] = self.initial_probs + np.log(1e-6)

        # recursion
        for t in range(sent_length - 2, -1, -1):
            word = sequence[t]  # emission (word) at t
            word_idx =  self.vocab.index(word) if word in self.vocab else -1
            for j in range(self.num_tags):
                # P(transition from j -> any) + beta(t + 1 -> any)
                # note: multiplication is addition in log space
                beta_sum = self.transition_probs[j, :] + beta[t + 1] 
                # add P(next evidence | any)
                if word_idx >= 0: # if word is in vocabulary
                   beta_sum += self.emission_probs[j, word_idx]
                else:
                    beta_sum += np.log(1e-6)
                    
                beta[t, j] = np.logaddexp.reduce(beta_sum)
        return beta

    def xi(self, t, sequence, alpha, beta):
        """Compute xi values, expected transition between the tags"""
        word = sequence[t+1]
        word_idx = self.vocab.index(word) if word in self.vocab else -1 

        xi = np.zeros((self.num_tags, self.num_tags))

        sum_sequence = [] # store logsumexp results

        for i in range(self.num_tags):
            for j in range(self.num_tags):
                emission_prob = (
                    self.emission_probs[j, word_idx] if word_idx >= 0 # if word exists
                    else np.log(1e-6)  # small probability for OOV words
                )

                # sum of log probability
                xi_sum = alpha[t, i] + self.transition_probs[i, j] + emission_prob + beta[t + 1, j]
                sum_sequence.append(xi_sum)
                xi[i, j] = np.logaddexp.reduce(xi_sum)

        # normalization factor
        denom = np.logaddexp.reduce(sum_sequence)

        # normalize
        xi -= denom

        return xi
    
    def gamma(self, t, sequence, alpha, beta, xi_vals):
        """Compute gamma values, posterior probability P(X_i = tag | words so far)"""
        if t >= len(sequence):
            gamma = np.logaddexp.reduce(xi_vals[t], axis=1) 
        else:
            gamma = alpha[t] + beta[t]
        
        gamma_denom = np.logaddexp.reduce(alpha[t] + beta[t]) 
        gamma -= gamma_denom

        return gamma



    def viterbi(self, sequence):
        """Predict best set of tags for a given sentence"""
        sequence = [strip_punctuation(word.lower()) for word in sequence]
        sent_length = len(sequence)  
        N = self.num_tags

        V = np.zeros((sent_length, N))
        B = np.zeros((sent_length, N), dtype=int)

        # initialize
        for i in range(N):
            word = sequence[0]
            emission_prob = self.emission_probs[i, self.vocab.index(word)] if word in self.vocab else 1e-4
            V[0, i] = self.initial_probs[i] * emission_prob
            B[0, i] = 0

        # recursion
        for t in range(1, sent_length):
            for j in range(N):
                word = sequence[t]
                emission_prob = (
                    self.emission_probs[j, self.vocab.index(word)]
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

        best_state_sequence = []
        prev_tag = None
        for word_idx, i in enumerate(best_path):
            tag = ""
            if sequence[word_idx] in self.vocab:
                tag = self.states[i]
            else:
                tag = self.handle_oov(sequence[word_idx], prev_tag)

            tag = self.states[i]
            prev_tag = tag
            best_state_sequence.append(tag)
        
        return best_state_sequence

    def handle_oov(self, word, prev_tag):
        """ Assign a POS tag to an OOV word based on regex rules"""
        # for pattern, tag in rules:
            # if re.fullmatch(pattern, word):  # check if word matches regex pattern
            #     return tag
            # instead of default fallback rule, maximize transition probability
        if prev_tag is not None:
            prev_tag_idx = self.states.index(prev_tag)
            tag_idx =  np.argmax(self.transition_probs[prev_tag_idx])
            return self.states[tag_idx]
        # select tag that is most likely to start a sentence
        else:
            tag_idx = np.argmax(self.initial_probs)
            return self.states[tag_idx]
