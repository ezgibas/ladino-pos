class Predictor:
    def __init__(self, tags, transitions, emissions, initial):
        self.tags = tags
        self.transitions = transitions
        self.emissions = emissions
        self.initial = initial

    def viterbi(self, sequence):
        """
        Predict the Part of Speech tags for a given sentence using the Viterbi algorithm
        """
        tags = self.tags
        transitions = self.transitions
        emissions = self.emissions
        initial = self.initial

        # viterbi matrix
        # V[t][i] = value of path with the highest probability that accounts for the first t observations
        V = [{}] 
        # path matrix
        # path[t][i] = path w/ highest probability that accounts for first t observations
        path = [{}]

        # i.e. V is the max(), path is the argmax() 
        
        # initialize first step
        for state in tags:
            # handle OOV words w/ small probability
            emission_prob = emissions[state].get(sequence[0], 1e-5)
            V[0][state] = initial[state] * emission_prob
            path[0][state] = [state]
        
        # recursion
        for t in range(1, len(sequence)):
            V.append({})
            path.append({})
            
            for cur_state in tags:
                # handle OOV
                emission_prob = emissions[cur_state].get(sequence[t], 1e-5)
                
                # initialize max, argmax to nothing
                max_prob = float('-inf')
                max_state = None
                
                # get max, argmax of V[t-1][i]*transitions[i][j] over all states i
                for prev_state in tags: # for all states i
                    # smoothing for missing transitions
                    transition_prob = transitions[prev_state].get(cur_state, 1e-5)

                    # V[t-1][i]*transitions[i][j]
                    prob = V[t-1][prev_state] * transition_prob * emission_prob
                    
                    # max, argmax
                    if prob > max_prob:
                        max_prob = prob
                        max_state = prev_state
                
                # V[t][j] = max(V[t-1][i]*transitions[i][j])*emissions[j][observation_t]
                V[t][cur_state] = max_prob
                # path[t][j] = argmax(V[t-1][i]*transitions[i][j])*emissions[j][observation_t]
                path[t][cur_state] = path[t-1][max_state] + [cur_state]
                    

        # termination + backtracking
        T = len(sequence)-1 # T = last time-step

        best_final_state = max(V[T], key=V[T].get)

        best_path = path[T][best_final_state] # path stores completes paths, so just access the last one

        return best_path