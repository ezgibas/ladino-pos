import numpy as np
from load_data import *

def create_count_dictionaries(data):
    """
    Count tags, tag transitions, and emissions of words to create the proper probability tables:
    P(Tag_{i} | Tag_{i-1})
    P(Word | Tag)
    Note: DOES NOT RETURN PROBABILITIES. Returns counts that can be used to calculate probabilities
    """
    tag_counts = {} # P(Tag)
    tag_transition_counts = {} # P(Tag_{i} | Tag_{i-1})
    # go through each sentence in the data
    for sentence in data:
        tags_sequence = [word.get_pos() for word in sentence]
        words_sequence = [word.get_word() for word in sentence]
        prev_tag = "<s>" # all sentences start with delimiter
        # go through each word and tag
        for _, tag in zip(words_sequence, tags_sequence):
            # P(Tag)
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

            # P(Tag_{i} | Tag_{i-1})
            tag_transition = (prev_tag, tag) # make key to indicate transitioning from the previous tag to current
            tag_transition_counts[tag_transition] = tag_transition_counts.get(tag_transition, 0) + 1
            prev_tag = tag
        
        # P(Tag_{i} | Tag_{i-1}) only for the end of the sentence
        tag_transition = (prev_tag, "<s/>") # all sentences end with delimiter
        tag_transition_counts[tag_transition] = tag_transition_counts.get(tag_transition, 0) + 1
    return tag_counts, tag_transition_counts

def word_to_tag_counts(data):
    """Creates a dictionary in the form:
    {word: {tag1: count, tag2: count, ...}}
    This is useful when creating the probability matrices
    """
    word_tags = defaultdict(Counter)  # {word: {tag1: count, tag2: count, ...}}
    for sentence in data:
        for token in sentence:
            word = token.get_word()
            tag = token.get_pos()
            word = word.lower()
            word_tags[word][tag] += 1  # count occurrences of each tag
    words = list(word_tags.keys())
    return words, word_tags

def create_probability_matrices(words, tags, tag_counts, tag_transition_counts, word_to_tag_counts):
    """
    Turn counts into matrices (numpy array). Creates the following tables:
    P(Tag_{i} | Tag_{i-1})
    P(Word | Tag)
    P(Tag)
    """
    tags_dict = word_to_tag_counts
    num_tags = len(tags)
    # columns are "tags" defined in previous cell

    # create mapping of words and tags to an index so that we can
    # add to the correct tag/word every time we are updating the matrix5
    word_to_index = {word: i for i, word in enumerate(words)}
    tag_to_index = {tag: j for j, tag in enumerate(tags)}

    # P(Tag_i | Tag_{i-1}) matrix
    transition_matrix = np.zeros((num_tags, num_tags))
    for tag_1 in tags:
        for tag_2 in tags:
            i = tag_to_index[tag_1]
            j = tag_to_index[tag_2]
            count_of_transition = tag_transition_counts.get((tag_1, tag_2), 0)
            transition_matrix[i, j] = count_of_transition/tag_counts.get(tag_1)


    transition_matrix = np.where(transition_matrix == 0.0, 1e-6, transition_matrix)
    transition_matrix = np.log(transition_matrix)


    # P(Word | Tag) emission matrix
    emission_matrix = np.zeros((len(tags), len(words)))

    for word, counter in tags_dict.items():
        for tag, count in counter.items():
            emission_matrix[tag_to_index[tag], word_to_index[word]] = count


    emission_matrix = emission_matrix / emission_matrix.sum(axis=1, keepdims=True)

    emission_matrix = np.where(emission_matrix == 0.0, 1e-6, emission_matrix)
    emission_matrix = np.log(emission_matrix)


    # Initial probabilities 
    initial_probs = np.zeros(len(tags))
    for i in range(len(tags)):
        prob = tag_transition_counts.get(('<s>', tags[i]), 0)
        initial_probs[i] = prob

    initial_probs = initial_probs / initial_probs.sum()

    initial_probs = np.where(initial_probs == 0.0, 1e-6, initial_probs)
    initial_probs = np.log(initial_probs)

    return transition_matrix, emission_matrix, initial_probs