# tokenize Ladino & Spanish data
import spacy
from spacy.lang.es.examples import sentences 

# tokenizes one sentence using spaCy's Spanish tokenizer
# and returns it as an array
def tokenize(sentence, give_pos=False):
    # load tokenizer
    nlp = spacy.load("es_core_news_sm")
    doc = nlp(sentence)
    tokens = []
    pos = []
    for token in doc:
        tokens.append(token)
        if give_pos:
            pos.append(token.pos_)
    return tokens, pos