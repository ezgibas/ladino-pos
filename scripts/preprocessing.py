# tokenize Ladino & Spanish data
import spacy
from spacy.lang.es.examples import sentences 

debug = False

# tokenizes one sentence using spaCy's Spanish tokenizer
# and returns it as an array, optinally returns an array of POS tags
def tokenize(sentence, give_pos=False):
    # load tokenizer
    nlp = spacy.load("es_core_news_sm")
    doc = nlp(sentence)
    tokens = []
    pos = []
    if debug:
        print(sentence)
    for token in doc:
        tokens.append(str(token))
        if give_pos:
            pos.append(str(token.pos_))
    return tokens, pos

# prepare 2 datasets of different languages    
# to be passed into fastalign
def fastalign_prep(dataset1, dataset2):
    parallel_data = [f"{ds1} ||| {ds2}" for ds1, ds2 in zip(dataset1, dataset2)]
    return parallel_data


def main():
    # file paths
    global ladino_file_path, spanish_file_path, ladino_tokenized, spanish_tokenized, spanish_pos, parallel_file_path

    ladino_file_path = '../data/tatoeba.spa-lad.lad'
    spanish_file_path = '../data/tatoeba.spa-lad.spa'
    parallel_file_path = '../data/parallel_spa-lad.txt'

    # load data
    with open(ladino_file_path, "r", encoding="utf-8") as file:
        ladino_data = file.read().splitlines()

    with open(spanish_file_path, "r", encoding="utf-8") as file:
        spanish_data = file.read().splitlines()

    # create arrays of tokens
    ladino_tokenized = [tokenize(sentence)[0] for sentence in ladino_data]
    spanish_tokenized, spanish_pos = zip(*[tokenize(sentence, give_pos=True) for sentence in spanish_data])
    spanish_tokenized = list(spanish_tokenized)
    spanish_pos = list(spanish_pos)

    # create sentences where tokens are separated by empty space
    ladino_tokenized_joined = [" ".join(sentence) for sentence in ladino_tokenized]
    spanish_tokenized_joined = [" ".join(sentence) for sentence in spanish_tokenized]

    # put results into fastalign format
    parallel_data = fastalign_prep(ladino_tokenized_joined, spanish_tokenized_joined)

    # save result as a file to be passed into fastalign
    with open(parallel_file_path, "w", encoding="utf-8") as f:
        for line in parallel_data:
            f.write(line + "\n")
    f.close()

main()