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


def preprocess(ladino_file_path, spanish_file_path, parallel_file_path, spanish_pos_file_path):
    # file paths

    # load data
    with open(ladino_file_path, "r", encoding="utf-8") as file:
        ladino_data = file.read().splitlines()

    with open(spanish_file_path, "r", encoding="utf-8") as file:
        spanish_data = file.read().splitlines()

    # create arrays of tokens
    if debug:
        ladino_tokenized = [tokenize(sentence)[0] for sentence in ladino_data[:5]]
        spanish_tokenized, spanish_pos = zip(*[tokenize(sentence, give_pos=True) for sentence in spanish_data[:5]])
    else:
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
            if debug:
                print(line)
            f.write(line + "\n")

    # also save spanish POS tags
    with open(spanish_pos_file_path, "w", encoding="utf-8") as f:
        for tags in spanish_pos:
            if debug:
                print(tags)
            f.write(str(tags) + "\n")

    f.close()