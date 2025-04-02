from collections import Counter, defaultdict

class Token:
    def __init__(self, word, pos):
        self.word = word
        self.pos = pos
    
    def get_word(self):
        return self.word
    
    def get_pos(self):
        return self.pos
    
    def correct_pos(self, pos):
        self.pos = pos

    # format to fit NLTK input type
    def format(self):
        return (self.word, self.pos)
    
    def __str__(self):
        return self.word + " (" + self.pos + ")"
    
def load_tagged_sentences(file_path, split=0.8):
    sentences = []  
    current_sentence = [] 

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            
            if not line:  # empty line - end of sentence
                if current_sentence:  # add if sentence not empty
                    sentences.append(current_sentence)
                    current_sentence = []  
                continue

            if "-" in line and line.split("-")[0].startswith("b"):  
                # skip sentence IDs
                continue

            parts = line.split("\t")  # split word, tag
            if len(parts) == 2:
                word, tag = parts
                current_sentence.append(Token(word.lower(), tag))

    # add the last sentence if the file doesn't end with a blank line
    if current_sentence:
        sentences.append(current_sentence)

    split_idx = int(len(sentences)*split)
    train = sentences[:split_idx]
    test = sentences[split_idx:]

    return train, test

def load_tags(file_path):
    tags = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            tags.append(line)
    return tags
    
def word_to_tag_counts(data):
    word_tags = defaultdict(Counter)  # {word: {tag1: count, tag2: count, ...}}
    for sentence in data:
        for token in sentence:
            word = token.get_word()
            tag = token.get_pos()
            word = word.lower()
            word_tags[word][tag] += 1  # count occurrences of each tag
    return word_tags