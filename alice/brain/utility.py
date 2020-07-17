import nltk
nltk.download('punkt')

import numpy as np

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence: str):
    return nltk.word_tokenize(sentence)


def stem(word: str):
    return stemmer.stem(word.lower())


def bag_of_world(tokenized_sentence: list, words: list):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)

    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag
