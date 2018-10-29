import pickle
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from linear_regression import predict
from pprint import pprint
import string
import nltk
# nltk.download()
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()
set_stop_words = set(stopwords.words('english'))
table = str.maketrans('','',string.punctuation)

def clean(line):
    for word in word_tokenize(line):
        word = word.lower().translate(table)
        if not word.isalpha():
            continue
        if word in set_stop_words:
            continue
        word = porter.stem(word)
        yield word

def sigmoid(t):
    return 1 / (1 + np.exp(-t))


class analyze():
    def __init__(self):
        f = open("weight",'rb')
        self.theta = pickle.loads(f.read())
        f.close()
        f = open("dict",'rb')
        self.dictionary = pickle.loads(f.read())
        f.close()

    def input(self, line):
        X = np.zeros((1, len(self.dictionary.keys()) + 1))
        index = 0
        for word in clean(line):
            try:
                index = list(self.dictionary.keys()).index(word)
                X[0][1 + index] = 1
            except ValueError:
                pass
        result = X.dot(self.theta)[0][0]
        return sigmoid(result)