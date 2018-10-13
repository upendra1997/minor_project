import pickle
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from linear_regression import predict
from pprint import pprint

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
        for word in line.split(' '):
            try:
                index = list(self.dictionary.keys()).index(word)
                X[0][1 + index] = 1
            except ValueError:
                pass
        result = X.dot(self.theta)[0][0]
        return result*100