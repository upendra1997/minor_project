#!/usr/bin/py -3
"""Logistic Regression by Upendra Upadhyay."""

import numpy as np
import pandas as pd
import pickle
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


def predict(theta, X):
    return sigmoid(X.dot(theta))


def cost(pred, Y):
    c = -Y * np.log(pred) - (1 - Y) * np.log(1 - pred)
    return c.mean()


def mse(X, Y):
    return ((X - Y) ** 2).mean()


def gradient_descent(theta, X, Y, alpha):
    m = len(Y)
    hypothesis = predict(theta, X)
    loss = hypothesis - Y
    transX = X.transpose()
    gradient = transX.dot(loss) / m
    theta = theta - alpha * gradient
    c = cost(predict(theta, X), Y)
    return theta, c


def load_matrice(file, input, output, sep):
    dictionary = {}
    DF = pd.read_csv(file, sep)
    for line in DF[input]:
        for word in clean(line):
            dictionary[word] = 1
    X = np.ndarray((len(DF[input]), len(dictionary.keys()) + 1))
    i = 0
    for line in DF[input]:
        for word in clean(line):
            X[i][1 + list(dictionary.keys()).index(word)] = 1
        i += 1
    X[..., 0] = np.ones((1, len(DF[input])))
    Y = np.ndarray((len(DF[input]), 1))
    Y = np.array([1 if i > 0 else 0 for i in DF[output]])
    Y = Y.reshape((len(DF[input]), 1))
    return X, Y, dictionary


def train(theta, X, Y, alpha, loop):
    c = cost(predict(theta, X), Y)
    print("Initial Cost: ", c)
    for i in range(loop):
        theta, c = gradient_descent(theta, X, Y, alpha)
        print("iteration: {} cost: {}".format(i, c))
    return theta


def normal_equation(theta, X, Y):
    return np.linalg.pinv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)


def test(pred, Y):
    pre = 1.0 * (pred > 0.5)
    return mse(pre, Y)


if __name__ == '__main__':
    Xtrain, Ytrain, dictionary = load_matrice('dataset/training_news.csv', 'title', 'sentiment', '\t')
    theta_g = np.random.uniform(size=(len(Xtrain[0]), 1))
    alpha = 0.01
    iterations = 1000
    theta = train(theta_g, Xtrain, Ytrain, alpha, iterations)
    # theta_ne = normal_equation(theta, Xtrain, Ytrain)
    temp1, Ytest, temp = load_matrice('dataset/testing_news.csv', 'title', 'sentiment score', '\t')
    del temp
    del temp1
    DF = pd.read_csv('dataset/testing_news.csv', '\t')
    i = 0
    Xtest = np.ndarray((len(Ytest), len(dictionary) + 1))
    for line in DF['title']:
        for word in clean(line):
            Xtest[i, ...] = 0
            Xtest[i][0] = 1
            try:
                index = list(dictionary.keys()).index(word)
                Xtest[i][1 + index] = 1
            except ValueError:
                pass
        i += 1

    predictedY = predict(theta, Xtest)
    # predictedY_ne = predict(normal_equation(Xtrain, Ytrain), Xtest)
    f = open("weight",'wb')
    f.write(pickle.dumps(theta))
    f.close()
    f = open("dict",'wb')
    f.write(pickle.dumps(dictionary))
    f.close()

    print("Mean Square Error by gradient descent with learning rate ", alpha, " and iterations ", iterations, " is ",
          mse(predictedY, Ytest))
    # print("error rate by normal equation ", mse(predict(theta_ne, Xtest), Ytest))
