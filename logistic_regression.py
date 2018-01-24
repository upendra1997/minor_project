"""Logistic Regression by Upendra Upadhyay."""

import numpy as np
import pandas as pd


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
        for word in line.split(' '):
            dictionary[word] = 1
    X = np.ndarray((len(DF[input]), len(dictionary.keys()) + 1))
    i = 0
    for line in DF[input]:
        for word in line.split(' '):
            X[i][1 + list(dictionary.keys()).index(word)] = 1
        i += 1
    X[..., 0] = np.ones((1, len(DF[input])))
    Y = np.ndarray((len(DF[input]), 1))
    Y = np.array([1 if i > 0 else 0 for i in DF[output]])
    Y = Y.reshape((len(DF[input]), 1))
    return X, Y, dictionary


def train(theta, X, Y, alpha, loop):
    c = cost(theta, X, Y)
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


Xtrain, Ytrain, dictionary = load_matrice('dataset/training_news.csv', 'title', 'sentiment', '\t')
theta_g = np.zeros((len(Xtrain[0]), 1))
alpha = 0.1
iterations = 100
theta = train(theta_g, Xtrain, Ytrain, alpha, iterations)
theta_ne = normal_equation(theta, Xtrain, Ytrain)
temp1, Ytest, temp = load_matrice('dataset/testing_news.csv', 'title', 'sentiment score', '\t')
del temp
del temp1
DF = pd.read_csv('dataset/testing_news.csv', '\t')
i = 0
Xtest = np.ndarray((len(Ytest), len(dictionary) + 1))
for line in DF['title']:
    for word in line.split(' '):
        Xtest[i, ...] = 0
        Xtest[i][0] = 1
        try:
            index = list(dictionary.keys()).index(word)
            Xtest[i][1 + index] = 1
        except ValueError:
            pass
    i += 1
print("Mean Square Error by gradient descent with learning rate ", alpha, " and iterations ", iterations, " is ",
      mse(predict(theta, Xtest), Ytest))
print("error rate by normal equation ", mse(predict(theta_ne, Xtest), Ytest))
