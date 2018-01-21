"""Logisitic Regression by Upendra Upadhyay"""
import math

import numpy as np
import pandas as pd


def sigmoid(t):
    return 1 / 1 + math.exp(-1 * t)


def predict(theta, X):
    return X.dot(theta)


def cost(theta, X, Y):
    return ((X.dot(theta) - Y) ** 2).mean() / 2


def gradient_descent(theta, X, Y, alpha):
    # temp = np.ndarray((len(theta),1))
    m = len(Y)
    hypothesis = predict(theta, X)
    loss = hypothesis - Y
    transX = X.transpose()
    gradient = transX.dot(loss) / m
    # for i in range(0,len(theta)):
    theta = theta - alpha * gradient
    # theta = temp
    c = cost(theta, X, Y)
    # print("Cost: ",c)
    return theta, c


def train(theta, X, Y, alpha, loop):
    c = cost(theta, X, Y)
    print("Initial Cost: ", c)
    for i in range(loop):
        theta, c = gradient_descent(theta, X, Y, alpha)
        print("iteration: {} cost: {}".format(i, c))
    return theta


def normal_equation(theta, X, Y):
    return np.linalg.pinv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)


def load_matrice(file, input, output, sep):
    dictionary = {}
    DF = pd.read_csv(file, sep)
    # print(DF.head(10))
    # print(len(DF[input]))
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
    Y = np.array(DF[output]).reshape((len(DF[input]), 1))
    return X, Y, dictionary


def test(theta, X, Y):
    # print("predict\toutput");
    # print(np.hstack((predict(theta,X),Y)))
    return cost(theta, X, Y) * 200


X, Y, dictionary = load_matrice('dataset/training_tweet.csv', 'Tweets', 'sentiment', '\t')
theta_g = np.zeros((len(X[0]), 1))
alpha = 0.001
iterations = 10000
theta = train(theta_g, X, Y, alpha, iterations)
# print(len(theta))
theta_ne = normal_equation(theta, X, Y)
temp1, Y, temp = load_matrice('dataset/testing_tweet.csv', 'tweets', 'sentiment score', ',')

DF = pd.read_csv('dataset/testing_tweet.csv', ',')
i = 0
X = np.ndarray((len(Y), len(dictionary) + 1))
for line in DF['tweets']:
    for word in line.split(' '):
        X[i, ...] = 0
        X[i][0] = 1
        try:
            index = list(dictionary.keys()).index(word)
            X[i][1 + index] = 1
        except ValueError:
            pass
    i += 1

print("error rate by gradient descent with learning rate ", alpha, " and iterations ", iterations, " is ",
      test(theta_g, X, Y), '%')
print("error rate by normal equation ", test(theta_ne, X, Y), '%')
