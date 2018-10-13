"""Logisitic Regression by Upendra Upadhyay"""

import numpy as np
import pandas as pd
import pickle

def predict(theta, X):
    return X.dot(theta)


def mse(X, Y):
    return ((X - Y) ** 2).mean()


def gradient_descent(theta, X, Y, alpha):
    m = len(Y)
    hypothesis = predict(theta, X)
    loss = hypothesis - Y
    transX = X.transpose()
    gradient = transX.dot(loss) / m
    theta = theta - alpha * gradient
    c = mse(predict(theta, X), Y)
    return theta, c


def train(theta, X, Y, alpha, loop):
    c = mse(predict(theta, X), Y)
    print("Initial Cost: ", c)
    for i in range(loop):
        theta, c = gradient_descent(theta, X, Y, alpha)
        print("iteration: {} MSE: {}".format(i, c))
    return theta


def normal_equation(X, Y):
    return np.linalg.pinv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)


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
    Y = np.array(DF[output]).reshape((len(DF[input]), 1))
    return X, Y, dictionary



if __name__ == '__main__':
    Xtrain, Ytrain, dictionary = load_matrice('dataset/training_tweet.csv', 'Tweets', 'sentiment', '\t')
    theta_g = np.zeros((len(Xtrain[0]), 1))
    alpha = 0.1
    iterations = 100
    theta = train(theta_g, Xtrain, Ytrain, alpha, iterations)
    temp1, Ytest, temp = load_matrice('dataset/testing_tweet.csv', 'tweets', 'sentiment score', ',')
    del temp1
    del temp
    DF = pd.read_csv('dataset/testing_tweet.csv', ',')
    i = 0
    Xtest = np.ndarray((len(Ytest), len(dictionary) + 1))
    for line in DF['tweets']:
        for word in line.split(' '):
            Xtest[i, ...] = 0
            Xtest[i][0] = 1
            try:
                index = list(dictionary.keys()).index(word)
                Xtest[i][1 + index] = 1
            except ValueError:
                Xtest[i][1 + index] = 0
        i += 1


    predictedY = predict(theta, Xtest)
    # predictedY_ne = predict(normal_equation(Xtrain, Ytrain), Xtest)
    f = open("weight",'wb')
    f.write(pickle.dumps(theta))
    f.close()
    f = open("dict",'wb')
    f.write(pickle.dumps(dictionary))
    f.close()
    print("Mean Squared error by gradient descent with learning rate ", alpha, " and iterations ", iterations, " is ",
          mse(predictedY, Ytest))
    # print("error rate by normal equation ", mse(predictedY_ne, Ytest))
