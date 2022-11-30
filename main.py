import csv
import numpy as np
from baselinefn import *
from sklearn.model_selection import ShuffleSplit, train_test_split


def load_data(path):
    with open(path, newline="") as file:
        arr = list(csv.reader(file))
    data = np.array(arr)

    X = data[:, :-1]
    Y = data[:, -1]

    return X, Y


# def train_test_split(X, Y):
#     shuffle_split = ShuffleSplit(n_splits=1, test_size=0.2)
#
#     for train_index, test_index in shuffle_split.split(X):
#         train = train_index
#         test = test_index
#
#     return train, test


class perceptron():

    def __init__(self):
        pass

    def trainPerceptron(self, train_set, train_labels, learning_rate, max_iter):
        # TODO: Write your code here
        # return the trained weight and bias parameters

        W, b = np.zeros(train_set.shape[1]), 0

        for _ in range(max_iter):

            for i in range(train_set.shape[0]):

                # label is 0 but the prediction is negative
                if (train_labels[i] == 1) and (np.dot(W, train_set[i]) + b < 0):
                    W, b = W + train_set[i] * learning_rate, b + learning_rate

                # label is 1 but the prediction is positive
                if (train_labels[i] == 0) and (np.dot(W, train_set[i]) + b >= 0):
                    W, b = W - train_set[i] * learning_rate, b - learning_rate

        return W, b

    def classifyPerceptron(self, train_set, train_labels, dev_set, learning_rate, max_iter):
        # TODO: Write your code here
        # Train perceptron model and return predicted labels of development set

        W, b = self.trainPerceptron(train_set, train_labels, learning_rate, max_iter)

        predictions = []

        for i in range(dev_set.shape[0]):

            if (np.dot(W, dev_set[i]) + b) > 0:

                predictions.append(1)

            else:

                predictions.append(0)

        return predictions

# Data processing
path = "spambase/spambase.data"
data, labels = load_data(path)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)

# MLP

# Baseline Functions
baselineNB(x_train, y_train, x_test, y_test, 2)
baselineDT(x_train, y_train, x_test, y_test)


