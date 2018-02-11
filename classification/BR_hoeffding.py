import numpy as np
from sklearn.tree import DecisionTreeClassifier
from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree


class BR_Hoeffding:

    def __init__(self):
        self.hoeffdingTreesList = []

    def fit(self, X, Y):
        self.hoeffdingTreesList = []
        for column in range(len(Y[0])):
            yvar = np.zeros((len(Y), 1))
            for row in range(len(Y)):
                yvar[row][0] = Y[row][column]
            self.hoeffdingTreesList.append(HoeffdingTree())
            self.hoeffdingTreesList[column].fit(X, yvar)
        return self

    def partial_fit(self, X, y, classes=None):
        if len(self.hoeffdingTreesList) == 0:
            self.hoeffdingTreesList = []
            for i in range(len(y[0])):
                self.hoeffdingTreesList.append(HoeffdingTree())
        self.L = len(y[0])
        for column in range(len(y[0])):
            yvar = y[:, column]
            self.hoeffdingTreesList[column].partial_fit(X, yvar)
        return self

    def predict(self, X):
        results = []

        print(self.L)
        if len(self.hoeffdingTreesList) == 0:
            results = np.zeros((len(X), self.L))
            return results
        c = None
        tmp = []
        for index in range(len(X)):
            for i in range(len(self.hoeffdingTreesList)):
                c = self.hoeffdingTreesList[i].predict(X[index])
                tmp.append(c)
            results.append(tmp)
            tmp = []
        return results
