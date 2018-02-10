import numpy as np
from sklearn.tree import DecisionTreeClassifier
from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree


class BR_hoeffding:

    def __init__(self):
        self.hoeffdingTreesList = None

    def fit(self, X, Y):
        self.hoeffdingTreesList = []
        for column in range(len(Y[0])):
            yvar = np.zeros((len(Y), 1))
            for row in range(len(Y)):
                yvar[row][0] = Y[row][column]
            self.hoeffdingTreesList.append(HoeffdingTree())
            self.hoeffdingTreesList[column].fit(X, yvar)
        return self

    def partial_fit(self, X, Y=None):
        if self.hoeffdingTreesList is None:
            self.hoeffdingTreesList = list(range(len(Y[0])))
        for column in range(len(Y[0])):
            yvar = np.zeros((len(Y), 1))
            for row in range(len(Y)):
                yvar[row][0] = Y[row][column]
            if self.hoeffdingTreesList[column]:
                self.hoeffdingTreesList[column].partial_fit(X, yvar)
            else:
                self.hoeffdingTreesList.append(HoeffdingTree())
                self.hoeffdingTreesList[column].partial_fit(X, yvar)
        return self

    def predict(self, X):
        results = []
        c = None
        for i in range(len(self.hoeffdingTreesList)):
            c = self.hoeffdingTreesList[i].predict(X)
            results.append(c)
        return results
