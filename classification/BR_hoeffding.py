import numpy as np
from sklearn.tree import DecisionTreeClassifier
from hoeffding_tree import HoeffdingTree


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
        # print("---" * 20 + "debug partial_fit BEGINNING......" + "---" * 20)
        # print("The shape of X: ", len(X), "  ", len(X[0]))
        # print("The shape of y: ", len(y), "  ", len(y[0]))
        # print("---" * 20 + "debug partial_fit ENDING........." + "---" * 20)
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
        # print("The shape of X: ", len(X), "  ", len(X[0]))
        if len(self.hoeffdingTreesList) == 0:
            results = np.zeros((len(X), self.L))
            print("-" * 10 + "debug 1" + "-" * 10)
            print("results:", results)
            print("-" * 10 + "debug 1 " + "-" * 10)
            return results
        c = None
        tmp = []
        for index in range(len(X)):
            #print("-" * 15 + "debug predict BEGINNING" + "-" * 15)
            #print("The shape of X: ", len(X), "  ", len(X[index]))
            #print("-" * 15 + "debug predict ENDING" + "-" * 15)
            for i in range(len(self.hoeffdingTreesList)):
                c = self.hoeffdingTreesList[i].predict(X[index])
                tmp.append(c[0])
            results.append(tmp)
            tmp = []
#        print("-" * 20 + "debug 2" + "-" * 20)
#        print("type of results:", type(results))
#        print("results: ", results)
#        print("-" * 20 + "debug 2" + "-" * 20)

        return results
