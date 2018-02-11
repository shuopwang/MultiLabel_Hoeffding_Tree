import numpy as np
from skmultiflow.classification.lazy.knn_adwin import KNNAdwin, KNN
## to add in run_experiments: ##
#from BR_KNN import BR_knn 
class BR_knn:

    def __init__(self):
        self.knnList = []

    def fit(self, X, Y):
        self.knnList = []
        for column in range(len(Y[0])):
            yvar = np.zeros((len(Y), 1))
            for row in range(len(Y)):
                yvar[row][0] = Y[row][column]
            self.knnList.append(KNN(k=10, max_window_size=100, leaf_size=30))
            self.knnList[column].fit(X, yvar)
        return self

    def partial_fit(self, X, y, classes=None):
        if len(self.knnList) == 0:
            self.knnList = []
            for i in range(len(y[0])):
                self.knnList.append(KNN(k=10, max_window_size=100, leaf_size=30))
        self.L = len(y[0])
        for column in range(len(y[0])):
            yvar = y[:, column]
            self.knnList[column].partial_fit(X, yvar)
        return self

    def predict(self, X):
        results = []
        M,N = X.shape
        if len(self.knnList) == 0:
            results = np.zeros((M, self.L))
            return results
        c = None
        tmp = []
        for index in range(M):
            for i in range(len(self.knnList)):
            	c = self.knnList[i].predict([X[index]])
            	tmp.append(c[0])
            results.append(tmp)
            tmp = []

        return results
