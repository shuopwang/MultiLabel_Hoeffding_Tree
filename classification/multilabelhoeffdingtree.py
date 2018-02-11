from numpy import *
import numpy as np
from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.classification.core.attribute_class_observers.gaussian_numeric_attribute_class_observer import GaussianNumericAttributeClassObserver
from skmultiflow.classification.core.attribute_class_observers.nominal_attribute_class_observer import NominalAttributeClassObserver
from skmultiflow.classification.core.attribute_class_observers.null_attribute_class_observer import NullAttributeClassObserver


class Majority():
    def __init__(self):
        self.vectorCounts = {}
        self.majority_labelset = None
        self.max_value = 0

    def setLearningImpl(self):
        self.majority_labelset = None
        return self

    def partial_fit(self, X, y, weight):

        labels = str(y)
        if(labels in self.vectorCounts):
            weight += self.vectorCounts[labels]
        self.vectorCounts[labels] = weight
        if (weight > self.max_value):
            self.max_value = weight
            self.majority_labelset = labels

    def get_votes_for_instance(self, X):
        return self.vectorCounts


class MultiLabelHoeffdingTree(HoeffdingTree):
    class MultilabelLearningNodeClassifieringNode(HoeffdingTree.ActiveLearningNode):
        def __init__(self, initial_class_observations, classification=None):
            super().__init__(initial_class_observations)
            if classification == None:
                self.classification = Majority()
                self.classification.setLearningImpl()
            else:
                self.classification = classification

        def learn_from_instance(self, X, y, weight, ht):
            self.classification.partial_fit(X, y, weight)
            if not self._is_initialized:
                self._attribute_observers = [None] * len(X)
                self._is_initialized = True
            # print(y)
            # print(y.shape)
            #print("---" * 20 + "debug partial_fit BEGINNING......" + "---" * 20)
            # print(X)
            # print(y)
            # tmp=[]
            # tmp.append(X)

            #print("The shape of X: ", len(X), "  ", len(X[0]))
            #print("The shape of y: ", len(y), "  ", len(y[0]))
            #print("---" * 20 + "debug partial_fit ENDING........." + "---" * 20)
            self.L = len(y)
            for l in y:
                if l != 0 and l != '0':
                    l = int(l)
                    if l not in self._observed_class_distribution:
                        self._observed_class_distribution[l] = weight
                    else:
                        self._observed_class_distribution[l] += weight
            for i in range(len(X)):
                obs = self._attribute_observers[i]
                if obs is None:
                    if i in ht.nominal_attributes:
                        obs = NominalAttributeClassObserver()
                    else:
                        obs = GaussianNumericAttributeClassObserver()
                    self._attribute_observers[i] = obs
                for l in y:
                    if l != 0 and l != '0':
                        obs.observe_attribute_class(X[i], int(l), weight)
            return self

        def get_class_votes(self, X, ht):
            return self.classification.get_votes_for_instance(X)

#    def partial_fit(self, X, y=None, classes=None):
#        super.partial_fit(X, y, classes)
#        return self

    def _new_learning_node(self, initial_class_observations=None):
        if initial_class_observations is None:
            initial_class_observations = {}
        return self.MultilabelLearningNodeClassifieringNode(initial_class_observations)

    def predict(self, X):
        # TODO
        #r, _ = get_dimensions(X)
        real = []
        t = []
        votes = self.get_votes_for_instance(X)
        if votes == {}:
            predictions = np.zeros((len(X), self.L))
            return predictions
        else:
            predictions = (max(votes, key=votes.get))
        for i in predictions:
            if i.isdigit():
                real.append(int(i))
        t.append(real)
#        print("-" * 20 + "debug 2" + "-" * 20)
#        print("type of results:", type(t))
#        print("len of results", len(t))
#        print("results: ", t)
#        print("-" * 20 + "debug 2" + "-" * 20)
        return t
