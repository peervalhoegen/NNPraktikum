# -*- coding: utf-8 -*-

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier
from model.logistic_regression import LogisticRegression

__author__ = "ABC XYZ"  # Adjust this when you copy the file
__email__ = "ABC.XYZ@student.kit.edu"  # Adjust this when you copy the file


class Layer():
    """
    one generic layer
    """

    def __init__(self, noPerceprons, predecessor, learningRate=0.01, epochs=50):
        #, train, valid, test für input-layer
        self.learningRate = learningRate
        self.epochs = epochs
        self.perceptronList = [LogisticRegression(None, None, None, learningRate, epochs) for i in range(noPerceprons)]

        self.predecessor = predecessor        #e.g. input-layer is predecessor of first hidden-layer
        #TODOself.predecessor.setSuccessor(self)   #set oneself as successor

    def setSuccessor(self, other):
        self.successor = other



    def predict(self,input):

        self.lastOuputList = [p.fire(input_from_somewhere) for p in self.perceptronList]
        self.successor.predict(lastOuputList)

    def propagateBack(self,error):

        lastOuputList = [p.fire(input_from_somewhere) for p in self.perceptronList]
        self.predecessor.propagateBack(error2)

        """Train the Logistic Regression"""
        # TODO: Here you have to implement the Logistic Regression Training
        # Algorithm
        # TODO: use self.trainingSet
        # TODO: use self.validationSet

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        # TODO: Here you have to implement the Logistic Regression Algorithm
        # to classify a single instance
        pass

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def fire(self, input):
        return Activation.sigmoid(np.dot(np.array(input), self.weight))
