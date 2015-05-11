
import time

import numpy as np

from util.activation_functions import Activation
from model.layer import Layer
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier


class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, layers=None, outputTask='classification',
                 inputWeights=None, learningRate=0.01, epochs=50):

        # Build up the network from specific layers
        # Here is an example
        # Should read those configuration from the command line or config file
        if layers is None:
            inputLayer = Layer(784, 100, weights=inputWeights)
            hiddenLayer = Layer(100, 50)
            outputLayer = LogisticLayer(50, 10)
            
            self.layers = []
            self.layers.append(inputLayer)
            self.layers.append(hiddenLayer)
            self.layers.append(outputLayer)
        else:
            self.layers = layers

        self.outputTask = outputTask  # Either classification or regression
        self.learningRate = learningRate
        self.epochs = epochs

    def getLayer(self, layerIndex):
        return self.layers[layerIndex]

    def feedForward(self, input):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        input : ndarray
            a numpy array (1,nIn + 1) containing the input of the layer

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the output layer
        """
        # Here you have to propagate forward through the layers
        lastOutput = input
        for layer in self.layers:
            lastOutput = layer.forward(lastOutput)
        return lastOutput

    def computeError(self, input, target):
        """
        Compute the total error of the network

        Parameters
        ----------
        input : ndarray
            a numpy array (1,nIn + 1) containing the input of the layer

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        return target - self.feedForward(input)

    def updateWeights(self, input, target):
        """
        Update the weights of the layers by propagating back the error

        Parameters
        ----------
        input : ndarray
            a numpy array (1,nIn + 1) containing the input of the layer

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        error = self.computeError(input, target)
        lastWeights = None
        for layer in reversed(self.layers):
            layer.updateWeights(input, lastWeights)
            lastWeights = layer.weights

    def train(self, trainingSet, validationSet):
        # train procedures of the classifier
        pass

    def classify(self, testInstance):
        # classify an instance given the model of the classifier
        pass

    def evaluate(self, test):
        # evaluate a whole test set given the model of the classifier
        pass
