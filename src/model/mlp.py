
import time
import math
import numpy as np

from util.activation_functions import Activation
from model.layer import Layer
# from model.logistic_layer import LogisticLayer
from model.classifier import Classifier
import random


class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, outputTask='classification',
                 inputWeights=None, learningRate=0.01, epochs=50):

        # Build up the network from specific layers
        # Here is an example
        # Should read those configuration from the command line or config file
        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        if layers is None:
            inputLayer = Layer(2, 2, weights=inputWeights)
            hiddenLayer = Layer(2, 2)
            outputLayer = Layer(2, 1)
            
            self.layers = []
#             self.layers.append(inputLayer)
#             self.layers.append(hiddenLayer)
            self.layers.append(outputLayer)
        else:
            self.layers = layers

        self.outputTask = outputTask  # Either classification or regression
        self.learningRate = learningRate
        self.epochs = epochs
	self.totalError = np.inf
	
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
            # print("layer " + str(i))
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
        if self.outputTask == 'classification':
                myTarget = [0] * target + [1] + [0] * (self.layers[-1].nOut - target - 1)
        else:
                myTarget = target
        networkOutput = self.feedForward(input)
        outputNodes = self.layers[-1].shape[0]
        #if outputNodes > 1:
        #    error = 0
        #    for index in range(outputNodes):
        #        error += np.power((myTarget[index] - networkOutput[index]), 2)
        #    return error / 2
        #else:
        #    return np.power((myTarget - networkOutput), 2) / 2
        return myTarget - networkOutput

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

        self.layers[-1].error = self.computeError(input, target)
        # print "error in outputLayer " + str(self.layers[-1].error)
        ds = self.layers[-1].error 
        for layer in reversed(self.layers):	    
            layer.updateWeights(ds, self.learningRate)  # todo implement for outLayer where ds is ignored!
            ds = np.dot(layer.delta, layer.weights)
            # 	    print("ds:" + str(ds.shape) + "=dot( layer.delta: " + str(layer.delta.shape) + " , " + " layer.weights: " + str(layer.weights.shape) +")" )
            ds = ds[:-1]  # remove ds for 'imaginary' bias-input.



    def train(self):
#        print('trainingSet.input: ' + str(self.trainingSet.input))
#       print('trainingSet.label: ' + str(self.trainingSet.label))
        # train procedures of the
	for i in range(self.epochs):
	        print("start of epoch "+str(i))
                foo = zip(self.trainingSet.input, self.trainingSet.label)
		#random.shuffle(foo)
		for input, target in foo: 
			#print("input:" + str(input))
			self.updateWeights(input, target)
			#print "weights after update:" 
                        #for i,l in enumerate(self.layers):
			#	print " layer"+str(i)+": " +str(l.weights)

                totalError = 0
                #for input, target in foo:
                totalError += abs(self.computeError(input, target))
                
                print("Total error: " + str(totalError))                        	     
		print("learning rate:" + str(self.learningRate))
		#eval fuer validation?
                if (totalError - self.totalError).any >0: 
			self.learningRate *= 0.8
		#else:
		#	self.learningRate *= 1-(i-1.0)/(self.epochs*50)

		self.totalError =   totalError

    def classify(self, testInstance):
        # classify an instance given the model of the classifier
        result = self.feedForward(testInstance)
        if self.outputTask == "classification":
            index = result.argmax()
            return index
        else:
            return result
        
        print "classify()-> " + str()
        return (self.feedForward(testInstance))[0]
        

   
    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        print('test' + str(self.testSet.label))
        return  list(map(self.classify, test))

