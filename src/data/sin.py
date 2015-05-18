# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import shuffle
from data.Xor_Data_set import Xor_DataSet


class Sin(object):
    """
    Small subset (5000 instances) of MNIST data to recognize the digit 7

    Parameters
    ----------
    dataPath : string
        Path to a CSV file with delimiter ',' and unint8 values.
    numTrain : int
        Number of training examples.
    numValid : int
        Number of validation examples.
    numTest : int
        Number of test examples.

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    """

    # dataPath = "data/xor.csv"

    def __init__(self, dataPath, numTrain=4, numValid=4, numTest=4):

        self.trainingSet = []
        self.validationSet = []
        self.testSet = []

        self.load(dataPath, numTrain, numValid, numTest)

    def load(self, dataPath, numTrain, numValid, numTest):
        """Load the data."""
        print("Loading data from " + dataPath + "...")

#        data = np.genfromtxt(dataPath, delimiter=",", dtype="uint8")
        x = np.linspace(0,100,10000);
        y = np.sin(x)/2 + 0.5
        data = np.zeros((10000,2))
        data[:,0] = y
        data[:,1] = x        
        # The last numTest instances ALWAYS comprise the test set.
        train, test = data[:numTrain+numValid], data[:numTest]
        #shuffle(train)
        
        train, valid = train[:numTrain], train[numTrain:]
#	print("train" + str(train))
#	print("valid" + str(valid))
#	print("test" + str(test))

        self.trainingSet = Xor_DataSet(train)
        self.validationSet = Xor_DataSet(valid)
        self.testSet = Xor_DataSet(test)

        print("Data loaded.")
