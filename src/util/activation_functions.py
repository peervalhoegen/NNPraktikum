# -*- coding: utf-8 -*-

"""
Activation functions which can be used within neurons.
"""

from numpy import exp
import numpy as np
from numpy import divide


class Activation:
    """
    Containing various activation functions and their derivatives
    """

    @staticmethod
    def sign(netOutput, threshold=0):
        return netOutput >= threshold

    @staticmethod
    def sigmoid(netOutput):
        # use e^x from numpy to avoid overflow
        return 1/(1+exp(-1.0*netOutput))

    @staticmethod
    def sigmoidPrime(netOutput):
        # Here you have to code the derivative of sigmoid function
        return netOutput * (1- netOutput)

    @staticmethod
    def tanh(netOutput):
        # return 2*Activation.sigmoid(2*netOutput)-1
        ex = exp(1.0*netOutput)
        exn = exp(-1.0*netOutput)
        return divide(ex-exn, ex+exn)

    @staticmethod
    def tanhPrime(netOutput):
        # Here you have to code the derivative of tanh function
        pass

    @staticmethod
    def rectified(netOutput):
        return lambda x: max(0.0, x)

    @staticmethod
    def rectifiedPrime(netOutput):
        # Here you have to code the derivative of rectified linear function
        pass

    @staticmethod
    def identity(netOutput):
        #return lambda x: x
        return np.array([max(netOutput[0],1,-1)])

    @staticmethod
    def identityPrime(netOutput):
        return 1
    
    @staticmethod
    def softmax(netOutput):
        # Here you have to code the softmax function
        return exp(netOutput)/sum(exp(netOutput)) 
    @staticmethod
    def getActivation(str):
        """
        Returns the activation function corresponding to the given string
        """

        if str == 'sigmoid':
            return Activation.sigmoid
        elif str == 'softmax':
            return Activation.softmax
        elif str == 'tanh':
            return Activation.tanh
        elif str == 'relu':
            return Activation.rectified
        elif str == 'linear':
            return Activation.identity
        else:
            raise ValueError('Unknown activation function: ' + str)

    @staticmethod
    def getDerivative(str):
        """
        Returns the derivative function corresponding to a given string which
        specify the activation function
        """

        if str == 'sigmoid':
            return Activation.sigmoidPrime
        elif str == 'tanh':
            return Activation.tanhPrime
        elif str == 'relu':
            return Activation.rectifiedPrime
        elif str == 'linear':
            return Activation.identityPrime
	elif str == 'softmax':
	    return Activation.sigmoidPrime		
        else:
            raise ValueError('Cannot get the derivative of'
                             ' the activation function: ' + str)
