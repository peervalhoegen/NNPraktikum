# -*- coding: utf-8 -*-

"""
Activation functions which can be used within neurons.
"""

from numpy import exp


class Activation:
    """
    Containing various activation functions
    """

    @staticmethod
    def sign(netOutput, threshold=0):
        return netOutput >= threshold

    @staticmethod
    def sigmoid(netOutput):
        # use e^x from numpy to avoid overflow
        return 1/(1+exp(-1.0*netOutput))

    @staticmethod
    def tanh(netOutput):
        pass

    @staticmethod
    def rectified(netOutput):
        pass

    @staticmethod
    def softmax(netOutput):
        pass
