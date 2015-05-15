#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from data.mnist_seven import MNISTSeven
from data.xor import Xor
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron
from report.evaluator import Evaluator
from data.data_set import DataSet
from model.layer import Layer
from model.logistic_layer import LogisticLayer


def main():
#    inputLayer = Layer(2, 2)
#    hiddenLayer1 = Layer(2, 2)
#    hiddenLayer2 = Layer(50, 50)    
    outputLayer = Layer(2, 1)
    
    layers = []
#    layers.append(inputLayer)
#    layers.append(hiddenLayer1)
#    layers.append(hiddenLayer2)
    layers.append(outputLayer)
 #   m=MultilayerPerceptron(layers)
 #   input=np.array([0,0])
    #,[0,1],[1,0],[1,1]]
 #   target=0
    #[0,1,1,0]
 #   print(m.computeError(input, target))
 #   m.updateWeights(input,target)
    

#    data = Xor("../data/xor.csv", 4, 4, 4)
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    #myStupidClassifier = StupidRecognizer(data.trainingSet,
    #                                      data.validationSet,
    #                                      data.testSet)
    #myPerceptronClassifier = Perceptron(data.trainingSet,
    #                                    data.validationSet,
    #                                    data.testSet,
    #                                    learningRate=0.005,
    #                                    epochs=30)
    #myLRClassifier = LogisticRegression(data.trainingSet,
    #                                    data.validationSet,
    #                                    data.testSet,
    #                                    learningRate=0.005,
    #                                    epochs=30)

    myLRClassifier = MultilayerPerceptron(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
					#layers,
#					outputTask='Regression',
			                inputWeights=None,
                                        learningRate=1,
                                        epochs=19)


   

    print("\nMultilayerPerceptron has been training..")
    myLRClassifier.train()
    print("Done..")

    

    pred = myLRClassifier.evaluate()
    print('pred' + str(pred))
    eval = Evaluator()
    #eval.printConfusionMatrix(data.testSet, pred)
    #eval.printClassificationResult(data.testSet, pred, target_names)

if __name__ == '__main__':
    main()
