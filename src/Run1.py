#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron
from report.evaluator import Evaluator
from data.data_set import DataSet
from model.layer import Layer
from model.logistic_layer import LogisticLayer


def main():
    inputLayer = Layer(2, 100)
    hiddenLayer = Layer(100, 50)
    outputLayer = Layer(50, 1)
    
    layers = []
    layers.append(inputLayer)
    layers.append(hiddenLayer)
    layers.append(outputLayer)
    m=MultilayerPerceptron(layers)
    input=np.array([0,0])
    #,[0,1],[1,0],[1,1]]
    target=0
    #[0,1,1,0]
    print(m.computeError(input, target))
    m.updateWeights(input,target)
    
    exit()
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)

    myStupidClassifier = StupidRecognizer(data.trainingSet,
                                          data.validationSet,
                                          data.testSet)
    myPerceptronClassifier = Perceptron(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.005,
                                        epochs=30)
    myLRClassifier = LogisticRegression(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.005,
                                        epochs=30)

    # Train the classifiers
    print("=========================")
    print("Training..")

    print("\nStupid Classifier has been training..")
    myStupidClassifier.train()
    print("Done..")

    print("\nPerceptron has been training..")
    myPerceptronClassifier.train()
    print("Done..")

    print("\nLogistic Regression has been training..")
    myLRClassifier.train()
    print("Done..")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    stupidPred = myStupidClassifier.evaluate()
    perceptronPred = myPerceptronClassifier.evaluate()
    lrPred = myLRClassifier.evaluate()

    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("Result of the stupid recognizer:")
    # evaluator.printComparison(data.testSet, stupidPred)
    evaluator.printAccuracy(data.testSet, stupidPred)

    print("\nResult of the Perceptron recognizer:")
    # evaluator.printComparison(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.testSet, perceptronPred)
    
    print("\nResult of the Logistic Regression recognizer:")
    # evaluator.printComparison(data.testSet, perceptronPred)    
    evaluator.printAccuracy(data.testSet, lrPred)

    # eval.printConfusionMatrix(data.testSet, pred)
    # eval.printClassificationResult(data.testSet, pred, target_names)

if __name__ == '__main__':
    main()