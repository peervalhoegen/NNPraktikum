#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from data.mnist_seven import MNISTSeven
from data.sin import Sin
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron
from report.evaluator import Evaluator
from data.data_set import DataSet
from model.layer import Layer
from model.logistic_layer import LogisticLayer
import matplotlib.pyplot as plt
from model.sin_Out_Layer import Sin_Out_Layer 
def main():
    numInp = 1
    numOut = 1
    numNeuronInp =2000
    numHiddenLayer = 2
    numNeuronHidden = 100
    epochs = 10
    inputLayer = Layer(numInp, numNeuronInp)
    hiddenLayer1 = Layer(numNeuronInp, numNeuronHidden)
    hiddenLayer = Layer(numNeuronHidden,numNeuronHidden )
    outputLayer = Sin_Out_Layer(numNeuronHidden, 1)#, activation = 'linear')
    layers = []
    layers.append(inputLayer)
    layers.append(hiddenLayer1)
    for i in range(numHiddenLayer-1):
        layers.append(hiddenLayer)
    layers.append(outputLayer)

 #   m=MultilayerPerceptron(layers)
 #   input=np.array([0,0])
    #,[0,1],[1,0],[1,1]]
 #   target=0
    #[0,1,1,0]
 #   print(m.computeError(input, target))
 #   m.updateWeights(input,target)
    

    data = Sin("", 1000, 100,10000)
#    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
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
					layers,
					outputTask='Regression',
			                inputWeights=None,
                                        learningRate=2,
                                        epochs=epochs)


   

    print("\nMultilayerPerceptron has been training..")
    myLRClassifier.train()
    print("Done..")

    

    pred = myLRClassifier.evaluate()
    plt.plot(pred[0],pred[1])
    plt.show()
    #print('pred' + str(pred))
    #eval = Evaluator()
    #eval.printConfusionMatrix(data.testSet, pred)
    #eval.printClassificationResult(data.testSet, pred, target_names)

if __name__ == '__main__':
    main()
