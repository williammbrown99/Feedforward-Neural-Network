'''FeedForward Neural Network by William Brown

This file initializes a feedforward neural network to approximate a continuous function.
The network trains using the given training data and parameters.
The network gives a prediction using the given test data.

This file does not import any external modules.
'''

import neuralNetwork

#Training Data
trainingX = [1, 3, 5, 7, 9, 11]
trainingY = [2, 6, 10, 14, 18, 22]

#Parameters
numNodes = 100
numIterations = 1000
learningRate = 0.01

#Initializing and Training Network
nn = neuralNetwork.NeuralNetwork(numNodes)
nn.train(trainingX, trainingY, numIterations, learningRate)

#Test Data
testX = [2, 4, 6, 8, 10]
testY = [4, 8, 12, 16, 20]

#Predicting using Network
print('Predicted Y:')
print(nn.predict(testX))
print('Actual Y:')
print(testY)