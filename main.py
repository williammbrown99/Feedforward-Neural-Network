'''FeedForward Neural Network by William Brown

This file initializes a feedforward neural network to approximate a continuous function.
The network trains using the given training data and parameters.
The network gives a prediction using the given test data.

This file does not import any external modules.
'''

import normalize
import neuralNetwork

#Sample Data
sampleX = [-3, 18, 4, -19, 0, 6, -2, 17, -11, -3, 12, -4, 18, 2, 17, 8, 15, -7, 13, -10] #20 random data points
sampleY = [2 * x - 5 for x in sampleX] #f(x) = 2x - 5

#Preprocessing Data
norm = normalize.Normalize(sampleX+sampleY)
normX = norm.zScore(sampleX)
normY = norm.zScore(sampleY)
#80% Training 20% Testing
trainingX = normX[:17]
trainingY = normY[:17]
testX = normX[17:]
testY = normY[17:]

#Parameters
numNodes = 100
numIterations = 1000
learningRate = 0.01

#Initializing and Training Network
nn = neuralNetwork.NeuralNetwork(numNodes)
nn.train(trainingX, trainingY, numIterations, learningRate)

#Predicting using Network
print('Predicted Y:')
print(norm.revZScore(nn.predict(testX)))
print('Actual Y:')
print(sampleY[17:])

