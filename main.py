'''FeedForward Neural Network by William Brown

This file initializes a one hidden layer feedforward neural network to approximate a continuous function.
The sample data is z-score normalized. 80% is used for training, 20% for testing
The network is initialized with the number of hidden nodes in the hidden layer.
The network trains using parameters: trainingX, trainingY, numIterations, learningRate. batchSize
The network gives a prediction using the given test data.
'''

'''Imports'''
import normalize
import neuralNetwork
import random

'''Data Preprocessing'''
#Sample Data of 100 random X values
sampleX = [random.randint(0, 20) for i in range(100)]
#Actual function to approximate: f(x) = 2x - 5
sampleY = [2 * x - 5 for x in sampleX]

#z-score normalizing the sample data
normalizedData = normalize.Normalize(sampleX+sampleY)
normalizedX = normalizedData.zScore(sampleX)
normalizedY = normalizedData.zScore(sampleY)
#80% of sample data is used for training
trainingX = normalizedX[:80]
trainingY = normalizedY[:80]
#20% of sample data is used for testing
testX = normalizedX[80:]
testY = normalizedY[80:]

'''Parameters'''
numHiddenNodes = 9
numIterations = 25000
learningRate = 0.01
batchSize = 3

'''Training'''
nn = neuralNetwork.NeuralNetwork(numHiddenNodes)
nn.train(trainingX, trainingY, numIterations, learningRate, batchSize)

'''Predicting'''
print('Predicted Y:')
#Reversing z-score normalization and Rounding to nearest integer
print([round(y) for y in normalizedData.revZScore(nn.predict(testX))])
print('Actual Y:')
print(sampleY[80:])

