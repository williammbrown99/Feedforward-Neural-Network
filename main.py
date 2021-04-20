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
#Sample Data of 100 random X integers between -20 and 20
sampleX = [random.randint(-20, 21) for i in range(100)]
#Actual function to approximate: f(x) = 3x - 5
sampleY = [3*x - 5 for x in sampleX]

#z-score normalizing the sample data
normalizedData = normalize.Normalize(sampleX+sampleY)
normalizedX = normalizedData.zScore(sampleX)
normalizedY = normalizedData.zScore(sampleY)
#60% of sample data is used for training
trainingX = normalizedX[:60]
trainingY = normalizedY[:60]
#20% of sample data is used for validation
validationX = normalizedX[60:80]
validationY = normalizedY[60:80]
#20% of sample data is used for testing
testX = normalizedX[80:]
testY = normalizedY[80:]

'''Parameters'''
numHiddenNodes = 16
numIterations = 500
learningRate = 0.03
batchSize = 10

'''Training'''
#One hidden layer Neural Network: (Universal approximation theorem)
nn = neuralNetwork.NeuralNetwork(numHiddenNodes)
nn.train(trainingX, trainingY, validationX, validationY, numIterations, learningRate, batchSize)

'''Evaluating'''
print('Predicted Y:')
#Reversing z-score normalization and Rounding to nearest integer
predictions = [round(y) for y in normalizedData.revZScore(nn.predict(testX))]
print(predictions)
print('Actual Y:')
print(sampleY[80:])
print('Test Error: {}'.format(nn.mean_squared_error(sampleY[80:], predictions)))

