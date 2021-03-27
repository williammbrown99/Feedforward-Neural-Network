'''FeedForward Neural Network by William Brown

This file initializes a one hidden layer feedforward neural network to approximate a continuous function.
The sample data is z-score normalized. 80% is used for training, 20% for testing
The network is initialized with the number of hidden nodes in the hidden layer.
The network trains using parameters: trainingX, trainingY, numIterations, learningRate.
The network gives a prediction using the given test data.

This file does not import any external modules.
'''

'''Imports'''
import normalize
import neuralNetwork

'''Data Preprocessing'''
#Sample Data of 20 random X values
sampleX = [-3, 18, 4, -19, 0, 6, -2, 17, -11, -3, 12, -4, 18, 2, 17, 8, 15, -7, 13, -10]
#Actual function to approximate: f(x) = 2x - 5
sampleY = [2 * x - 5 for x in sampleX]

#z-score normalizing the sample data
normalizedData = normalize.Normalize(sampleX+sampleY)
normalizedX = normalizedData.zScore(sampleX)
normalizedY = normalizedData.zScore(sampleY)
#80% of sample data is used for training
trainingX = normalizedX[:17]
trainingY = normalizedY[:17]
#20% of sample data is used for testing
testX = normalizedX[17:]
testY = normalizedY[17:]

'''Parameters'''
numHiddenNodes = 20
numIterations = 1000
learningRate = 0.01

#Initializing and Training Network
nn = neuralNetwork.NeuralNetwork(numHiddenNodes)
nn.train(trainingX, trainingY, numIterations, learningRate)

#Predicting using Network
print('Predicted Y:')
#Reversing z-score normalization
print(normalizedData.revZScore(nn.predict(testX)))
print('Actual Y:')
print(sampleY[17:])

