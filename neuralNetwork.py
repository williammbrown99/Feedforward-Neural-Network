import matplotlib.pyplot as plt
import hiddenNode
import outputNode

class NeuralNetwork(object):
    '''
    A class used to represent a neural network.
    ...
    Attributes:
    numHiddenNodes: number of nodes in the hidden layer
    hiddenLayer: list of the hidden nodes
    outputNode1: output node
    ...
    Functions:
    costFunction: sum of squares function
    feed_forward: feeds an input x through the network
    back_propogation: performs gradient descent using: new_weight = old_weight - lr*gradient
    sigmoidGradient: returns derivative of cost function with respect to hiddenNode[i].weight
    linearGradient: returns derivative of cost function with respect to outputNode.weights[i]
    train: trains the network and prints error every iteration
    iterationErrorGraph: plots the Error vs Iteration Graph
    predict: predicts y values using given x values
    '''
    def __init__(self, numHiddenNodes: int):
        self.numHiddenNodes = numHiddenNodes
        #Hidden layer = list of hiddenNodes
        self.hiddenLayer= []
        hiddenNodeOutputs = []
        for i in range(self.numHiddenNodes):
            #Hidden layer nodes recieves default value: -1
            self.hiddenLayer.append(hiddenNode.HiddenNode(-1))
            hiddenNodeOutputs.append(self.hiddenLayer[i].out)
        #Output node revieves all hidden node outputs
        self.outputNode1 = outputNode.OutputNode(hiddenNodeOutputs)

    '''Functions'''
    def mean_squared_error(self, y: list, predY: list) -> float:
        '''Calculating Mean Squared Error'''
        total = 0
        for i in range(len(y)):
            total += (y[i] - predY[i])**2
        return total/len(y)

    def feed_forward(self, x: float) -> float:
        '''feeding x through the network.'''
        hiddenNodeOutputs = []
        #Passing input X to each hidden node
        for i in range(self.numHiddenNodes):
            self.hiddenLayer[i].passX(x)
            hiddenNodeOutputs.append(self.hiddenLayer[i].out)
        #Passing all hidden node outputs to the output node
        self.outputNode1.inputs = hiddenNodeOutputs
        return self.outputNode1.out

    def back_propogation(self, lr: float) -> None:
        """
        Apply Back Propagation Algorithm to calculate gradients.
        
        Using actual Y value and learning rate for gradient descent
        Gradient Descent: new_weight = old_weight - lr*gradient
        Adjusting all weights in the network.
        """
        for i in range(self.numHiddenNodes):
            #Finding average of gradients for each batch
            avgHiddenGradient = sum(self.hiddenLayer[i].gradients)/len(self.hiddenLayer[i].gradients)
            avgOutputGradient = sum(self.outputNode1.gradients[i])/len(self.outputNode1.gradients[i])

            #Updating weights
            self.hiddenLayer[i].weight = self.hiddenLayer[i].weight - lr*avgHiddenGradient
            self.outputNode1.weights[i] = self.outputNode1.weights[i] - lr*avgOutputGradient

            #Updating hiddenNode sigmoid Derivatives
            #self.hiddenLayer[i].derivativeCalc()

            #Weights must be between -100 and 100
            if self.hiddenLayer[i].weight > 100:
                self.hiddenLayer[i].weight = 100
            elif self.hiddenLayer[i].weight < -100:
                self.hiddenLayer[i].weight = -100
            if self.outputNode1.weights[i] > 100:
                self.outputNode1.weights[i] = 100
            elif self.outputNode1.weights[i] < -100:
                self.outputNode1.weights[i] = -100

            #Reseting gradients to empty
            self.hiddenLayer[i].gradients = []
            self.outputNode1.gradients[i] = []

    def sigmoidGradient(self, y: float, index: int) -> float:
        '''
        Calculating the derivative of the sum of squares function with respect to the given hidden weight
        Using the chain rule
        '''
        return 2*(y - self.outputNode1.out)*(-self.outputNode1.weights[index]*self.hiddenLayer[index].sigmoidDerivative)

    def linearGradient(self, y: float, index: int) -> float:
        '''
        Calculating the derivative of the sum of squares function with respect to the given output weight
        Using the chain rule
        '''
        return 2*(y-self.outputNode1.out)*(-self.hiddenLayer[index].out)

    def train(self, X: list, Y: list, valX: list, valY: list, num_iterations: int, lr: float, batch_size: int) -> None:
        '''Training model'''
        index = 0 #Cycling through training data
        trainingErrors = [] #Training Errors used for graphing Error vs Iterations
        validationErrors = [] #Validation Errors used for graphing Error vs Iterations
        for i in range(num_iterations):
            print('Iteration: {}/{}'.format(i, num_iterations))
            #Creating Batches
            batchX = []
            batchY = []
            for j in range(batch_size):
                if index == len(X):
                    index = 0
                batchX.append(X[index])
                batchY.append(Y[index])
                index += 1
            
            #Running batches through model
            predictions = []
            for i in range(batch_size):
                #feeding x
                predictions.append(self.feed_forward(batchX[i]))
                #appending gradient to nodes
                for nodeNum in range(self.numHiddenNodes):
                    self.hiddenLayer[nodeNum].gradients.append(self.sigmoidGradient(batchY[i], nodeNum))
                    self.outputNode1.gradients[nodeNum].append(self.linearGradient(batchY[i], nodeNum))
            error = self.mean_squared_error(batchY, predictions)
            #Validation Testing
            valPredictions = []
            for i in range(len(valX)):
                valPredictions.append(self.feed_forward(valX[i]))
            valError = self.mean_squared_error(valY, valPredictions)
            print('Training Error: {}           Validation Error: {}'.format(error, valError))
            trainingErrors.append(error)
            validationErrors.append(valError)
            #Applying back propogation
            self.back_propogation(lr)
        #Graphing Error vs Iterations
        self.iterationErrorGraph(num_iterations, trainingErrors, validationErrors)

    def iterationErrorGraph(self, num_iterations: int, trainingErrors: list, validationErrors: list) -> None:
        '''Plotting Error vs Iterations'''
        plt.plot(range(num_iterations), trainingErrors, label='Training Error')
        plt.plot(range(num_iterations), validationErrors, label='Validation Error')
        
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        
        plt.title('Error vs Iterations')
        plt.legend()

        plt.show()

    def predict(self, X: list) -> list:
        '''Making predictions from list X'''
        predictedY = []
        for i in range(len(X)):
            predictedY.append(self.feed_forward(X[i]))
        return predictedY
        