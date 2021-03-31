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
    hiddenGradient: returns derivative of cost function with respect to hiddenNode[i].weight
    outputGradient: returns derivative of cost function with respect to outputNode.weights[i]
    train: trains the network and prints error every iteration
    predict: predicts y values using given x values
    '''
    def __init__(self, numHiddenNodes: int):
        self.numHiddenNodes = numHiddenNodes
        #Hidden layer nodes recieves default value: -1
        self.hiddenLayer= []
        hiddenNodeOutputs = []
        for i in range(self.numHiddenNodes):
            self.hiddenLayer.append(hiddenNode.HiddenNode(-1))
            hiddenNodeOutputs.append(self.hiddenLayer[i].out)
        #Output node revieves all hidden node outputs
        self.outputNode1 = outputNode.OutputNode(hiddenNodeOutputs)

    '''Functions'''
    def costFunction(self, y: float, predY: float) -> float:
        '''Using sum of squares as error function'''
        return (y - predY)**2

    def feed_forward(self, x: float) -> float:
        '''feeding x through the network.'''
        hiddenNodeOutputs = []
        #Passing input X to each hidden node
        for i in range(self.numHiddenNodes):
            self.hiddenLayer[i].inputX = x
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

    def train(self, X: list, Y: list, num_iterations: int, lr: float, batch_size: int) -> None:
        '''Training model'''
        index = 0 #Cycling through training data
        trainingErrors = [] #Training Errors used for graphing Error vs Iterations
        for i in range(num_iterations):
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
            error = 0
            for i in range(batch_size):
                #feeding x
                self.feed_forward(batchX[i])
                #adding error
                error += self.costFunction(batchY[i], self.outputNode1.out)
                #appending gradient to nodes
                for nodeNum in range(self.numHiddenNodes):
                    self.hiddenLayer[nodeNum].gradients.append(self.sigmoidGradient(batchY[i], nodeNum))
                    self.outputNode1.gradients[nodeNum].append(self.linearGradient(batchY[i], nodeNum))
            print('Error: {}'.format(error))
            trainingErrors.append(error)
            #Applying back propogation
            self.back_propogation(lr)
        #Graphing Error vs Iterations
        self.iterationErrorGraph(num_iterations, trainingErrors)

    def iterationErrorGraph(self, num_iterations: int, trainingErrors: list) -> None:
        '''Plotting Error vs Iterations'''
        # plotting the points 
        plt.plot(range(num_iterations), trainingErrors)
        
        plt.xlabel('iterations')
        plt.ylabel('Error')
        
        plt.title('Error vs Iterations')
        plt.show()

    def predict(self, X: list) -> list:
        '''Making predictions from list X'''
        predictedY = []
        for i in range(len(X)):
            self.feed_forward(X[i])
            predictedY.append(self.outputNode1.out)
        return predictedY