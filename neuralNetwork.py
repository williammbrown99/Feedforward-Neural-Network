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
        #Using sum of squares as error function
        return(y - predY)**2

    def feed_forward(self, x: float):
        hiddenNodeOutputs = []
        #Passing input X to each hidden node
        for i in range(self.numHiddenNodes):
            self.hiddenLayer[i].inputX = x
            hiddenNodeOutputs.append(self.hiddenLayer[i].out)
        #Passing all hidden node outputs to the output node
        self.outputNode1.inputs = hiddenNodeOutputs

    def back_propogation(self, y: float, lr: float):
        #Using actual Y value and learning rate for gradient descent
        #Gradient Descent: new_weight = old_weight - lr*gradient
        #Adjusting all weights in the network
        for i in range(self.numHiddenNodes):
            self.hiddenLayer[i].weight = self.hiddenLayer[i].weight - lr*self.hiddenGradient(y, i)
            self.outputNode1.weights[i] = self.outputNode1.weights[i] - lr*self.outputGradient(y, i)

    def hiddenGradient(self, y: float, index: int) -> float:
        #Calculating the derivative of the sum of squares function with respect to the given hidden weight
        #Using the chain rule
        return 2*(y - self.outputNode1.out)*(-self.outputNode1.weights[index]*self.hiddenLayer[index].sigmoidDerivative)

    def outputGradient(self, y: float, index: int) -> float:
        #Calculating the derivative of the sum of squares function with respect to the given output weight
        #Using the chain rule
        return 2*(y-self.outputNode1.out)*-self.hiddenLayer[index].out

    def train(self, X: list, Y: list, num_iterations: int, lr: float):
        for i in range(num_iterations):
            error = 0
            #Stochastic gradient descent: adjusting weights after each training X
            for j in range(len(X)):
                self.feed_forward(X[j])
                error += self.costFunction(Y[j], self.outputNode1.out)
                self.back_propogation(Y[j], lr)
            print('Error: {}'.format(error))

    def predict(self, X: list) -> list:
        predictedY = []
        for i in range(len(X)):
            self.feed_forward(X[i])
            predictedY.append(self.outputNode1.out)
        return predictedY