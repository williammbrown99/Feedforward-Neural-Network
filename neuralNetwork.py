import hiddenNode
import outputNode

class NeuralNetwork(object):
    '''
    A class used to represent a neural network.
    ...
    Attributes:
    numNodes: number of nodes in the hidden layer
    hiddenLayer: list of hidden nodes
    outputNode1: output node
    ...
    Functions:
    costFunction: returns (y - predictedY)**2
    feed_forward: feeds an input x through the network
    back_propogation: performs gradient descent using: new_weight = old_weight - lr*gradient
    chainRuleHiddenWeight: returns derivative of cost function with respect to hiddenNode[i].weight
    chainRuleOutputWeight: returns derivative of cost function with respect to outputNode.weights[i]
    train: trains the network and prints error every iteration
    predict: predicts y values using given x values
    '''
    def __init__(self, numNodes: int):
        self.numNodes = numNodes
        #Network Architecture
        #Hidden layer: Sigmoid Activation Function
        self.hiddenLayer= []
        nodeOutputs = []
        for i in range(self.numNodes):
            self.hiddenLayer.append(hiddenNode.HiddenNode(-1))  # -1 = Default Value
            nodeOutputs.append(self.hiddenLayer[i].out)
        #Output layer: Linear Activation Function
        self.outputNode1 = outputNode.OutputNode(nodeOutputs)

    '''Functions'''
    def costFunction(self, y: float, predY: float) -> float:
        #Sum of squares
        return(y - predY)**2

    def feed_forward(self, x: float):
        nodeOutputs = []
        for i in range(self.numNodes):
            self.hiddenLayer[i].inputX = x
            nodeOutputs.append(self.hiddenLayer[i].out)
        self.outputNode1.inputs = nodeOutputs

    def back_propogation(self, y: float, lr: float):
        #Gradient Descent
        #new_weight = old_weight - lr*gradient
        for i in range(self.numNodes):
            self.hiddenLayer[i].weight = self.hiddenLayer[i].weight - lr*self.chainRuleHiddenWeight(y, i)
            self.outputNode1.weights[i] = self.outputNode1.weights[i] - lr*self.chainRuleOutputWeight(y, i)

    def chainRuleHiddenWeight(self, y: float, index: int) -> float:
        return 2*(y - self.outputNode1.out)*(self.hiddenLayer[index].weightDerivative)

    def chainRuleOutputWeight(self, y: float, index: int) -> float:
        return 2*(y-self.outputNode1.out)*-self.hiddenLayer[index].out

    def train(self, X: list, Y: list, num_iterations: int, lr: float):
        for i in range(num_iterations):
            error = 0
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