import random

class OutputNode(object):
    '''
    A class used to represent an output node
    ...
    Attributes:
    inputs: List of hidden layer nodes' outputs
    weights: List of weights for each input
    ...
    Functions:
    weightedSum: returns weighted sum using list of inputs and list of weights
    out: returns weightedSum of all inputs and weights
    '''
    def __init__(self, inputs: list):
        self.inputs = inputs
        self.weights = []
        for i in range(len(inputs)):
            self.weights.append(random.random()) #Random Value
        self.gradients = []
        #gradients used to update weights
        for i in range(len(self.weights)):
            self.gradients.append([])

    '''Functions'''
    def weightedSum(self, inputs: list, weights: list) -> float:
        weightsum = 0
        for i in range(len(inputs)):
            weightsum += inputs[i]*weights[i]
        return weightsum

    @property
    def out(self) -> float:
        #Linear Activation function: f(x) = x
        return self.weightedSum(self.inputs, self.weights)