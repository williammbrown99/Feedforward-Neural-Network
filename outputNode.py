class OutputNode(object):
    '''
    A class used to represent an output node
    ...
    Attributes:
    inputs: List of hidden layer nodes' outputs
    weights: List of weights for each input
    ...
    Functions:
    weightedSum: returns sum
    out: returns weighted sum of all inputs * weights
    '''
    def __init__(self, inputs: list):
        self.inputs = inputs
        self.weights = []
        for i in range(len(inputs)):
            self.weights.append(0.8263728) #Random Value

    '''Functions'''
    def weightedSum(self, inputs: list, weights: list) -> float:
        #Weighted Sum
        weightsum = 0
        for i in range(len(inputs)):
            weightsum += inputs[i]*weights[i]
        return weightsum

    @property
    def out(self) -> float:
        return self.weightedSum(self.inputs, self.weights)