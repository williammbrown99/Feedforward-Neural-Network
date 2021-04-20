import random

e = 2.718281828 #Approximate e value

class HiddenNode(object):
    '''
    A class used to represent a hidden node
    ...
    Attributes:
    inputX: X input
    weight: weight for X input
    sigmoidDerivative: derivative of sigmoid function for gradient descent
    ...
    Functions:
    sigmoid: returns 1/(1+e**(-x))
    out: returns sigmoid(inputX, weight)
    '''
    def __init__(self, inputX: float):
        self.inputX = inputX
        self.weight = random.random() #Random Value
        #Updating Derivative
        self.passX(inputX, 1) #init nextWeight = 1
        #Gradients used to update weights
        self.gradients = []

    '''Functions'''
    def passX(self, x: float, nextWeight: float) -> None:
        self.inputX = x
        #sigmoid dervivative used for gradient descent
        #derivative of sigmoid = f(x)*f(-x)
        self.sigmoidDerivative = (self.sigmoid(self.inputX*self.weight)*self.sigmoid(-self.inputX*self.weight)
                                    *self.inputX*nextWeight)

    def addGradient(self, actualY: float, predY: float) -> None:
        '''
        Calculating the derivative of the sum of squares function with respect to the given hidden weight
        Using the chain rule
        '''
        self.gradients.append(2*(actualY - predY)*(-self.sigmoidDerivative))

    def sigmoid(self, x: float) -> float:
        '''Sigmoid Activation Function'''
        #Avoiding Overflow
        if x > 500:
            return 1.0
        elif x < -500:
            return 0.0
        return 1/(1+e**(-x))

    @property
    def out(self) -> float:
        return self.sigmoid(self.inputX*self.weight)