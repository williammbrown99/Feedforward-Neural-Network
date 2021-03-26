e = 2.718281828 #Approximate e value

class HiddenNode(object):
    '''
    A class used to represent a hidden node
    ...
    Attributes:
    inputX: X input
    weight: weight for X input
    weightDerivative: Calculation used for gradient descent for weight value
    ...
    Functions:
    sigmoid: returns 1/(1+e**(-x))
    out: returns sigmoid(inputX, weight)
    '''
    def __init__(self, inputX: float):
        self.inputX = inputX
        self.weight = 0.293873 #Random Value
        #sigmoid dervivative used for gradient descent
        #derivative of sigmoid = f(x)*f(-x)
        self.weightDerivative = -self.sigmoid(self.inputX*self.weight)*self.sigmoid(-self.inputX*self.weight)

    '''Functions'''
    def sigmoid(self, x: float) -> float:
        #Sigmoid Activation Function
        return 1/(1+e**(-x))

    @property
    def out(self) -> float:
        return self.sigmoid(self.inputX*self.weight)