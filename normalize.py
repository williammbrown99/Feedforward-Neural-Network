class Normalize(object):
    '''
    A class used to normalize data.
    ...
    Attributes:
    data: data being normalized
    mean: sum/len of data
    variance: sum((x - mean)**2)/len of data
    stdDev = sqrt(variance)
    ...
    Functions:
    zScore: returns zScore normalized data
    revZScore: returns original data from zScores
    '''
    def __init__(self, data: list):
        self.mean = sum(data)/len(data)
        self.variance = sum([((x - self.mean) ** 2) for x in data]) / len(data)
        self.stdDev = self.variance ** 0.5

    '''Functions'''
    def zScore(self, data: list) -> list:
        return [(x - self.mean)/self.stdDev for x in data]

    def revZScore(self, zScores: list) -> list:
        return [x*self.stdDev + self.mean for x in zScores]