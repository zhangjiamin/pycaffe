from solver import Solver

class SGDSolver(Solver):

    def __init__(self):
        Solver.__init__(self)

    def type(self):
        return 'SGD'

    def ApplyUpdate(self):
        pass

    def GetLearningRate(self):
        pass

    def Normalize(self, param):
        pass

    def Regularize(self, param):
        pass

    def ClipGradients(self):
        pass

    def ComputeUpdateValue(self, param, rate):
        pass

