from solver import Solver

class SGDSolver(Solver):

    def __init__(self):
        Solver.__init__(self)
        self.lr_ = 0.01

    def type(self):
        return 'SGD'

    def ApplyUpdate(self):
        rate = self.GetLearningRate()
        self.ClipGradients()
        params = self.net_.learnable_params()
        for i in range(len(params)):
            self.Normalize(params[i])
            self.Regularize(params[i])
            self.ComputeUpdateValue(params[i], rate)

        self.net_.Update()

    def GetLearningRate(self):
        return self.lr_

    def Normalize(self, param):
        pass

    def Regularize(self, param):
        pass

    def ClipGradients(self):
        pass

    def ComputeUpdateValue(self, param, rate):
        pass

