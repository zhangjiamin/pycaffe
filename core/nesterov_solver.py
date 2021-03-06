import numpy as np
from blob import Blob
from solver import Solver
from sgd_solver import SGDSolver

class NesterovSolver(SGDSolver):

    def __init__(self, lr):
        SGDSolver.__init__(self, lr)
        self.delta_ = 1e-8

    def type(self):
        return 'Nesterov'

    def AddTrainNet(self, net):
        Solver.AddTrainNet(self, net)
        params = self.net_.learnable_params()
        for i in range(len(params)):
            blob = Blob()
            blob.ReshapeLike(params[i])
            self.history_.append(blob)

    def ApplyUpdate(self):
        rate = self.GetLearningRate()
        self.ClipGradients()
        params = self.net_.learnable_params()
        for i in range(len(params)):
            self.Normalize(params[i])
            self.Regularize(params[i])
            self.ComputeUpdateValue(i, rate)

        self.net_.Update()

    def GetLearningRate(self):
        return self.lr_

    def Normalize(self, param):
        return
        param.set_diff(1.0/20000 * param.diff())

    def Regularize(self, param):
        return
        weight_decay = 0.005
        param.set_diff( param.diff() + weight_decay*param.data() )

    def ClipGradients(self):
        pass

    def ComputeUpdateValue(self, param_id, rate):
        param = self.net_.learnable_params()[param_id]
        history = self.history_[param_id]
        update = 1.0*history.data()
        history.set_data( rate*param.diff() + 0.9*history.data() )
        update = (1.0 + 0.9)*history.data() - 0.9*update
        param.set_diff( update )

