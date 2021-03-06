import numpy as np
from blob import Blob
from solver import Solver
from sgd_solver import SGDSolver

class RMSPropSolver(SGDSolver):

    def __init__(self, lr):
        SGDSolver.__init__(self, lr)
        self.delta_ = 1e-8

    def type(self):
        return 'RMSProp'

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
        param2 = np.square(param.diff())
        history.set_data(0.99*history.data() + (1-0.99)* param2)
        his_sqrt = rate/(np.sqrt(history.data()) + self.delta_)
        param.set_diff( his_sqrt*param.diff() )

