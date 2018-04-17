from blob import Blob
from solver import Solver

class SGDSolver(Solver):

    def __init__(self):
        Solver.__init__(self)
        self.lr_ = 0.01
        self.momentum_ = 0.9
        self.history_ = []

    def type(self):
        return 'SGD'

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
        weight_decay = 0.0005
        param.set_diff( param.diff() + weight_decay*param.data() )

    def ClipGradients(self):
        pass

    def ComputeUpdateValue(self, param_id, rate):
        param = self.net_.learnable_params()[param_id]
        history = self.history_[param_id]
        history.set_data(self.momentum_*history.data() + self.lr_*param.diff())
        param.set_diff( history.data() )


