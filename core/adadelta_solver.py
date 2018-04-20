import numpy as np
from blob import Blob
from solver import Solver
from sgd_solver import SGDSolver

class AdaDeltaSolver(SGDSolver):

    def __init__(self, lr):
        SGDSolver.__init__(self, lr)
        self.delta_ = 1e-8
        self.history_ = []
        self.update_ = []
        self.temp_ = []
        self.momentum_ = 0.9

    def type(self):
        return 'AdaDelta'

    def AddTrainNet(self, net):
        Solver.AddTrainNet(self, net)
        params = self.net_.learnable_params()
        for i in range(len(params)):
            h1 = Blob()
            h2 = Blob()
            u = Blob()
            t = Blob()
            h1.ReshapeLike(params[i])
            h2.ReshapeLike(params[i])
            u.ReshapeLike(params[i])
            t.ReshapeLike(params[i])
            self.history_.append(h1)
            self.history_.append(h2)
            self.update_.append(u)
            self.temp_.append(t)

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
        update = np.square(param.diff())
        s = self.s_[param_id]    
        r = self.r_[param_id]
        s.set_data(self.b1_*s.data() + (1.0-self.b1_)*update )
        update = s.data() + self.delta_

        r.set_data(self.b2_*r.data() + (1.0-self.b2_)*np.square(param.diff()))
        s_ = s.data()/(1.0-np.power(self.b1_,(self.iter_)))
        r_ = r.data()/(1.0-np.power(self.b2_,(self.iter_)))
        runing_lr = rate*s_/(np.sqrt(r_)+self.delta_)       
        param.set_diff( runing_lr )

