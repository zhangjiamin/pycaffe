
class Net:

    def __init__(self):
        self.layers_  = []
        self.bottoms_ = []
        self.tops_    = []
        self.learnable_params_ = []

    def Forward(self, loss):
        loss_ = 0
        for i in range(len(layers_)):
            loss_ += self.layers_[i].Forward(self.bottoms_[i], self.tops_[i])

        loss = loss_

    def ClearParamDiffs(self):
        for i in range(len(self.learnable_params_)):
            self.learnable_params_[i].set_diff( numpy.zeros(blobs[ii].shape()) )

    def Backward(self):
        for i in reversed(range(len(self.layers_))):
            self.layers_[i].Backward(self.tops_[i], [], self.bottoms_[i])

    def Reshape(self):
        pass

    def ForwardBackward(self):
        loss =0;
        self.Forward(loss);
        self.Backward();
        return loss;

    def Update(self):
        for i in range(len(self.learnable_params_)):
            self.learnable_params_[i].Update()
 
if __name__ == '__main__':
    print 'Net Class'

