
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
        pass

    def Backward(self):
        for i in reversed(range(len(self.layers_))):
            self.layers_[i].Backward(self.tops_[i], [], self.bottoms_[i])

    def Reshape(self):
        pass

    def ForwardBackward(self):
        pass

    def Update(self):
        pass
 

if __name__ == '__main__':
    print 'Net Class'
