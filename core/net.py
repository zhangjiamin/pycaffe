
class Net:

    def __init__(self):
        self.layers_  = []
        self.bottoms_ = []
        self.tops_    = []
        self.learnable_params_ = []

    def Forward(self, loss):
        pass

    def ClearParamDiffs(self):
        pass

    def Backward(self):
        pass

    def Reshape(self):
        pass

    def ForwardBackward(self):
        pass

    def Update(self):
        pass
 

if __name__ == '__main__':
    print 'Net Class'
