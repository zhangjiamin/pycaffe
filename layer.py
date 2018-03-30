import numpy
from blob import Blob

class Layer:

    def __init__(self):
        self.blobs_ = []
        self.loss_  = []
        self.param_propagate_down_ = []

    def Setup(self, bottom, top):
        pass

    def LayerSetup(self, bottom, top):
        pass

    def Reshape(self, bottom, top):
        pass

    def Forward(self, bottom, top):
        pass

    def Backward(self, top, propagate_down, bottom):
        pass

    def blobs(self):
        pass

    def layer_param(self):
        pass

    def loss(self, top_index):
        pass

    def set_loss(self, top_index, value):
        pass

    def type(self):
        return ''

    def ExactNumBottomBlobs(self):
        return -1

    def MinBottomBlobs(self):
        return -1

    def MaxBottomBlobs(self):
        return -1

    def ExactNumTopBlobs(self):
        return -1

    def MinTopBlobs(self):
        return -1

    def MaxTopBlobs(self):
        return -1

    def EqualNumBottomTopBlobs(self):
        return False

    def AutoTopBlobs(self):
        return False

    def AllowForceBackward(self, bottom_index):
        return True

    def param_propagate_down(self, param_id):
        return self.param_propagate_down_[param_id]

    def set_param_propagate_down(self, param_id, value): 
        self.param_propagate_down_[param_id] = value

    

if __name__ == '__main__':
    blob  = Blob(numpy.float, (2,3))
    layer = Layer()


