import numpy
from layer import Layer
from blob import Blob

class LossLayer(Layer):

    def __init__(self):
        Layer.__init__(self)

    def LayerSetup(self, bottom, top):
        pass

    def Reshape(self, bottom, top):
        self.shape_ = 0
        top[0].Reshape(0)

    def ExactNumBottomBlobs(self):
        return 2

    def AutoTopBlobs(self):
        return True

    def ExactNumTopBlobs(self):
        return 1

    def AllowForceBackward(self, bottom_index):
        return (bottom_index != 1)

class EuclideanLossLayer(LossLayer):

    def __init__(self):
        LossLayer.__init__(self)
        self.diff_ = Blob(numpy.float, [6])

    def Reshape(self, bottom, top):
        LossLayer.Reshape(self, bottom, top)
        self.diff_.ReshapeLike(bottom[0])

    def type(self):
        return 'EuclideanLoss'

    def AllowForceBackward(self, bottom_index):
        return True

    def Forward_cpu(self, bottom, top):
        self.diff_.set_data(bottom[0].data() - bottom[1].data())
        dot = numpy.dot(self.diff_.data(), self.diff_.data())
        loss = dot / bottom[0].shape()[0] / 2
        top[0].set_data(loss)

    def Backward_cpu(self, top, propagate_down, bottom):
        pass

if __name__ == '__main__':
    bottom_0 = Blob(numpy.float, [6])
    bottom_1 = Blob(numpy.float, [6])

    bottom_0.set_data([1,2,3,4,5,6])
    bottom_1.set_data([2,2,4,4,6,9])

    bottom_0.Reshape([6])
    bottom_1.Reshape([6])

    top = Blob(numpy.float, 0)

    layer = EuclideanLossLayer()
    layer.Setup([bottom_0, bottom_1], [top])
    layer.Forward([bottom_0, bottom_1], [top])

    print top.data() 
