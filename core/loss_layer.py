import numpy
from layer import Layer
from blob import Blob

class LossLayer(Layer):

    def __init__(self):
        Layer.__init__(self)
        self.loss_.append(1)

    def LayerSetup(self, bottom, top):
        pass

    def Reshape(self, bottom, top):
        self.shape_ = 0
        top[0].Reshape(0)
        top[0].set_diff(1.0)

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
        print top[0].diff()
        print bottom[0].shape()[0]
        print self.diff_.data()
        bottom[0].set_diff(top[0].diff()/bottom[0].shape()[0]*self.diff_.data())

if __name__ == '__main__':
    bottom_0 = Blob(numpy.float, [6])
    bottom_1 = Blob(numpy.float, [6])

    bottom_0.set_data([1.0,2.0,3.0,4.0,5.0,6.0])
    bottom_1.set_data([1.0,0.0,0.0,0.0,0.0,0.0])

    bottom_0.Reshape([1,6])
    bottom_1.Reshape([1,6])

    top = Blob(numpy.float, 0)
    top.set_diff(10.0)

    layer = EuclideanLossLayer()
    layer.Setup([bottom_0, bottom_1], [top])
    layer.Forward([bottom_0, bottom_1], [top])
    layer.Backward([top], [], [bottom_0, bottom_1])
    print top.data() 
    print bottom_0.diff()

