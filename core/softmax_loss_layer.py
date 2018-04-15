import numpy
from loss_layer import LossLayer
from blob import Blob

class SoftmaxLossLayer(LossLayer):

    def __init__(self):
        LossLayer.__init__(self)
        self.probs_ = None

    def LayerSetup(self, bottom, top):
        LossLayer.LayerSetup(self, bottom, top)

    def Reshape(self, bottom, top):
        LossLayer.Reshape(self, bottom, top)

    def type(self):
        return 'SoftmaxLoss'

    def ExactNumTopBlobs(self):
        return -1

    def MinTopBlobs(self):
        return 1

    def MaxTopBlobs(self):
        return 2

    def Forward_cpu(self, bottom, top):
        data = bottom[0].data()
        data1 = numpy.exp(data - numpy.max(data, axis=1, keepdims=True))
        self.probs_ = data1/numpy.sum(data1, axis=1, keepdims=True)
        N = data.shape[0]
        label   = bottom[1].data()
        probs   = numpy.sum(numpy.multiply( label, self.probs_),axis=1)
        loss    = -numpy.sum(numpy.log(numpy.where(probs>1.175494351e-38, probs, 1.175494351e-38)))/N
        top[0].set_data(loss)

    def Backward_cpu(self, top, propagate_down, bottom):
        label = bottom[1].data()
        bottom[0].set_diff( top[0].diff()*(self.probs_ - label)/self.probs_.shape[0] )

