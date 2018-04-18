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
        top[1].ReshapeLike(bottom[0])

    def type(self):
        return 'SoftmaxLoss'

    def ExactNumTopBlobs(self):
        return 2

    def MinTopBlobs(self):
        return 1

    def MaxTopBlobs(self):
        return 2

    def Forward_cpu(self, bottom, top):
        data = bottom[0].data()
        data1 = numpy.exp(data - numpy.max(data, axis=1, keepdims=True))
        self.probs_ = data1/numpy.sum(data1, axis=1, keepdims=True)
        Count = data.shape[0]
        label   = bottom[1].data()
        loss    = -numpy.sum(numpy.multiply((label), numpy.log(numpy.where(self.probs_>2.220446049250313e-16, self.probs_, 2.220446049250313e-16))))/Count
        top[0].set_data(loss)
        top[1].set_data(self.probs_)

    def Backward_cpu(self, top, propagate_down, bottom):
        label = bottom[1].data()
        bottom[0].set_diff( top[0].diff()*(self.probs_ - label)/label.shape[0] )

