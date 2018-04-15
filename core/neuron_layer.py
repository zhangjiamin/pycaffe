import numpy
from layer import Layer
from blob import Blob

class NeuronLayer(Layer):

    def __init__(self):
        Layer.__init__(self)

    def Reshape(self, bottom, top):
        top[0].ReshapeLike(bottom[0])

    def ExactNumBottomBlobs(self):
        return 1

    def ExactNumTopBlobs(self):
        return 1

class ExpLayer(NeuronLayer):

    def __init__(self):
        NeuronLayer.__init__(self)

    def type(self):
        return 'Exp'

    def LayerSetup(self, bottom, top):
        pass

    def Forward_cpu(self, bottom, top):
        top[0].set_data( numpy.exp(bottom[0].data()))

    def Backward_cpu(self, top, propagate_down, bottom):
        bottom[0].set_diff(top[0].data()*top[0].diff())


class ReLULayer(NeuronLayer):

    def __init__(self):
        NeuronLayer.__init__(self)

    def type(self):
        return 'ReLU'

    def Forward_cpu(self, bottom, top):
        top[0].set_data( numpy.maximum(bottom[0].data(), 0))

    def Backward_cpu(self, top, propagate_down, bottom):
        bottom[0].set_diff((bottom[0].data()>0)*top[0].diff())

class DropoutLayer(NeuronLayer):

    def __init__(self, threshold):
        NeuronLayer.__init__(self)
        self.threshold_ = threshold
        self.rand_blob_ = Blob()
        if 1.0 == threshold:
            self.scale_ = 1.0
        else:
            self.scale_     = 1.0/(1.0-threshold)

    def type(self):
        return 'Dropout'

    def LayerSetup(self, bottom, top):
        self.rand_blob_.ReshapeLike(bottom[0])
        self.rand_blob_.set_data( numpy.random.binomial(n=1, p=self.threshold_, size=bottom[0].data().shape) )

    def Reshape(self, bottom, top):
        NeuronLayer.Reshape(self, bottom, top)
        self.rand_blob_.ReshapeLike(bottom[0])

    def Forward_cpu(self, bottom, top):
        self.rand_blob_.set_data( numpy.random.binomial(n=1, p=self.threshold_, size=bottom[0].data().shape) )
        self.rand_blob_.ReshapeLike(bottom[0])
        top[0].set_data( numpy.multiply(bottom[0].data(), self.rand_blob_.data()*self.scale_) )

    def Backward_cpu(self, top, propagate_down, bottom):
        bottom[0].set_diff( numpy.multiply(top[0].diff(), self.rand_blob_.data()*self.scale_) )

if __name__ == '__main__':
    layer = DropoutLayer(0.5)
    bottom = Blob()
    top = Blob()

    bottom.Reshape((4,4))
    bottom.set_data( numpy.arange(16)*1.0 )
    bottom.Reshape((4,4))

    layer.Setup([bottom], [top])

    layer.Forward([bottom], [top])

    top.set_diff( numpy.arange(16)*1.0 )
    top.Reshape((4,4))
    layer.Backward([top], [], [bottom])
    
    print bottom.data()
    print layer.rand_blob_.data()
    print layer.scale_
    print top.data()
    print bottom.diff()
