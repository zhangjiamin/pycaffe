from layer import Layer
from blob import Blob
import numpy

class InnerProductLayer(Layer):

    def __init__(self):
        Layer.__init__(self)
        self.M_ = None
        self.K_ = None
        self.N_ = None

        self.bias_term_       = None
        self.bias_multiplier_ = None
        self.transpose_       = False

        self.W = Blob(numpy.float, (1,1))
        self.b = Blob(numpy.float, (1,))

    def LayerSetUp(self):
        pass

    def Reshape(self, bottom, top):
        bot_shape = list(bottom[0].shape())
        top_shape = list(top[0].shape())
        bot_shape.reverse()

        top_shape.extend(bot_shape)
        W_shape = top_shape
        b_shape = top[0].shape()

        self.W.Reshape(W_shape)
        self.b.Reshape(b_shape)

        # Xavier
        fan_in  = self.W.count()/self.W.shape()[0]
        fan_out = self.W.count()/self.W.shape()[1]

        n = (fan_in + fan_out)/2

        scale  = numpy.sqrt(3/n)
        self.W.set_data(numpy.random.uniform(-scale, scale, self.W.count()) )
        self.W.Reshape(W_shape)

    def type(self):
        return 'InnerProduct'

    def ExactNumBottomBlobs(self):
        return 1

    def ExactNumTopBlobs(self):
        return 1

    def Forward_cpu(self, bottom, top):
        top[0].set_data( numpy.matmul(self.W.data(), bottom[0].data()) + self.b.data() )

    def Backward_cpu(self, top, propagate_down, bottom):
        self.W.set_diff( numpy.matmul(top[0].diff(), bottom[0].data().transpose()) )
        self.b.set_diff( top[0].diff() )
