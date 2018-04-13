from layer import Layer
from blob import Blob
import numpy

class InnerProductLayer(Layer):

    def __init__(self, M, K, N):
        Layer.__init__(self)

        # batch size
        self.M_ = M

        # input number of neuron
        self.K_ = K

        # output number of neuron
        self.N_ = N

        self.bias_term_       = None
        self.bias_multiplier_ = None
        self.transpose_       = False

        self.W = Blob()
        self.b = Blob()
        self.blobs_.append(self.W)
        self.blobs_.append(self.b)

    def LayerSetup(self, bottom, top):
        W_shape = (self.K_, self.N_)
        b_shape = (self.M_, self.N_)

        self.W.Reshape(W_shape)
        self.b.Reshape(b_shape)

        # Xavier
        fan_in  = self.W.count()/self.W.shape()[0]
        fan_out = self.W.count()/self.W.shape()[1]

        n = (fan_in + fan_out)/2

        scale  = numpy.sqrt(3.0/n)
        self.W.set_data(numpy.random.uniform(-scale, scale, self.W.count()) )
        self.W.Reshape(W_shape)

    def Reshape(self, bottom, top):
        top_shape = (self.M_, self.N_)
        top[0].Reshape(top_shape)

    def type(self):
        return 'InnerProduct'

    def ExactNumBottomBlobs(self):
        return 1

    def ExactNumTopBlobs(self):
        return 1

    def Forward_cpu(self, bottom, top):
        top[0].set_data( numpy.matmul(bottom[0].data(), self.W.data()) + self.b.data() )

    def Backward_cpu(self, top, propagate_down, bottom):
        self.W.set_diff( numpy.matmul(bottom[0].data().transpose(), top[0].diff()) )
        self.b.set_diff( top[0].diff() )
