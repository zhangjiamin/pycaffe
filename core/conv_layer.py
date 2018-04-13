from base_conv_layer import BaseConvolutionLayer
import numpy
import numpy as np
from layer import Layer
from blob import Blob
from conv_forward_naive import conv_forward_naive
from conv_backward_naive import conv_backward_naive

class ConvolutionLayer(BaseConvolutionLayer):

    def __init__(self, C, hh, ww, fout, pad, stride):
        BaseConvolutionLayer.__init__(self, C, hh, ww, fout, pad, stride)

    def LayerSetup(self, bottom, top):
        W_shape = (self.fout, self.C, self.hh, self.ww)
        b_shape = (self.fout,)

        self.W.Reshape(W_shape)
        self.b.Reshape(b_shape)

        # Xavier
        fan_in  = self.W.count()/self.W.shape()[0]
        fan_out = self.W.count()/self.W.shape()[1]

        n = (fan_in + fan_out)/2

        scale  = numpy.sqrt(3.0/n)
        self.W.set_data(numpy.random.uniform(-scale, scale, self.W.count()) )
        self.W.Reshape(W_shape)

    def type(self):
        return 'Convolution'

    def Forward_cpu(self, bottom, top):
        out = conv_forward_naive(bottom[0].data(), self.W.data(), self.b.data(), self.pad, self.stride)
        top[0].set_data(out)

    def Backward_cpu(self, top, propagate_down, bottom):
        dw, db, _ = conv_backward_naive(bottom[0].data(), self.W.data(), self.b.data(), top[0].diff(), self.pad, self.stride)
        self.W.set_diff(dw)
        self.b.set_diff(db)

