from base_conv_layer import BaseConvolutionLayer
import numpy as np
from layer import Layer
from blob import Blob
from conv_forward_naive import conv_forward_naive
from conv_backward_naive import conv_backward_naive

class ConvolutionLayer(BaseConvolutionLayer):

    def __init__(self, hh, ww, fout):
        BaseConvolutionLayer.__init__(self, hh, ww, fout)

    def Forward_cpu(self, bottom, top):
        out = conv_forward_naive(bottom[0].data(), self.W, self.b, 0, 1)
        top[0].set_data(out)

    def Backward_cpu(self, top, propagate_down, bottom):
        dw, db, _ = conv_backward_naive(bottom[0].data(), self.W, self.b, top[0].diff(), 0, 1)
        self.W.set_diff(dw)
        self.b.set_diff(db)

