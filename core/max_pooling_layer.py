from layer import Layer
from blob import Blob
import numpy as np
from max_pool_forward_naive import max_pool_forward_naive
from max_pool_backward_naive import max_pool_backward_naive

class MaxPoolingLayer(Layer):

    def __init__(self, pool_height, pool_width, stride):
        Layer.__init__(self)
        self.pool_height = pool_height
        self.pool_width  = pool_width
        self.stride      = stride

    def LayerSetup(self, bottom, top):
        pass

    def Reshape(self, bottom, top):
        N, C, H, W = bottom[0].data().shape
        H_out = 1 + (H - self.pool_height) / self.stride
        W_out = 1 + (W - self.pool_width) / self.stride
        top[0].Reshape((N, C, H_out, W_out))

    def type(self):
        return 'MaxPooling'

    def ExactNumBottomBlobs(self):
        return 1

    def MinTopBlobs(self):
        return 1

    def MaxTopBlobs(self):
        return 1

    def Forward_cpu(self, bottom, top):
        top[0].set_data(max_pool_forward_naive(bottom[0].data(), self.pool_height, self.pool_width, self.stride))

    def Backward_cpu(self, top, propagate_down, bottom):
        bottom[0].set_diff(max_pool_backward_naive(bottom[0].data(), top[0].diff(), self.pool_height, self.pool_width, self.stride))

