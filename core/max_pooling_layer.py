
from layer import Layer
from blob import Blob
import numpy as np

class MaxPoolingLayer(Layer):

    def __init__(self):
        Layer.__init__(self)

    def LayerSetup(self, bottom, top):
        pass

    def Reshape(self, bottom, top):
        pass

    def type(self):
        return 'MaxPooling'

    def ExactNumBottomBlobs(self):
        return 1

    def MinTopBlobs(self):
        return 1

    def MaxTopBlobs(self):
        return 1

    def Forward_cpu(self, bottom, top):
        pass

    def Backward_cpu(self, top, propagate_down, bottom):
        pass


 
