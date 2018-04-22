from layer import Layer
from blob import Blob
import numpy

class SoftMaxLayer(Layer):

    def __init__(self):
        Layer.__init__(self)

    def Reshape(self, bottom, top):
        top[0].ReshapeLike(bottom[0])

    def type(self):
        return 'SoftMax'

    def ExactNumBottomBlobs(self):
        return 1

    def ExactNumTopBlobs(self):
        return 1

    def Forward_cpu(self, bottom, top):
        top[0].set_data( numpy.exp(bottom[0].data())/numpy.sum(numpy.exp(bottom[0].data())) )
        

    def Backward_cpu(self, top, propagate_down, bottom):
        top_data = top[0].data()
        top_diff = top[0].diff()

        dot      = numpy.dot(top_data, top_diff)
        topdiff  = top_diff - dot
        bottom[0].set_diff( numpy.multiply(top_data, topdiff) )

