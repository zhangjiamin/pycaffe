from layer import Layer
from blob import Blob
import numpy as np

class AccuracyLayer(Layer):

    def __init__(self):
        Layer.__init__(self)

    def LayerSetup(self, bottom, top):
        pass

    def Reshape(self, bottom, top):
        top[0].Reshape(0)
        top[1].Reshape(0)

    def type(self):
        return 'InnerProduct'

    def ExactNumBottomBlobs(self):
        return 2

    def ExactNumTopBlobs(self):
        return 2

    def Forward_cpu(self, bottom, top):
        count = np.sum( np.equal( np.argmax(bottom[0].data(),axis=1), np.argmax(bottom[1].data(),axis=1) ) )
        total = bottom[0].data().shape[0]
        top[0].set_data(count)
        top[1].set_data(total)
