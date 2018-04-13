import numpy
from layer import Layer

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


if __name__ == '__main__':
    layer = ExpLayer()

