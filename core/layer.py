import numpy
from blob import Blob

class Layer:

    def __init__(self):
        self.blobs_ = []
        self.loss_  = []
        self.param_propagate_down_ = []

    def Setup(self, bottom, top):
        self.CheckBlobCounts(bottom, top)
        self.LayerSetup(bottom, top)
        self.Reshape(bottom, top)
        self.SetLossWeights(top)

    def LayerSetup(self, bottom, top):
        pass

    def Reshape(self, bottom, top):
        pass

    def Forward(self, bottom, top):
        loss = 0
        self.Reshape(bottom, top)
        self.Forward_cpu(bottom, top)

        for top_id in range(len(top)):
            if 0 != self.loss(top_id):
                loss = loss + numpy.dot(top[top_id].diff(), top[top_id].data())

        return loss

    def Backward(self, top, propagate_down, bottom):
        self.Backward_cpu(top, propagate_down, bottom)

    def blobs(self):
        return self.blobs_

    def layer_param(self):
        pass

    def loss(self, top_index):
        if len(self.loss_) <= top_index:
            return 0
        else: 
            return self.loss_[top_index]

    def set_loss(self, top_index, value):
        self.loss_[top_index] = value

    def type(self):
        return ''

    def ExactNumBottomBlobs(self):
        return -1

    def MinBottomBlobs(self):
        return -1

    def MaxBottomBlobs(self):
        return -1

    def ExactNumTopBlobs(self):
        return -1

    def MinTopBlobs(self):
        return -1

    def MaxTopBlobs(self):
        return -1

    def EqualNumBottomTopBlobs(self):
        return False

    def AutoTopBlobs(self):
        return False

    def AllowForceBackward(self, bottom_index):
        return True

    def param_propagate_down(self, param_id):
        return self.param_propagate_down_[param_id]

    def set_param_propagate_down(self, param_id, value): 
        self.param_propagate_down_[param_id] = value

    def Forward_cpu(self, bottom, top):
        pass

    def Forwad_gpu(self, bottom, top):
        pass

    def Backward_cpu(self, top, propagate_down, bottom):
        pass

    def Backward_gpu(self, top, propagate_down, bottom):
        pass

    def CheckBlobCounts(self, bottom, top):
        if self.ExactNumBottomBlobs() >= 0:
            if len(bottom) != self.ExactNumBottomBlobs():
                print self.type() + " Layer takes " + self.ExactNumBottomBlobs() + " bottom blob(s) as input."

    def SetLossWeights(self, top):
        pass


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


class InnerProductLayer(Layer):

    def __init__(self):
        Layer.__init__(self)
        self.M_ = None
        self.K_ = None
        self.N_ = None

        self.bias_term_       = None
        self.bias_multiplier_ = None
        self.transpose_       = False

    def LayerSetUp(self):
        pass

    def Reshape(self, bottom, top):
        pass

    def type(self):
        return 'InnerProduct'

    def ExactNumBottomBlobs(self):
        return 1

    def ExactNumTopBlobs(self):
        return 1

    def Forward_cpu(self, bottom, top):
        top[0].set_data( numpy.maximum(bottom[0].data(), 0))

    def Backward_cpu(self, top, propagate_down, bottom):
        bottom[0].set_diff((bottom[0].data()>0)*top[0].diff())
   

if __name__ == '__main__':
    bottom  = Blob(numpy.float, (2,3))
    top     = Blob(numpy.float, (2,3))
    top2    = Blob(numpy.float, (2,3))

    bottom.set_data([-2,-1,0,1,2,3])
    bottom.Reshape((2,3))

    top2.set_diff([-2,-1,0,1,2,5])
    top2.Reshape((2,3))

    layer1 = ReLULayer()
    layer1.Setup([bottom], [top])
    layer1.Forward([bottom], [top])

    layer2 = ExpLayer()
    layer2.Setup([top], [top2])
    layer2.Forward([top], [top2])

    layer2.Backward([top2], [], [top])
    layer1.Backward([top], [], [bottom])

    print 'bottom'
    print bottom.data()
    print 'top1'
    print top.data()
    print 'top2'
    print top2.data()

    print 'top2.diff'
    print top2.diff()
    print 'top1.diff'
    print top.diff()
    print 'bottom.diff'
    print bottom.diff()
