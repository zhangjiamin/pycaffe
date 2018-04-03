import numpy
from blob import Blob

class Layer:

    def __init__(self):
        self.blobs_ = []
        self.loss_  = []
        self.param_propagate_down_ = []
        self.layer_param_ = None

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
        return self.layer_param_

    def ToProto(self, param, write_diff=False):
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

        if self.MinBottomBlobs() >= 0:
            if len(bottom) < self.MinBottomBlobs():
                print self.type() + " Layer takes at least " + self.MinBottomBlobs() + " bottom blob(s) as input."

        if self.MaxBottomBlobs() >= 0:
            if len(bottom) > self.MaxBottomBlobs():
                print self.type() + " Layer takes at most " + self.MaxBottomBlobs() + " bottom blob(s) as input."

        if self.ExactNumTopBlobs() >= 0:
            if len(top) != self.ExactNumTopBlobs():
                print self.type() + " Layer produces " + self.ExactNumTopBlobs() + " top blob(s) as output."

        if self.MinTopBlobs() >= 0:
            if len(top) < self.MinTopBlobs():
                print self.type() + " Layer produces at least " + self.MinTopBlobs() + " top blob(s) as output."

        if self.MaxTopBlobs() >= 0:
            if len(top) > self.MaxTopBlobs():
                print self.type() + " Layer produces at most " + self.MaxTopBlobs() + " top blob(s) as output."

        if self.EqualNumBottomTopBlobs() == True:
            if len(top) != len(bottom):
                print self.type() + " Layer produces one top blob as output for each " + "bottom blob input."


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
        b_shape = top_shape

        self.W.Reshape(W_shape)
        self.b.Reshape(b_shape) 

    def type(self):
        return 'InnerProduct'

    def ExactNumBottomBlobs(self):
        return 1

    def ExactNumTopBlobs(self):
        return 1

    def Forward_cpu(self, bottom, top):
        top[0].set_data( numpy.matmul(self.W.data(), bottom[0].data()))

    def Backward_cpu(self, top, propagate_down, bottom):
        top_shape = list(top[0].shape())
        bot_shape = list(bottom[0].shape())
        bot_shape.reverse()

        t = top[0].diff().reshape(top_shape)
        b = bottom[0].data().transpose().reshape(bot_shape)
        mul = numpy.tensordot(t, b, axes=0)

        self.W.set_diff( mul )
        self.W.Reshape(self.W.shape())

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

    bottom.set_data([-2,-1,0,1,2,3])
    bottom.Reshape((6,))

    top.Reshape((5,))
    top.set_diff([-2,-1,0,1,2])
    top.Reshape((5,))

    layer = InnerProductLayer()
    layer.Setup([bottom], [top])
    layer.Forward([bottom], [top])
    print 'top.diff:'
    print top.diff()
 
    print 'W:'
    print layer.W.data()
    print 'InnerProduct:'
    print top.data()
    print 'Befor Backward:'
    print layer.W.diff()
    layer.Backward([top], [], [bottom])
    print 'Backward:'
    print layer.W.diff()
    print bottom.diff().shape
    print layer.W.data().shape
    print 'top.diff:'
    print top.diff()
    print 'bottom.data:'
    print bottom.data()
