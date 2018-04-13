import unittest
import numpy
import numpy as np
from blob import Blob

from inner_product_layer import InnerProductLayer
from softmax_layer import SoftMaxLayer
from conv_layer import ConvolutionLayer
from max_pooling_layer import MaxPoolingLayer
from softmax_loss_layer import SoftmaxLossLayer

class TestLayer(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_SoftmaxLossLayer(self):
        bottom_0 = Blob(numpy.float, [6])
        bottom_1 = Blob(numpy.float, [6])

        bottom_0.set_data([1.0,2.0,3.0,4.0,5.0,6.0])
        bottom_1.set_data([1.0,0.0,0.0,0.0,0.0,0.0])

        bottom_0.Reshape([1,6])
        bottom_1.Reshape([1,6])

        top = Blob(numpy.float, 0)
        top.set_diff(10.0)

        layer = SoftmaxLossLayer()

        for i in range(10):
            layer.Setup([bottom_0, bottom_1], [top])
            layer.Forward([bottom_0, bottom_1], [top])
            print 'SoftmaxLoss:'

            print 'bot.data(%d):',i,bottom_0.data()
            print 'top.data(%d):',i,top.data()
            top.set_diff(top.data())
            layer.Backward([top], [], [bottom_0, bottom_1])
            print 'bot.diff(%d):',i,bottom_0.diff()
            bottom_0.set_diff(bottom_0.diff()*(0.01))
            bottom_0.Update()

    def test_ConvolutionLayer(self):
        bottom = Blob(np.float, (1,3,28,28))
        top    = Blob(np.float, (28,28))
        bottom.set_data(range(3*28*28))
        bottom.Reshape((1,3,28,28))

        W       = Blob(np.float, (1,3,3,3))
        fan_in  = W.count()/3
        fan_out = W.count()/3

        n = (fan_in + fan_out)/2

        scale  = np.sqrt(3.0/n)
        W.set_data(np.random.uniform(-scale, scale, W.count()) )

        W.Reshape((1,3,3,3))
        print 'W:', W.data()
        
        layer  = ConvolutionLayer(3,3,1,0,1)
        layer.Setup([bottom], [top])
        layer.W = W
        layer.Forward([bottom], [top])

        layer1 = MaxPoolingLayer(2, 2, 1)
        top1    = Blob(np.float, (28,28))

        layer1.Forward([top], [top1])
        top1.set_diff(top1.data())
        layer1.Backward([top1], [], [top])
        layer.Backward([top], [], [bottom])

        print 'bottom',bottom.data(),bottom.data().shape
        print 'top',top.data(),top.data().shape
        print 'top1:',top1.data(),top1.data().shape
        print 'W',layer.W.data(),layer.W.data().shape
        print 'b',layer.b.data(),layer.b.data().shape
        print 'top1.diff',top1.diff(),top1.data().shape
        print 'top.diff',top.diff(),top.data().shape
        print 'W.diff',layer.W.diff(),layer.W.data().shape
        print 'b.diff',layer.b.diff(),layer.b.data().shape

    def test_InnerProductLayer(self):
        bottom = Blob(np.float, (2,))
        top    = Blob(np.float, (2,))
        bottom.set_data([1,2])
        bottom.Reshape((2,))
       
        layer  = InnerProductLayer(1,2,2)
        layer.Setup([bottom], [top])
        layer.Forward([bottom], [top])
        top.set_diff(top.data())
        layer.Backward([top], [], [bottom])

        print 'InnerProductLayer:'
        print 'bottom',bottom.data(),bottom.data().shape
        print 'top',top.data(),top.data().shape
        print 'W',layer.W.data(),layer.W.data().shape
        print 'b',layer.b.data(),layer.b.data().shape
        print 'top.diff',top.diff(),top.data().shape
        print 'W.diff',layer.W.diff(),layer.W.data().shape
        print 'b.diff',layer.b.diff(),layer.b.data().shape


    def test_SoftMaxLayer(self):
        bottom = Blob(np.float, (3,))
        top    = Blob(np.float, (3,))
        bottom.set_data([1,2,3])
        bottom.Reshape((3,))

        layer  = SoftMaxLayer()
        layer.Setup([bottom], [top])
        layer.Forward([bottom], [top])
        top.set_diff(top.data())
        layer.Backward([top], [], [bottom])

        print 'bottom',bottom.data(),bottom.data().shape
        print 'top',top.data(),top.data().shape
        print 'top.diff',top.diff(),top.data().shape
        print 'bottom.diff',bottom.diff(),bottom.diff().shape


if __name__ == '__main__':
    unittest.main()
