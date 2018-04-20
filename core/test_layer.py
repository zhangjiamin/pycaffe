import unittest
import numpy
import numpy as np
from blob import Blob
from load_data import load_data
from net import Net
from sgd_solver import SGDSolver
from adagrad_solver import AdaGradSolver
from adam_solver import AdamSolver
from nesterov_solver import NesterovSolver
from rmsprop_solver import RMSPropSolver

from accuracy_layer import AccuracyLayer
#from mnist_train_data_layer import MNISTTrainDataLayer
#from mnist_test_data_layer import MNISTTestDataLayer
from new_mnist_train_data_layer import MNISTTrainDataLayer
from new_mnist_test_data_layer import MNISTTestDataLayer
from inner_product_layer import InnerProductLayer
from softmax_layer import SoftMaxLayer
from conv_layer import ConvolutionLayer
from max_pooling_layer import MaxPoolingLayer
from softmax_loss_layer import SoftmaxLossLayer
from neuron_layer import ReLULayer
from neuron_layer import DropoutLayer

class TestLayer(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ReLULayer(self):
        bottom_0 = Blob()

        bottom_0.Reshape([2,6])

        bottom_0.set_data([-1.0,2.0,-3.0,4.0,-5.0,6.0,-1.0,2.0,-3.0,4.0,-5.0,6.0])

        bottom_0.Reshape([2,6])

        top = Blob()

        top.Reshape([2,6])

        top.set_diff([1.0,-2.0,3.0,-4.0,5.0,-6.0,1.0,-2.0,3.0,-4.0,5.0,-6.0])

        top.Reshape([2,6])

        layer = ReLULayer()

        for i in range(1):
            layer.Setup([bottom_0], [top])
            layer.Forward([bottom_0], [top])
            print 'ReLU:'

            print 'bot.data(%d):',i,bottom_0.data()
            print 'top.data(%d):',i,top.data()
            #top.set_diff([1.0,2.0,3.0,4.0,5.0,6.0,1.0,2.0,3.0,4.0,5.0,6.0])
            layer.Backward([top], [], [bottom_0])
            print 'top.diff(%d):',i,top.diff()
            print 'bot.diff(%d):',i,bottom_0.diff()

    def test_SoftmaxLossLayer(self):
        bottom_0 = Blob()
        bottom_1 = Blob()

        bottom_0.Reshape([2,6])
        bottom_1.Reshape([2,6])

        bottom_0.set_data([1.0,2.0,3.0,4.0,5.0,6.0,1.0,2.0,3.0,4.0,5.0,6.0])
        bottom_1.set_data([1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0])

        bottom_0.Reshape([2,6])
        bottom_1.Reshape([2,6])

        top = Blob()
        top1 = Blob()
        top.set_diff(10.0)

        layer = SoftmaxLossLayer()

        for i in range(1):
            layer.Setup([bottom_0, bottom_1], [top,top1])
            layer.Forward([bottom_0, bottom_1], [top,top1])
            print 'SoftmaxLoss:'

            print 'bot.data(%d):',i,bottom_0.data()
            print 'label.data(%d):',i,bottom_1.data()
            print 'top.data(%d):',i,top.data()
            print 'top1.data(%d):',i,top1.data()
            top.set_diff(1.0)
            layer.Backward([top,top1], [], [bottom_0, bottom_1])
            print 'bot.diff(%d):',i,bottom_0.diff()

    def test_ConvolutionLayer(self):
        bottom = Blob()
        top    = Blob()
        bottom.set_data(range(3*28*28))
        bottom.Reshape((1,3,28,28))
       
        layer  = ConvolutionLayer(3,3,1,0,1)
        layer.Setup([bottom], [top])
        layer.Forward([bottom], [top])

        layer1 = MaxPoolingLayer(2, 2, 1)
        top1   = Blob()

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
        bottom = Blob()
        top    = Blob()
        
        bottom.Reshape((1,2))
        bottom.set_data([1,2])
        bottom.Reshape((1,2))
       
        layer  = InnerProductLayer(2,2)
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
        bottom = Blob()
        top    = Blob()
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

    def test_mnist_mlp_net_solver(self):
        train_net = Net()
        test_net = Net()

        bottom = Blob()
        label  = Blob()
        top    = Blob()
        top1   = Blob()
        top2   = Blob()
        loss   = Blob()
        top4   = Blob()
        top5   = Blob()
        top6   = Blob()
        top7   = Blob()

        batch_size = 100

        test = MNISTTestDataLayer(batch_size)
        train = MNISTTrainDataLayer(batch_size)
        acc   = AccuracyLayer()

        fc1  = InnerProductLayer(784,300)
        relu = ReLULayer()
        drop = DropoutLayer(0.75)
        drop2 = DropoutLayer(1.0)
        fc2  = InnerProductLayer(300,10)
        softmaxloss = SoftmaxLossLayer()

        train_net.AddLayer(train, [], [bottom,label])
        train_net.AddLayer(fc1, [bottom], [top])
        train_net.AddLayer(relu, [top], [top1])
        train_net.AddLayer(drop, [top1], [top4])
        train_net.AddLayer(fc2, [top4], [top2])
        train_net.AddLayer(softmaxloss, [top2,label], [loss,top5])

        test_net.AddLayer(test, [], [bottom,label])
        test_net.AddLayer(fc1, [bottom], [top])
        test_net.AddLayer(relu, [top], [top1])
        test_net.AddLayer(drop2, [top1], [top4])
        test_net.AddLayer(fc2, [top4], [top2])
        test_net.AddLayer(softmaxloss, [top2,label], [loss,top5])
        test_net.AddLayer(acc, [top5,label], [top6,top7])

        test_net.AddOutputBlob(top6)
        test_net.AddOutputBlob(top7)

        solver = AdamSolver(0.001)
        solver.AddTrainNet(train_net)
        solver.AddTestNet(test_net)
        solver.Solve(3000)

if __name__ == '__main__':
    unittest.main()
