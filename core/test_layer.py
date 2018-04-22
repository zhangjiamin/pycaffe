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
from adadelta_solver import AdaDeltaSolver

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
        bottom_1 = Blob()
        bottom_0.Reshape([2,6])
        bottom_1.Reshape([2,6])
        bottom_0.set_data([-1.0,2.0,-3.0,4.0,-5.0,6.0,-1.0,2.0,-3.0,4.0,-5.0,6.0])
        bottom_1.set_data([0.0,2.0,0.0,4.0,0.0,6.0,0.0,2.0,0.0,4.0,0.0,6.0])
        bottom_0.Reshape([2,6])
        bottom_1.Reshape([2,6])

        top0 = Blob()
        top1 = Blob()
        top0.Reshape([2,6])
        top1.Reshape([2,6])
        top0.set_diff([1.0,-2.0,3.0,-4.0,5.0,-6.0,1.0,-2.0,3.0,-4.0,5.0,-6.0])
        top0.Reshape([2,6])
        top1.set_diff([0.0,-2.0,0.0,-4.0,0.0,-6.0,0.0,-2.0,0.0,-4.0,0.0,-6.0])
        top1.Reshape([2,6])

        layer = ReLULayer()

        layer.Setup([bottom_0], [top0])
        layer.Forward([bottom_0], [top0])
        np.testing.assert_array_equal( top0.data(), bottom_1.data() )

        layer.Backward([top0], [], [bottom_0])
        np.testing.assert_array_equal( bottom_0.diff(), top1.diff() )

    def test_SoftmaxLossLayer(self):
        bottom_0 = Blob()
        bottom_1 = Blob()

        bottom_0.Reshape([2,6])
        bottom_1.Reshape([2,6])

        bottom_0.set_data([1.0,2.0,3.0,4.0,5.0,6.0,1.0,2.0,3.0,4.0,5.0,6.0])
        bottom_1.set_data([1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0])

        bottom_0.Reshape([2,6])
        bottom_1.Reshape([2,6])

        top0 = Blob()
        top1 = Blob()

        top3 = Blob()
        top4 = Blob()
        top3.Reshape([2,6])
        top4.Reshape([2,6])

        top3.set_diff([-0.49786511,0.00580323,0.01577482,0.0428804,0.116561,0.31684566,-0.49786511,0.00580323,0.01577482,0.0428804,0.116561,0.31684566])
        top4.set_data([0.00426978, 0.01160646, 0.03154963, 0.08576079, 0.23312201,0.63369132,0.00426978, 0.01160646, 0.03154963, 0.08576079, 0.23312201,0.63369132])
        top3.Reshape([2,6])
        top4.Reshape([2,6])

        layer = SoftmaxLossLayer()

        layer.Setup([bottom_0, bottom_1], [top0,top1])
        layer.Forward([bottom_0, bottom_1], [top0,top1])

        np.testing.assert_array_almost_equal( top1.data(), top4.data() )

        top0.set_diff(1.0)
        layer.Backward([top0,top1], [], [bottom_0, bottom_1])

        np.testing.assert_array_almost_equal( bottom_0.diff(), top3.diff() )

    def test_SoftMaxLayer(self):
        bottom = Blob()
        top0    = Blob()
        top1    = Blob()
        top2    = Blob()
        top3    = Blob()

        bottom.Reshape((3,))
        bottom.set_data([1,2,3])
        bottom.Reshape((3,))
        top0.Reshape((3,))
        top1.Reshape((3,))
        top2.Reshape((3,))
        top3.Reshape((3,))

        top1.set_data([0.09003057, 0.24472847, 0.66524096])
        top1.Reshape((3,))
        top2.set_data([1.0, 1.0, 1.0])
        top2.Reshape((3,))
        top3.set_data([0.0, 0.0, 0.0])
        top3.Reshape((3,))

        layer  = SoftMaxLayer()
        layer.Setup([bottom], [top0])
        layer.Forward([bottom], [top0])

        np.testing.assert_array_almost_equal( top0.data(), top1.data() )

        top0.set_diff(top2.data())
        layer.Backward([top0], [], [bottom])
        
        np.testing.assert_array_almost_equal( bottom.diff(), top3.data() )

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

if __name__ == '__main__':
    unittest.main()
