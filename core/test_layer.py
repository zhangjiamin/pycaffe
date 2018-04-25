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
        # top0.data = ReLU(bot0.data)
        # bot0.diff = ReLUGrad(top0.diff) 
        bot0 = Blob()
        bot0.Reshape([2,6])
        bot0.set_data([-1.0,2.0,-3.0,4.0,-5.0,6.0,-1.0,2.0,-3.0,4.0,-5.0,6.0])
        bot0.Reshape([2,6])

        top0 = Blob()
        top0.Reshape([2,6])
        top0.set_diff([1.0,-2.0,3.0,-4.0,5.0,-6.0,1.0,-2.0,3.0,-4.0,5.0,-6.0])
        top0.Reshape([2,6])

        expect_bot0 = Blob()
        expect_bot0.Reshape([2,6])
        expect_bot0.set_diff([0.0,-2.0,0.0,-4.0,0.0,-6.0,0.0,-2.0,0.0,-4.0,0.0,-6.0])
        expect_bot0.Reshape([2,6])

        expect_top0 = Blob()
        expect_top0.Reshape([2,6])
        expect_top0.set_data([0.0,2.0,0.0,4.0,0.0,6.0,0.0,2.0,0.0,4.0,0.0,6.0])
        expect_top0.Reshape([2,6])

        layer = ReLULayer()

        layer.Setup([bot0], [top0])
        layer.Forward([bot0], [top0])
        np.testing.assert_array_equal( expect_top0.data(), top0.data() )

        layer.Backward([top0], [], [bot0])
        np.testing.assert_array_equal( expect_bot0.diff(), bot0.diff() )

    def test_SoftmaxLossLayer(self):
        bot0 = Blob()
        bot0.Reshape([2,6])
        bot0.set_data([1.0,2.0,3.0,4.0,5.0,6.0,1.0,2.0,3.0,4.0,5.0,6.0])
        bot0.Reshape([2,6])

        bot1 = Blob()
        bot1.Reshape([2,6])
        bot1.set_data([1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0])
        bot1.Reshape([2,6])

        top0 = Blob()
        top1 = Blob()

        expect_bot0 = Blob()
        expect_bot0.Reshape([2,6])
        expect_bot0.set_diff([-0.49786511,0.00580323,0.01577482,0.0428804,0.116561,0.31684566,-0.49786511,0.00580323,0.01577482,0.0428804,0.116561,0.31684566])
        expect_bot0.Reshape([2,6])

        expect_top1 = Blob()
        expect_top1.Reshape([2,6])
        expect_top1.set_data([0.00426978, 0.01160646, 0.03154963, 0.08576079, 0.23312201,0.63369132,0.00426978, 0.01160646, 0.03154963, 0.08576079, 0.23312201,0.63369132])
        expect_top1.Reshape([2,6])

        layer = SoftmaxLossLayer()

        layer.Setup([bot0, bot1], [top0,top1])
        layer.Forward([bot0, bot1], [top0,top1])

        np.testing.assert_array_almost_equal( top1.data(), expect_top1.data() )

        top0.set_diff(1.0)
        layer.Backward([top0,top1], [], [bot0, bot1])

        np.testing.assert_array_almost_equal( bot0.diff(), expect_bot0.diff() )

    def test_SoftMaxLayer(self):
        bot0 = Blob()
        bot0.Reshape((3,))
        bot0.set_data([1,2,3])
        bot0.Reshape((3,))

        top0 = Blob()
        top0.Reshape((3,))

        expect_top0 = Blob()
        expect_top0.Reshape((3,))
        expect_top0.set_data([0.09003057, 0.24472847, 0.66524096])
        expect_top0.Reshape((3,))

        expect_bot0 = Blob()
        expect_bot0.Reshape((3,))
        expect_bot0.set_diff([0.0, 0.0, 0.0])
        expect_bot0.Reshape((3,))

        layer = SoftMaxLayer()
        layer.Setup([bot0], [top0])
        layer.Forward([bot0], [top0])

        np.testing.assert_array_almost_equal( top0.data(), expect_top0.data() )

        top0.set_diff(np.ones_like(top0.data()))
        layer.Backward([top0], [], [bot0])
        
        np.testing.assert_array_almost_equal( bot0.diff(), expect_bot0.diff() )

    def test_InnerProductLayer(self):
        bot0 = Blob()
        bot0.Reshape((2,3))
        bot0.set_data([1,2,3,4,5,6])
        bot0.Reshape((2,3))

        top0 = Blob()

        expect_b = Blob()
        expect_b.Reshape((1,2))
        expect_b.set_diff([2,2])
        expect_b.Reshape((1,2))

        expect_W = Blob()
        expect_W.Reshape((3,2))
        expect_W.set_diff([5,5,7,7,9,9])
        expect_W.Reshape((3,2))

        expect_bot0 = Blob()
        expect_bot0.Reshape((2,3))
        expect_bot0.set_diff([3,7,11,3,7,11])
        expect_bot0.Reshape((2,3))

        layer  = InnerProductLayer(3,2)
        layer.Setup([bot0], [top0])

        layer.W.set_data([1,2,3,4,5,6])
        layer.W.Reshape((3,2))

        expect_top0 = Blob()
        expect_top0.Reshape((2,2))
        expect_top0.set_data([22,28,49,64])
        expect_top0.Reshape((2,2))

        layer.Forward([bot0], [top0])
        np.testing.assert_array_almost_equal( top0.data(), expect_top0.data() )

        top0.set_diff(np.ones_like(top0.data()))
        layer.Backward([top0], [], [bot0])

        np.testing.assert_array_almost_equal( layer.b.diff(), expect_b.diff() )
        np.testing.assert_array_almost_equal( layer.W.diff(), expect_W.diff() )
        np.testing.assert_array_almost_equal( bot0.diff(), expect_bot0.diff() )

    def test_MaxPoolingLayer(self):
        top0 = Blob()

        bot0 = Blob()
        bot0.Reshape((1,1,4,4))
        bot0.set_data([5,3,1,2,1,2,3,2,4,2,2,5,3,6,1,1])
        bot0.Reshape((1,1,4,4))

        expect_top0 = Blob()
        expect_top0.Reshape((1,1,2,2))
        expect_top0.set_data([5,3,6,5])
        expect_top0.Reshape((1,1,2,2))

        expect_bot0 = Blob()
        expect_bot0.Reshape((1,1,4,4))
        expect_bot0.set_diff([1,0,0,0,0,0,0.8,0,0,0,0,0.6,0,0.4,0,0])
        expect_bot0.Reshape((1,1,4,4))

        layer  = MaxPoolingLayer(2,2,2)
        layer.Setup([bot0], [top0])

        layer.Forward([bot0], [top0])
        np.testing.assert_array_almost_equal( top0.data(), expect_top0.data() )

        top0.set_diff([1.0,0.8,0.4,0.6])
        top0.Reshape((1,1,2,2))
       
        layer.Backward([top0], [], [bot0])
        np.testing.assert_array_almost_equal( bot0.diff(), expect_bot0.diff() )

    def test_ConvolutionLayer(self):
        bot0 = Blob()
        bot0.Reshape((1,1,4,4))
        bot0.set_data([1,1,0,1,1,0,0,1,0,1,1,0,1,1,1,1])
        bot0.Reshape((1,1,4,4))

        top0 = Blob()

        expect_top0 = Blob()
        expect_top0.Reshape((1,1,3,3))
        expect_top0.set_data([2,1,1,2,2,1,2,3,3])
        expect_top0.Reshape((1,1,3,3))

        expect_W = Blob()
        expect_W.Reshape((1,1,2,2))
        expect_W.set_diff([5,5,6,6])
        expect_W.Reshape((1,1,2,2))

        expect_bot0 = Blob()
        expect_bot0.Reshape((1,1,4,4))
        expect_bot0.set_diff([1,1,1,0,2,3,3,1,2,3,3,1,1,2,2,1])
        expect_bot0.Reshape((1,1,4,4))

        expect_b = Blob()
        expect_b.Reshape((1,))
        expect_b.set_diff([9])
        expect_b.Reshape((1,))
     
        layer  = ConvolutionLayer(2,2,1,0,1)
        layer.Setup([bot0], [top0])
        layer.W.set_data([1,0,1,1])
        layer.W.Reshape((1,1,2,2))
        layer.Forward([bot0], [top0])

        np.testing.assert_array_almost_equal( top0.data(), expect_top0.data() )

        top0.set_diff(np.ones_like(top0.data()))
       
        layer.Backward([top0], [], [bot0])

        np.testing.assert_array_almost_equal( layer.W.diff(), expect_W.diff() )
        np.testing.assert_array_almost_equal( layer.b.diff(), expect_b.diff() )
        np.testing.assert_array_almost_equal( bot0.diff(), expect_bot0.diff() )

if __name__ == '__main__':
    unittest.main()
