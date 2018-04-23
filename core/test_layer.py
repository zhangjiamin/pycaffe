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
        bot1 = Blob()

        bot0.Reshape([2,6])
        bot1.Reshape([2,6])

        bot0.set_data([1.0,2.0,3.0,4.0,5.0,6.0,1.0,2.0,3.0,4.0,5.0,6.0])
        bot1.set_data([1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0])

        bot0.Reshape([2,6])
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
        top1   = Blob()

        btd    = Blob()
        wwd    = Blob()
        bbd    = Blob()
       
        bbd.Reshape((1,2))
        bbd.set_data([2,2])
        bbd.Reshape((1,2))

        wwd.Reshape((3,2))
        wwd.set_data([5,5,7,7,9,9])
        wwd.Reshape((3,2))

        btd.Reshape((2,3))
        btd.set_data([3,7,11,3,7,11])
        btd.Reshape((2,3))

        bottom.Reshape((2,3))
        bottom.set_data([1,2,3,4,5,6])
        bottom.Reshape((2,3))
       
        layer  = InnerProductLayer(3,2)
        layer.Setup([bottom], [top])

        layer.W.set_data([1,2,3,4,5,6])
        layer.W.Reshape((3,2))

        top1.Reshape((2,2))
        top1.set_data([22,28,49,64])
        top1.Reshape((2,2))

        layer.Forward([bottom], [top])
        np.testing.assert_array_almost_equal( top.data(), top1.data() )

        top.set_diff([1,1,1,1])
        top.Reshape((2,2))
        layer.Backward([top], [], [bottom])

        np.testing.assert_array_almost_equal( layer.b.diff(), bbd.data() )
        np.testing.assert_array_almost_equal( layer.W.diff(), wwd.data() )
        np.testing.assert_array_almost_equal( bottom.diff(), btd.data() )

    def test_MaxPoolingLayer(self):
        bottom = Blob()
        top    = Blob()
        top1   = Blob()
        top2   = Blob()
        top3   = Blob()
        top4   = Blob()

        bottom.Reshape((1,1,4,4))
        bottom.set_data([5,3,1,2,1,2,3,2,4,2,2,5,3,6,1,1])
        bottom.Reshape((1,1,4,4))

        top1.Reshape((1,1,2,2))
        top1.set_data([5,3,6,5])
        top1.Reshape((1,1,2,2))

        top3.Reshape((1,1,4,4))
        top3.set_data([1,0,0,0,0,0,0.8,0,0,0,0,0.6,0,0.4,0,0])
        top3.Reshape((1,1,4,4))

        top4.Reshape((1,))
        top4.set_data([9])
        top4.Reshape((1,))
     
        layer  = MaxPoolingLayer(2,2,2)
        layer.Setup([bottom], [top])

        layer.Forward([bottom], [top])
        np.testing.assert_array_almost_equal( top.data(), top1.data() )

        top.set_diff([1.0,0.8,0.4,0.6])
        top.Reshape((1,1,2,2))
       
        layer.Backward([top], [], [bottom])
        np.testing.assert_array_almost_equal( bottom.diff(), top3.data() )

    def test_ConvolutionLayer(self):
        bottom = Blob()
        top    = Blob()
        top1   = Blob()
        top2   = Blob()
        top3   = Blob()
        top4   = Blob()

        bottom.Reshape((1,1,4,4))
        bottom.set_data([1,1,0,1,1,0,0,1,0,1,1,0,1,1,1,1])
        bottom.Reshape((1,1,4,4))

        top1.Reshape((1,1,3,3))
        top1.set_data([2,1,1,2,2,1,2,3,3])
        top1.Reshape((1,1,3,3))

        top2.Reshape((1,1,2,2))
        top2.set_data([5,5,6,6])
        top2.Reshape((1,1,2,2))

        top3.Reshape((1,1,4,4))
        top3.set_data([1,1,1,0,2,3,3,1,2,3,3,1,1,2,2,1])
        top3.Reshape((1,1,4,4))

        top4.Reshape((1,))
        top4.set_data([9])
        top4.Reshape((1,))
     
        layer  = ConvolutionLayer(2,2,1,0,1)
        layer.Setup([bottom], [top])
        layer.W.set_data([1,0,1,1])
        layer.W.Reshape((1,1,2,2))
        layer.Forward([bottom], [top])

        np.testing.assert_array_almost_equal( top.data(), top1.data() )

        top.set_diff([1,1,1,1,1,1,1,1,1])
        top.Reshape((1,1,3,3))
       
        layer.Backward([top], [], [bottom])

        np.testing.assert_array_almost_equal( layer.W.diff(), top2.data() )
        np.testing.assert_array_almost_equal( layer.b.diff(), top4.data() )
        np.testing.assert_array_almost_equal( bottom.diff(), top3.data() )

if __name__ == '__main__':
    unittest.main()
