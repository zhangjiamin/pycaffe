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

        solver = AdaDeltaSolver(0.1)
        solver.AddTrainNet(train_net)
        solver.AddTestNet(test_net)
        solver.Solve(3000)

if __name__ == '__main__':
    unittest.main()
