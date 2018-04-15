import unittest
import numpy
import numpy as np
from blob import Blob
from load_data import load_data

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

    def test_SoftmaxLossLayer(self):
        bottom_0 = Blob()
        bottom_1 = Blob()

        bottom_0.set_data([1.0,2.0,3.0,4.0,5.0,6.0])
        bottom_1.set_data([1.0,0.0,0.0,0.0,0.0,0.0])

        bottom_0.Reshape([1,6])
        bottom_1.Reshape([1,6])

        top = Blob()
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
        bottom.set_data([1,2])
        bottom.Reshape((1,2))
       
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

    def test_mnist(self):
        datasets = load_data('mnist.pkl.gz')
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        bottom = Blob()
        top    = Blob()
        top1   = Blob()
        top2   = Blob()
        bottom.set_data(train_set_x[0])
        bottom.Reshape((1,1,28,28))
       
        conv1  = ConvolutionLayer(3,3,32,0,1)
        conv1.Setup([bottom], [top])
        conv1.Forward([bottom], [top])

        max_pool1 = MaxPoolingLayer(2,2,2)
        max_pool1.Setup([top], [top1])
        max_pool1.Forward([top], [top1])

        relu1 = ReLULayer()
        relu1.Setup([top1], [top2])
        relu1.Forward([top1], [top2])
        #layer.Backward([top], [], [bottom])

        print 'bottom',bottom.data(),bottom.data().shape
        print 'top',top.data(),top.data().shape
        print 'top1',top1.data(),top1.data().shape
        print 'top2',top2.data(),top2.data().shape

        #print 'W',layer.W.data(),layer.W.data().shape
        #print 'b',layer.b.data(),layer.b.data().shape
        #print 'top.diff',top.diff(),top.data().shape
        #print 'W.diff',layer.W.diff(),layer.W.data().shape
        #print 'b.diff',layer.b.diff(),layer.b.data().shape

    def test_mnist_mlp(self):
        datasets = load_data('mnist.pkl.gz')
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        bottom = Blob()
        label  = Blob()
        top    = Blob()
        top1   = Blob()
        top2   = Blob()
        loss   = Blob()
        top4   = Blob()

        layers = []

        fc1  = InnerProductLayer(1,784,384)
        relu = ReLULayer()
        drop = DropoutLayer(0.5)
        fc2  = InnerProductLayer(1,384,10)
        softmaxloss = SoftmaxLossLayer()

        layers.append(fc1)
        layers.append(relu)
        layers.append(drop)
        layers.append(fc2)
        layers.append(softmaxloss)

        bottom.set_data(train_set_x[0])
        label.set_data(train_set_y[0])
        bottom.Reshape((1,784))
        label.Reshape((1,10))

        bottoms = []
        tops = []

        bottoms.append([bottom])
        tops.append([top])
        bottoms.append([top])
        tops.append([top1])

        bottoms.append([top1])
        tops.append([top4])

        bottoms.append([top4])
        tops.append([top2])
        bottoms.append([top2,label])
        tops.append([loss])

        for i in range(len(layers)):
            layers[i].Setup(bottoms[i], tops[i])

        blobs = []
        blobs = fc1.blobs()
        blobs.extend(fc2.blobs())

        for j in range(100):
            count = 0
            for i in range(test_set_x.shape[0]):
                bottom.set_data(test_set_x[i])
                label.set_data(test_set_y[i])
                bottom.Reshape((1,784))
                label.Reshape((1,10))

                for ii in range(len(layers)):
                    layers[ii].Forward(bottoms[ii], tops[ii])
     
                if np.argmax(softmaxloss.probs_) == np.argmax(test_set_y[i]):
                    count = count + 1

            print 'Accurary:', j, count, test_set_x.shape[0], count*1.0/test_set_x.shape[0]

            for i in range(train_set_x.shape[0]):
                bottom.set_data(train_set_x[i])
                label.set_data(train_set_y[i])
                bottom.Reshape((1,784))
                label.Reshape((1,10))

                for ii in range(len(layers)):
                    layers[ii].Forward(bottoms[ii], tops[ii])
       
                loss.set_diff(loss.data()*0.01)

                for ii in reversed(range(len(layers))):
                    layers[ii].Backward(tops[ii], [], bottoms[ii])

                for ii in range(len(blobs)):
                    blobs[ii].Update()

if __name__ == '__main__':
    unittest.main()
