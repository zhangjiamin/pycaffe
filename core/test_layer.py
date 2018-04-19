import unittest
import numpy
import numpy as np
from blob import Blob
from load_data import load_data
from net import Net
from sgd_solver import SGDSolver
from adagrad_solver import AdaGradSolver
from adam_solver import AdamSolver

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

        print 'bottom',bottom.data(),bottom.data().shape
        print 'top',top.data(),top.data().shape
        print 'top1',top1.data(),top1.data().shape
        print 'top2',top2.data(),top2.data().shape

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
        top5   = Blob()

        layers = []

        batch_size = 100

        fc1  = InnerProductLayer(784,392)
        relu = ReLULayer()
        drop = DropoutLayer(1.0)
        fc2  = InnerProductLayer(392,10)
        softmaxloss = SoftmaxLossLayer()

        layers.append(fc1)
        layers.append(relu)
        layers.append(drop)
        layers.append(fc2)
        layers.append(softmaxloss)

        bottom.set_data(train_set_x[0:batch_size])
        label.set_data(train_set_y[0:batch_size])
        bottom.Reshape((batch_size,784))
        label.Reshape((batch_size,10))

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
        tops.append([loss,top5])

        for i in range(len(layers)):
            layers[i].Setup(bottoms[i], tops[i])

        blobs = []
        blobs = fc1.blobs()
        blobs.extend(fc2.blobs())

        b1 = 0.9
        b2 = 0.999
        s = []
        r = []

        for i in range(len(blobs)):
            t1 = Blob()
            t2 = Blob()
            t1.ReshapeLike(blobs[i])
            t2.ReshapeLike(blobs[i])
            s.append(t1)
            r.append(t2)

        lr = 0.001
        eps = 1e-8

        t = 0
        runing_lr = 0
        for j in range(0):
            count = 0
            total = 0
            
            for i in range(0, test_set_x.shape[0], batch_size):
                bottom.set_data(test_set_x[i:batch_size+i])
                label.set_data(test_set_y[i:batch_size+i])
                bottom.Reshape((batch_size,784))
                label.Reshape((batch_size,10))

                for ii in range(len(layers)):
                    layers[ii].Forward(bottoms[ii], tops[ii])
    
                count = count + np.sum( np.equal( np.argmax(softmaxloss.probs_,axis=1), np.argmax(test_set_y[i:i+batch_size],axis=1) ) )
                total = total + batch_size

            print 'Accuracy:', j, count, total, count*1.0/total

            for i in range(0, train_set_x.shape[0], batch_size):
                bottom.set_data(train_set_x[i:i+batch_size])
                label.set_data(train_set_y[i:i+batch_size])
                bottom.Reshape((batch_size,784))
                label.Reshape((batch_size,10))

                loss_ = 0
                for ii in range(len(layers)):
                    loss_ += layers[ii].Forward(bottoms[ii], tops[ii])

                for ii in reversed(range(len(layers))):
                    layers[ii].Backward(tops[ii], [], bottoms[ii])

                t += 1

                for ii in range(len(blobs)):
                    s[ii].set_data( b1*s[ii].data() + (1.0 - b1)*blobs[ii].diff()  )
                    r[ii].set_data( b2*r[ii].data() + (1.0 - b2)*np.square(blobs[ii].diff())  )
                    s_ = s[ii].data()/(1.0-np.power(b1,t))
                    r_ = r[ii].data()/(1.0-np.power(b2,t))
                    runing_lr = lr*s_/(np.sqrt(r_) + eps)
                    blobs[ii].set_diff( runing_lr )
                    blobs[ii].Update()
                    blobs[ii].set_diff( numpy.zeros(blobs[ii].shape()) )

    def test_mnist_mlp_net(self):
        return
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

        fc1  = InnerProductLayer(784,392)
        relu = ReLULayer()
        drop = DropoutLayer(1.0)
        fc2  = InnerProductLayer(392,10)
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
        test_net.AddLayer(drop, [top1], [top4])
        test_net.AddLayer(fc2, [top4], [top2])
        test_net.AddLayer(softmaxloss, [top2,label], [loss,top5])
        test_net.AddLayer(acc, [top5,label], [top6,top7])

        b1 = 0.9
        b2 = 0.999
        s = []
        r = []

        for i in range(len(train_net.learnable_params_)):
            t1 = Blob()
            t2 = Blob()
            t1.ReshapeLike(train_net.learnable_params_[i])
            t2.ReshapeLike(train_net.learnable_params_[i])
            s.append(t1)
            r.append(t2)

        lr = 0.001
        eps = 1e-8

        t = 0
        runing_lr = 0
        for j in range(1000):
            count = 0
            total = 0
            
            for i in range(100):
                test_net.Forward()
                count = count + top6.data()
                total = total + top7.data()

            print 'Accuracy:', j, count, total, count*1.0/total

            for i in range(500):
                train_net.ForwardBackward()
                t += 1
                blobs = train_net.learnable_params_

                for ii in range(len(blobs)):
                    s[ii].set_data( b1*s[ii].data() + (1.0 - b1)*blobs[ii].diff()  )
                    r[ii].set_data( b2*r[ii].data() + (1.0 - b2)*np.square(blobs[ii].diff())  )
                    s_ = s[ii].data()/(1.0-np.power(b1,t))
                    r_ = r[ii].data()/(1.0-np.power(b2,t))
                    runing_lr = lr*s_/(np.sqrt(r_) + eps)
                    blobs[ii].set_diff( runing_lr )

                train_net.Update()
                train_net.ClearParamDiffs()
              
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
        solver.Solve(50000)

if __name__ == '__main__':
    unittest.main()
