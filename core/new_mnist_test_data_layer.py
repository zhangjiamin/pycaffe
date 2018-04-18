from layer import Layer
from blob import Blob
import numpy
from mnist import load_mnist

class MNISTTestDataLayer(Layer):

    def __init__(self, batch_size):
        Layer.__init__(self)
        self.batch_size_ = batch_size
        self.datasets_  = None

    def LayerSetup(self, bottom, top):
        self.datasets_ = load_mnist()
        self.train_set_ = self.datasets_[0]
        self.valid_set_ = self.datasets_[1]
        self.test_set_  = self.datasets_[2]
        print self.train_set_.images.shape
        print self.train_set_.labels.shape
        print self.valid_set_.images.shape
        print self.valid_set_.labels.shape
        print self.test_set_.images.shape
        print self.test_set_.labels.shape

    def Reshape(self, bottom, top):
        top0_shape = (self.batch_size_, 784)
        top1_shape = (self.batch_size_, 10)
        top[0].Reshape(top0_shape)
        top[1].Reshape(top1_shape)

    def type(self):
        return 'MNISTTrainDataLayer'

    def ExactNumBottomBlobs(self):
        return 0

    def ExactNumTopBlobs(self):
        return 2

    def Forward_cpu(self, bottom, top):
        images,labels = self.test_set_.next_batch(self.batch_size_)
        top[0].set_data( images )
        top[1].set_data( labels )
        top0_shape = (self.batch_size_, 784)
        top1_shape = (self.batch_size_, 10)
        top[0].Reshape(top0_shape)
        top[1].Reshape(top1_shape)

