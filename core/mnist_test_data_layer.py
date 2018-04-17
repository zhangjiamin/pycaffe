from layer import Layer
from blob import Blob
import numpy
from load_data import load_data

class MNISTTestDataLayer(Layer):

    def __init__(self, batch_size):
        Layer.__init__(self)
        self.batch_size_ = batch_size
        self.datasets_  = None
        self.cur_ = 0

    def LayerSetup(self, bottom, top):
        self.datasets_ = load_data('mnist.pkl.gz')
        self.train_set_x, self.train_set_y = self.datasets_[0]
        self.valid_set_x, self.valid_set_y = self.datasets_[1]
        self.test_set_x,  self.test_set_y  = self.datasets_[2]
        print self.train_set_x.shape
        print self.train_set_y.shape
        print self.valid_set_x.shape
        print self.valid_set_y.shape
        print self.test_set_x.shape
        print self.test_set_y.shape


    def Reshape(self, bottom, top):
        top0_shape = (self.batch_size_, 784)
        top1_shape = (self.batch_size_, 10)
        top[0].Reshape(top0_shape)
        top[1].Reshape(top1_shape)

    def type(self):
        return 'MNISTTestDataLayer'

    def ExactNumBottomBlobs(self):
        return 0

    def ExactNumTopBlobs(self):
        return 2

    def Forward_cpu(self, bottom, top):
        top[0].set_data( self.test_set_x[self.cur_:self.cur_+self.batch_size_]  )
        top[1].set_data( self.test_set_y[self.cur_:self.cur_+self.batch_size_]  )
        self.cur_ = self.cur_ + self.batch_size_
        if self.cur_ >= self.test_set_x.shape[0]:
            self.cur_ = 0

