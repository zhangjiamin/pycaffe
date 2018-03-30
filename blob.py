import numpy

class Blob:

    def __init__(self, dtype, shape):
        self.shape_ = shape
        self.data_  = numpy.zeros(shape,dtype)

    def reshape(self, shape):
        self.shape_ = shape
        self.data_  = numpy.reshape(self.data_, shape)

if __name__ == '__main__':
    blob = Blob(numpy.float, (5,6))
    print blob.data_
    print blob.data_.shape
    print blob.shape_
    blob.reshape((2,15))
    print blob.data_
    print blob.data_.shape
    print blob.shape_

