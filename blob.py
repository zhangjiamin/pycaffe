import numpy

class Blob:

    def __init__(self, dtype, shape):
        self.shape_ = shape
        self.data_  = numpy.zeros(shape, dtype)
        self.diff_  = numpy.zeros(shape, dtype)

    def Reshape(self, shape):
        if self.volume(self.shape_) == self.volume(shape):
            self.shape_ = shape
            self.data_  = numpy.reshape(self.data_, shape)
            self.diff_  = numpy.reshape(self.diff_, shape)
        else:
            self.shape_ = shape
            self.data_  = numpy.zeros(shape, dtype)
            self.diff_  = numpy.zeros(shape, dtype)

    def ReshapeLike(self, other):
        self.Reshape(other.shape())

    def shape_string(self):
        return str(self.shape_)

    def shape(self):
        return self.shape_

    def shape_index(self, index):
        return self.shape_[index]

    def num_axes(self):
        return self.data_.ndim

    def count(self):
        return self.data_.size

    def volume(self, shape):
        volume = 1
        for i in shape:
            volume = volume * i
        return volume

    def data(self):
        return self.data_

    def diff(self):
        return self.diff_

    def set_data(self, data):
        self.data_ = data

    def set_diff(self, diff):
        self.diff_ = diff

    def scale_data(self, scale_factor):
        self.data_ = self.data_ * scale_factor

    def scale_diff(self, scale_factor):
        self.diff_ = self.diff_ * scale_factor

if __name__ == '__main__':
    blob = Blob(numpy.float, (5,6))
    othe = Blob(numpy.float, (6,5))

    blob.set_data(numpy.array(range(30),float))
    blob.set_diff(numpy.array(range(30),float))

    blob.Reshape((2,15))
    blob.ReshapeLike(othe)
    print blob.data_
    print blob.shape()
    print blob.shape_index(1)
    print blob.shape_string()
    print blob.num_axes()
    print blob.count()

    blob.scale_data(0.1)
    blob.scale_diff(0.01)
    print blob.data()
    print blob.diff()
