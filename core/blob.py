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

    def ShareData(self, other):
        self.data_ = other.data()

    def ShareDiff(self, other):
        self.diff_ = other.diff()

    def ShapeEquals(self, other):
        return (self.shape_ == other.shape)

    def Update(self):
        self.data_ = self.data_ - self.diff_

    def asum_data(self):
        return numpy.sum(numpy.abs(self.data_))

    def asum_diff(self):
        return numpy.sum(numpy.abs(self.diff_))

    def sumsq_data(self):
        return numpy.sum(numpy.dot(self.data_, self.data_))

    def sumsq_diff(self):
        return numpy.sum(numpy.dot(self.diff_, self.diff_))

    def data_at(self, indices):
        return self.data_[indices]

    def diff_at(self, indices):
        return self.diff_[indices]

    def CopyFrom(self, source, copy_diff=False, reshape=False):
        if True == reshape:
            self.ReshapeLike(source)

        if True == copy_diff:
            self.diff_ = source.diff()
        else:
            self.data_ = source.data()

    def FromProto(self, proto, reshape=True):
        pass

    def ToProto(self, proto, write_diff=False):
        pass


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