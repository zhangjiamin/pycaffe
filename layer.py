import numpy
from blob import Blob

class Layer:

    def __init__(self):
        self.blobs_ = []
        self.loss_  = []
        self.param_propagate_down_ = []

if __name__ == '__main__':
    blob  = Blob(numpy.float, (2,3))
    layer = Layer()


