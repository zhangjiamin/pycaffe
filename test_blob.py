import unittest
import numpy
from blob import Blob

class TestBlob(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_asum(self):
        blob = Blob(numpy.float, (5,6))
        blob.set_data(numpy.array(range(30),float))
        print blob.asum_data()

    def test_2(self):
        print 'test_2'

if __name__ == '__main__':
    unittest.main()

'''
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
'''
