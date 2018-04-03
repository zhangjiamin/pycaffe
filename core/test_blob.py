import unittest
import numpy
from blob import Blob

class TestBlob(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_Reshape(self):
        blob = Blob(numpy.float, (5,6))
        blob.set_data(numpy.array(range(30),float))
        blob.set_diff(numpy.array(range(30),float))
        blob.Reshape((5,3))
        blob.Reshape((5,6))
        blob.Reshape((2,15))

    def test_ReshapeLike(self):
        blob   = Blob(numpy.float, (5,6))
        other1 = Blob(numpy.float, (5,6))
        other2 = Blob(numpy.float, (5,3))
        blob.set_data(numpy.array(range(30),float))
        blob.set_diff(numpy.array(range(30),float))
        blob.ReshapeLike(other1)
        blob.ReshapeLike(other2)
        blob.ReshapeLike(other1)

    def test_asum_data(self):
        blob = Blob(numpy.float, (5,6))
        blob.set_data(numpy.array(range(30),float))
        res = blob.asum_data()
        self.assertEqual(res, 435.0)

    def test_asum_diff(self):
        blob = Blob(numpy.float, (5,6))
        blob.set_diff(numpy.array(range(30),float))
        res = blob.asum_diff()
        self.assertEqual(res, 435.0)

    def test_sumsq_data(self):
        blob = Blob(numpy.float, (2,2))
        blob.set_data(numpy.array(range(4),float))
        res = blob.sumsq_data()
        self.assertEqual(res, 14.0)

    def test_sumsq_diff(self):
        blob = Blob(numpy.float, (2,2))
        blob.set_diff(numpy.array(range(4),float))
        res = blob.sumsq_diff()
        self.assertEqual(res, 14.0)

    def test_data_at(self):
        blob = Blob(numpy.float, (2,2))
        blob.set_data(numpy.array(range(4)).reshape(2,2))
        res = blob.data_at((1,1))
        self.assertEqual(res, 3)

    def test_diff_at(self):
        blob = Blob(numpy.float, (2,2))
        blob.set_diff(numpy.array(range(4)).reshape(2,2))
        res = blob.diff_at((1,1))
        self.assertEqual(res, 3)


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
