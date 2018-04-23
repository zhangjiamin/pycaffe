import unittest
import numpy
from blob import Blob

class TestBlob(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_Reshape(self):
        blob = Blob()
        blob.set_data(numpy.array(range(30),float))
        blob.set_diff(numpy.array(range(30),float))
        blob.Reshape((5,3))
        blob.Reshape((5,6))
        blob.Reshape((2,15))

    def test_ReshapeLike(self):
        blob   = Blob()
        other1 = Blob()
        other2 = Blob()
        blob.set_data(numpy.array(range(30),float))
        blob.set_diff(numpy.array(range(30),float))
        blob.ReshapeLike(other1)
        blob.ReshapeLike(other2)
        blob.ReshapeLike(other1)

    def test_asum_data(self):
        blob = Blob()
        blob.set_data(numpy.array(range(30),float))
        res = blob.asum_data()
        self.assertEqual(res, 435.0)

    def test_asum_diff(self):
        blob = Blob()
        blob.set_diff(numpy.array(range(30),float))
        res = blob.asum_diff()
        self.assertEqual(res, 435.0)

    def test_sumsq_data(self):
        blob = Blob()
        blob.set_data(numpy.array(range(4),float))
        res = blob.sumsq_data()
        self.assertEqual(res, 14.0)

    def test_sumsq_diff(self):
        blob = Blob()
        blob.set_diff(numpy.array(range(4),float))
        res = blob.sumsq_diff()
        self.assertEqual(res, 14.0)

    def test_data_at(self):
        blob = Blob()
        blob.set_data(numpy.array(range(4)).reshape(2,2))
        res = blob.data_at((1,1))
        self.assertEqual(res, 3)

    def test_diff_at(self):
        blob = Blob()
        blob.set_diff(numpy.array(range(4)).reshape(2,2))
        res = blob.diff_at((1,1))
        self.assertEqual(res, 3)


if __name__ == '__main__':
    unittest.main()

