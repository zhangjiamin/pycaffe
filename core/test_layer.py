import unittest
import numpy as np
from blob import Blob

from inner_product_layer import InnerProductLayer

class TestLayer(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_InnerProductLayer(self):
        bottom = Blob(np.float, (2,))
        top    = Blob(np.float, (2,))
        bottom.set_data([1,2])
        bottom.Reshape((2,))

        W      = Blob(np.float, (2,2))
        W.set_data([1,2,3,4])
        W.Reshape((2,2))
        
        layer  = InnerProductLayer()
        layer.Setup([bottom], [top])
        layer.W = W
        layer.Forward([bottom], [top])
        top.set_diff(top.data())
        layer.Backward([top], [], [bottom])

        print 'bottom',bottom.data(),bottom.data().shape
        print 'top',top.data(),top.data().shape
        print 'W',layer.W.data(),layer.W.data().shape
        print 'b',layer.b.data(),layer.b.data().shape
        print 'top.diff',top.diff(),top.data().shape
        print 'W.diff',layer.W.diff(),layer.W.data().shape
        print 'b.diff',layer.b.diff(),layer.b.data().shape


    def test_2(self):
        print 'test_2'

if __name__ == '__main__':
    unittest.main()
