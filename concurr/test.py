import unittest

import numpy as np
import numpy.testing

import concurr


class ConcurrTests(unittest.TestCase):
    def test_multiply(self):
        a = np.random.randn(20).astype(np.float64).reshape((4, 5))
        b = np.random.randn(20).astype(np.float64).reshape((5, 4))
        np_out = np.dot(a, b)
        out = concurr.matrix_multiply(a, b)
        numpy.testing.assert_array_almost_equal(out, np_out, 10)

    def test_multiply_big(self):
        x, y, z = 511, 512, 513
        a = np.random.randn(x * y).astype(np.float64).reshape((x, y))
        b = np.random.randn(y * z).astype(np.float64).reshape((y, z))
        np_out = np.dot(a, b)
        out = concurr.matrix_multiply(a, b)
        numpy.testing.assert_array_almost_equal(out, np_out, 5)

    def test_multiply_transposed(self):
        x, y, z = 511, 512, 513
        a = np.random.randn(y * x).astype(np.float64).reshape((y, x))
        a = a.T

        b = np.random.randn(y * z).astype(np.float64).reshape((y, z))

        a = np.array(a)
        b = np.array(b)
        out = concurr.matrix_multiply(a, b)
        np_out = np.dot(a, b)
        numpy.testing.assert_array_almost_equal(out, np_out, 5)



if __name__ == '__main__':
    unittest.main()
