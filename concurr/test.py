import random
import unittest

import numpy as np
import numpy.testing

import concurr.matrix


class ConcurrTests(unittest.TestCase):
    def test_multiply(self):
        a = np.random.randn(20).astype(np.float64).reshape((4, 5))
        b = np.random.randn(20).astype(np.float64).reshape((5, 4))
        np_out = np.dot(a, b)
        out = concurr.matrix.multiply(a, b).get()
        numpy.testing.assert_array_almost_equal(out, np_out, 10)

    def test_multiply_big(self):
        x, y, z = 511, 512, 513
        a = np.random.randn(x * y).astype(np.float64).reshape((x, y))
        b = np.random.randn(y * z).astype(np.float64).reshape((y, z))
        np_out = np.dot(a, b)
        out = concurr.matrix.multiply(a, b).get()
        numpy.testing.assert_array_almost_equal(out, np_out, 5)

    def test_multiply_transposed(self):
        x, y, z = 511, 512, 513
        a = np.random.randn(y * x).astype(np.float64).reshape((y, x))
        a = a.T

        b = np.random.randn(y * z).astype(np.float64).reshape((y, z))

        a = np.array(a)
        b = np.array(b)
        out = concurr.matrix.multiply(a, b).get()
        np_out = np.dot(a, b)
        numpy.testing.assert_array_almost_equal(out, np_out, 5)

    def test_multiply_tn(self):
        x, y, z = 10, 3, 100
        a = np.random.randn(x * y).astype(np.float64).reshape((y, x))
        b = np.random.randn(y * z).astype(np.float64).reshape((y, z))

        out = concurr.matrix.multiply_tn(a, b).get()
        np_out = np.dot(a.T, b)
        numpy.testing.assert_array_almost_equal(out, np_out, 5)

    def test_add(self):
        x, y = 10, 12
        a = np.random.randn(x * y).astype(np.float64).reshape((y, x))

        val = random.random()

        out = concurr.matrix.add(a, val).get()
        np_out = a + val
        numpy.testing.assert_array_almost_equal(out, np_out, 5)

    def test_mul(self):
        x, y = 10, 12
        a = np.random.randn(x * y).astype(np.float64).reshape((y, x))

        val = random.random()

        out = concurr.matrix.mul(a, val).get()
        np_out = a * val
        numpy.testing.assert_array_almost_equal(out, np_out, 5)

    def test_sum(self):
        x, y = 10, 12
        a = np.random.randn(x * y).astype(np.float64).reshape((y, x))
        b = np.random.randn(x * y).astype(np.float64).reshape((y, x))

        out = concurr.matrix.sum(a, b).get()
        np_out = a + b
        numpy.testing.assert_array_almost_equal(out, np_out, 5)

    def test_append_value_line(self):
        x, y = 10, 12
        x, y = 2, 3
        a = np.random.randn(x * y).astype(np.float64).reshape((y, x))
        b = np.ones((1, x))
        np_out = np.vstack((a, b))
        out = concurr.matrix.append_value_line(a, 1).get()

        numpy.testing.assert_array_almost_equal(out, np_out, 5)


if __name__ == '__main__':
    unittest.main()
