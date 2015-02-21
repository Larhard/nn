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


if __name__ == '__main__':
    unittest.main()
