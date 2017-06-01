from __future__ import absolute_import

import unittest
import numpy as np
from numpy.testing import assert_allclose

from . import array as varray

class Test_shift_ip(unittest.TestCase):
    def runTest(self):
        x = np.asarray(np.arange(5), order='C')
        varray.shift_ip(x, 1)
        assert_allclose(x, np.array((0.0, 0.0, 1.0, 2.0, 3.0)))

        x = np.asarray(np.arange(5), order='F')
        varray.shift_ip(x, 1)
        assert_allclose(x, np.array((0.0, 0.0, 1.0, 2.0, 3.0)))

        x = np.asarray(np.arange(5), order='C')
        varray.shift_ip(x, -1)
        assert_allclose(x, np.array((1.0, 2.0, 3.0, 4.0, 4.0)))

        x = np.asarray(np.arange(5), order='F')
        varray.shift_ip(x, -1)
        assert_allclose(x, np.array((1.0, 2.0, 3.0, 4.0, 4.0)))

main = unittest.main

if __name__ == '__main__':
    main()
