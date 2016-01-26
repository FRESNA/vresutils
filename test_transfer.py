from __future__ import absolute_import

import unittest
import numpy as np
from numpy.testing import assert_allclose

import scipy.sparse as sparse

from . import transfer

def make_orig_dest():
    orig = np.array(((0., 0.), (1., 1.)))
    dest = np.array(((1.5, 1.), (0., 0.5), (1.5, 0.)))

    return orig, dest

class Points2PointsTest(unittest.TestCase):
    def test_not_surjective(self):
        orig, dest = make_orig_dest()
        spmatrix = transfer.Points2Points(orig, dest)
        self.assertTrue(sparse.issparse(spmatrix))
        self.assertEqual(spmatrix.shape, (len(dest), len(orig)))
        matrix = spmatrix.todense()
        assert_allclose(matrix, np.array(((0., 1.),
                                          (1., 0.),
                                          (0., 0.))))

    def test_surjective(self):
        orig, dest = make_orig_dest()
        spmatrix = transfer.Points2Points(orig, dest, surjective=True)
        self.assertTrue(sparse.issparse(spmatrix))
        self.assertEqual(spmatrix.shape, (len(dest), len(orig)))
        matrix = spmatrix.todense()
        assert_allclose(matrix, np.array(((0., 2./3.),
                                          (1., 0.),
                                          (0., 1./3.))))


main = unittest.main

if __name__ == '__main__':
    main()
