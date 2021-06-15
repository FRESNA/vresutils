from __future__ import absolute_import

import pytest
import numpy as np
from numpy.testing import assert_allclose

import scipy.sparse as sparse

from vresutils import transfer

@pytest.fixture
def orig():
    return np.array(((0., 0.), (1., 1.)))

@pytest.fixture
def dest():
    return np.array(((1.5, 1.), (0., 0.5), (1.5, 0.)))

class TestPoints2Points:
    def test_not_surjective(self, orig, dest):
        spmatrix = transfer.Points2Points(orig, dest)

        assert sparse.issparse(spmatrix)
        assert spmatrix.shape == (len(dest), len(orig))

        matrix = spmatrix.todense()
        assert_allclose(matrix, np.array(((0., 1.),
                                          (1., 0.),
                                          (0., 0.))))

    def test_surjective(self, orig, dest):
        spmatrix = transfer.Points2Points(orig, dest, surjective=True)

        assert sparse.issparse(spmatrix)
        assert spmatrix.shape == (len(dest), len(orig))

        matrix = spmatrix.todense()
        assert_allclose(matrix, np.array(((0., 2./3.),
                                          (1., 0.),
                                          (0., 1./3.))))
