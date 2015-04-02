#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp, scipy.sparse

def positive(a):
    if hasattr(a, "multiply"):
        if sp.sparse.isspmatrix_csc(a) or sp.sparse.isspmatrix_csr(a):
            m = a.__class__((positive(a.data), a.indices.copy(), a.indptr.copy()),
                            shape=a.shape, dtype=a.dtype)
            m.eliminate_zeros()
            return m
        else:
            return - a.multiply(a<0)
    else:
        return a * (a>0)

def negative(a):
    if hasattr(a, "multiply"):
        if sp.sparse.isspmatrix_csc(a) or sp.sparse.isspmatrix_csr(a):
            m = a.__class__((negative(a.data), a.indices.copy(), a.indptr.copy()),
                            shape=a.shape, dtype=a.dtype)
            m.eliminate_zeros()
            return m
        else:
            return - a.multiply(a<0)
    else:
        return - a * (a<0)

def spdiag(v, k=0):
    if k == 0:
        N = len(v)
        inds = np.arange(N+1, dtype=np.int32)
        return sp.sparse.csc_matrix((v, inds[:-1], inds), (N, N))
    else:
        return sp.sparse.diags((v,),(k,))

def densify(a):
    """Return a dense array version of a """
    if sp.sparse.issparse(a):
        a = a.todense()
    return np.asarray(a)

def interpolate(a, axis=0):
    """Interpolate NaN values"""
    def interpolate1d(y):
        nan = np.isnan(y)
        x = lambda z: z.nonzero()[0]
        y[nan] = np.interp(x(nan), x(~nan), y[~nan])
        return y
    a = np.apply_along_axis(interpolate1d, axis, a)
    return a

def disable_sparse_safety_checks():
    import scipy.sparse, scipy.sparse.compressed

    def _check_format(s, full_check=True): pass
    scipy.sparse.compressed._cs_matrix.check_format = _check_format

    def _get_index_dtype(arrays=[], maxval=None, check_contents=False):
        return np.int32
    scipy.sparse.csc.get_index_dtype = _get_index_dtype
    scipy.sparse.csr.get_index_dtype = _get_index_dtype
    scipy.sparse.compressed.get_index_dtype = _get_index_dtype
