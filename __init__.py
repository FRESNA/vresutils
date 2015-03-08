#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import networkx as nx
from itertools import izip
from functools import wraps

import cPickle
from warnings import warn
import time, sys
import os.path, string

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

class timer(object):
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        if len(self.name) > 0:
            sys.stdout.write(self.name + ": ")
            sys.stdout.flush()

        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            stop = time.time()
            usec = (stop - self.start) * 1e6
            if usec < 1000:
                print "%.1f usec" % usec
            else:
                msec = usec / 1000
                if msec < 1000:
                    print "%.1f msec" % msec
                else:
                    sec = msec / 1000
                    print "%.1f sec" % sec
        else:
            print "failed"

        return False

def indicator(N, indices):
    m = np.zeros(N)
    m[indices] = 1.
    return m

def rank(G, P=None):
    # we import only here, to avoid a cyclic import dependency
    import flow

    flower = flow.FlowerBicg(G)
    N = G.number_of_nodes()

    if P is None or len(P) != N:
        P = 2*np.random.random(N) - 1
        P[-1] = - P[:-1].sum()

    F = flower(P)
    flowtracer = flow.FlowTracer(G, P, F)

    return np.linalg.matrix_rank(densify(flowtracer.M))

def disable_sparse_safety_checks():
    import scipy.sparse, scipy.sparse.compressed

    def _check_format(s, full_check=True): pass
    scipy.sparse.compressed._cs_matrix.check_format = _check_format

    def _get_index_dtype(arrays=[], maxval=None, check_contents=False):
        return np.int32
    scipy.sparse.csc.get_index_dtype = _get_index_dtype
    scipy.sparse.csr.get_index_dtype = _get_index_dtype
    scipy.sparse.compressed.get_index_dtype = _get_index_dtype


def format_filename(s):
    """
    Take a string and return a valid filename constructed from the string.
    Uses a whitelist approach: any characters not present in valid_chars are
    removed. Also spaces are replaced with underscores.
    """
    valid_chars = frozenset("-_.() %s%s" % (string.ascii_letters, string.digits))
    return ''.join(c for c in s if c in valid_chars).replace(' ','_')


def cachable(func=None, version=None, cache_dir="/tmp/compcache", verbose=True):
    """
    Decorator to mark long running functions, which should be saved to
    disk for a pickled version of their hopefully short arguments.

    Arguments:
    func      - Shouldn't be supplied, but instead contains the function,
                if no arguments are supplied to the decorator
    version   - if given it is saved together with the arguments and
                must be the same as the cache to be valid.
    cache_dir - where to save the cached files, if it does not exist
                it is created (default: /tmp/compcache)
    verbose   - Output cache hits and timing information.
    """

    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)

    def deco(func):
        """
        Wrap function to check for a cached result. Everything is
        pickled to a file of the name
        cache_dir/<funcname>_param1_param2_kw1key.kw1val
        (it would be better to use np.save/np.load for numpy arrays)
        """

        cache_fn = cache_dir + "/" + func.__name__ + "_"
        if version is not None:
            cache_fn += format_filename("ver" + str(version) + "_")

        @wraps(func)
        def wrapper(*args, **kwds):
            # Check if there is a cached version
            fn = cache_fn + format_filename(
                '_'.join(str(a) for a in args) + '_' +
                '_'.join(str(k) + '.' + str(v) for k,v in kwds.iteritems()) +
                '.pickle'
            )

            if os.path.exists(fn):
                try:
                    with open(fn) as f:
                        return cPickle.load(f)
                except Exception as e:
                    warn("Couldn't unpickle from %s: %s" % (fn, e.message))

            with optional(verbose,
                          timer("Caching call to {} in {}"
                                .format(func.__name__, os.path.basename(fn)))):
                ret = func(*args, **kwds)
                try:
                    with open(fn, 'w') as f:
                        cPickle.dump(ret, f)
                except Exception as e:
                    warn("Couldn't pickle to %s: %s" % (fn, e.message))

                return  ret

        return wrapper

    if callable(func):
        return deco(func)
    else:
        return deco

class optional(object):
    def __init__(self, variable, contextman):
        self.variable = variable
        self.contextman = contextman

    def __enter__(self):
        if self.variable:
            return self.contextman.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.variable:
            return self.contextman.__exit__(exc_type, exc_val, exc_tb)
        return False
