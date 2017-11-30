from __future__ import absolute_import
from __future__ import print_function

import sys
import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import scipy.stats
from itertools import repeat
from collections import namedtuple

from .array import positive, negative, densify, spdiag, normed

from . import flow as vflow
from six.moves import range, zip

def _as_incidence_matrix(topology):
    """
    Return incidence_matrix for `topology`

    Parameters
    ----------
    topology : nx.OrderedGraph | N x L
        Topology as an NetworkX graph or directly as an incidence matrix
        (optimally given as a sparse matrix in csc format)

    Returns
    -------
    K : sp.sparse.csc_matrix
        incidence matrix with +1 out-link, -1 in-link convention
    """
    if isinstance(topology, nx.Graph):
        return - sp.sparse.csc_matrix(nx.incidence_matrix(topology, oriented=True))
    else:
        return sp.sparse.csc_matrix(topology)


def flowtracing(topology, P, F, qin, chi=None, **flowtraceropts):
    """
    Compute the out-partition from the in-partition qin using
    flowtracing/average participation.

    Parameters
    ----------
    topology : nx.OrderedGraph | N x L
        Topology as an NetworkX graph or directly as an incidence matrix
        (optimally given as a sparse matrix in csc format)
    P : T x N | T x 2 x N
        Injection pattern, if it is 3 dimensional, :,0,: is
        interpreted as input and :,1,: as output of each node.
    F : T x L
        In-Flows
    qin : T x A x N | A x N
        Time-varying or constant in-partition for a set of entities A
    chi : T x L
        Link Losses (defaults to None)

    Returns
    -------
    qout : T x A x N
        Out-partition

    Remark
    ------
    The incidence matrix is expected to follow the convention that an
    out-link is marked with +1 and an in-link is marked with a
    -1. This convention is the opposite of the convention of NX.
    """
    no_of_times, no_of_nodes = P.shape[0], P.shape[-1]
    no_of_links = F.shape[-1]
    no_of_entities = qin.shape[0 if np.ndim(qin) <= 2 else 1]

    K = _as_incidence_matrix(topology)
    qout = np.empty((no_of_times, no_of_entities, no_of_nodes))
    #ql = np.empty((no_of_times, no_of_entities, no_of_links))

    if np.ndim(qin) <= 2:
        qin = repeat(qin)

    if chi is None:
        chi = repeat(chi)

    def lines_at_bus(bus):
        return K[bus].nonzero()[1]
    def no_of_neighbours(bus):
        lines = lines_at_bus(bus)
        buses = np.concatenate([K[:,line].nonzero()[0]
                                for line in lines])
        return len(np.unique(buses)) - 1

    buses = frozenset(bus
                      for bus in np.where((abs(P) < 1e-12).any(axis=0))[0]
                      if  no_of_neighbours(bus) < 2)
    reduced_configurations = {}

    Configuration = namedtuple('Configuration', ['K', 'busmask', 'linemask'])
    for i, qin_i, chi_i in zip(np.arange(no_of_times), qin, chi):
        busesdel = buses.intersection(np.where(abs(P[i]) < 1e-12)[0])
        if busesdel and False:
            if busesdel not in reduced_configurations:
                linesdel = np.concatenate(list(map(lines_at_bus, busesdel)))
                busmask = ~np.in1d(np.r_[:K.shape[0]], list(busesdel))
                linemask = ~np.in1d(np.r_[:K.shape[1]], linesdel)
                c = Configuration(K=K[np.ix_(busmask, linemask)],
                                  busmask=busmask, linemask=linemask)
                reduced_configurations[busesdel] = c
            else:
                c = reduced_configurations[busesdel]

            flowtracer = FlowTracer(c.K,
                                    P[i, c.busmask],
                                    F[i, c.linemask],
                                    chi=chi_i[c.linemask] if chi_i is not None else None,
                                    **flowtraceropts)
            qout_r = flowtracer(qin_i[:, c.busmask])
            qout[i:i+1, :, c.busmask] = qout_r
            qout[i:i+1, :, ~c.busmask] = np.nan
        else:
            flowtracer = FlowTracer(K, P[i], F[i], chi_i,
                                    **flowtraceropts)
            qout[i] = flowtracer(qin_i)
        #ql[i] = qout[i] * flowtracer.K3

    return qout #, ql
averageparticipation = flowtracing

def virtualinjectionpattern(G, P, qin, PTDF=None, susceptance=None):
    """
    Compute the flow-partition from the in-partition qin using the
    virtual injection pattern method, a part of the renormalized
    marginal participation term.

    Parameters
    ----------
    G : nx.OrderedGraph
        Topology
    P : T x N
        Injection pattern
    qin : T x A x N | A x N
        Time-varying or constant in-partition for a set of entities A

    Returns
    -------
    ql : T x A x L
        Flow partition
    """

    if PTDF is None:
        PTDF = vflow.PTDF(G, susceptance=susceptance)
    PTDFt = PTDF.T

    Pn = normed(negative(P), axis=1)
    Pp = qin[...,:,:] * positive(P)[:,np.newaxis,:]
    Fout = (Pp - Pn[:,np.newaxis,:] * Pp.sum(axis=-1, keepdims=True)).dot(PTDFt)
    F = P.dot(PTDFt)

    return Fout / F[:,np.newaxis,:]

def flowpartition(topology, F, qout):
    no_of_times, no_of_links = F.shape
    no_of_entities = qout.shape[1]

    K = _as_incidence_matrix(topology)
    qf = np.empty((no_of_times, no_of_entities, no_of_links))

    for i in range(no_of_times):
        F_i = np.where(abs(F[i]) > 1e-13, np.sign(F[i]), np.nan)
        qf[i] = qout[i] * positive(K.dot(spdiag(F_i)))

    return qf

def attribution(qf, F, quantile=0.99, no_of_bins=50, intermediates=None):
    """
    Flowtracing doubleintegral

    Parameters
    ----------
    qf : T x A x L
        Flow partition
    F : T x L
        Flows on the links
    quantile : None|Float
        If not None, largest flows will be moved to respective quantile
    no_of_bins : int
        Number of bins on which to calculate the integrals
    intermediates : None|dict
        If a dict is provided, then intermediate results are added to
        the dictionary

    Returns
    -------
    capacity : L x A
    """
    if intermediates is None:
        intermediates = dict()

    no_of_times, no_of_links = F.shape
    no_of_entities = qf.shape[1]
    assert qf.shape[0] == no_of_times and qf.shape[2] == no_of_links

    F = abs(F)
    if quantile is not None:
        maxf = np.percentile(F, 100.*quantile, axis=0)
        np.putmask(F, F > maxf, maxf)

    bins = intermediates['bins'] = \
           np.array([np.linspace(0., c, no_of_bins+1, endpoint=True)
                     for c in F.max(axis=0)])

    fcounts = intermediates['fcounts'] = np.empty((no_of_links, 1, no_of_bins))
    qmean = intermediates['qmean'] = np.empty((no_of_links, no_of_entities, no_of_bins))

    for l in range(no_of_links):
        fl = F[:,l]
        ql = qf[:,:,l]
        sorting_index = np.argsort(fl)
        sfl = fl[sorting_index]
        sql = ql[sorting_index]

        bin_index = np.r_[sfl.searchsorted(bins[l,:-1], 'left'), \
                          sfl.searchsorted(bins[l,-1], 'right')]
        fcounts[l,0] = np.diff(bin_index)

        zero = np.zeros((no_of_entities,))
        qmean[l] = np.array([np.nanmean(a, axis=0) if (~np.isnan(a)).any() else zero
                             for a in np.split(sql, bin_index[1:-1])]).T

    binsizes = np.diff(bins, axis=-1)[:,np.newaxis]
    p = intermediates['p'] = \
        (fcounts
         / binsizes
         / fcounts.sum(axis=-1)[:,:,np.newaxis])

    qmean = np.nan_to_num(qmean / qmean.sum(axis=1, keepdims=True))
    h = intermediates['h'] = qmean

    N = np.empty(p.shape)
    N[:,0,0]  = 1.
    N[:,0,1:] = 1 - np.cumsum(p*binsizes, axis=-1)[:,0,:-1]
    w = intermediates['w'] = \
        np.cumsum((p*h*binsizes)[:,:,::-1], axis=-1)[:,:,::-1] / N

    return (w*binsizes).sum(axis=-1)


class FlowTracer(object):
    count_singular = 0

    def __init__(self, topology, P, F, chi=None, expect_singular=False, raise_on_singular=True):
        # split P into inputs and outputs, keep sources for later
        if np.ndim(P) > 1:
            assert len(P) == 2
            self.Pp = P[0]
            Pm = P[1]
        else:
            self.Pp = positive(P)
            Pm = negative(P)

        K = _as_incidence_matrix(topology)

        # Build flow tracing matrix
        K2 = K * spdiag(np.sign(F))
        self.K3 = positive(K2)

        # Handle losses
        if chi is not None:
            Pm = Pm + positive(K * spdiag(np.sign(chi))) * abs(chi)

        self.M = M = K2 * spdiag(np.abs(F)) * self.K3.T + spdiag(Pm)

        def itsolver(q):
            qs, info = sp.sparse.linalg.gmres(M, q)
            if info != 0:
                print('SERIOUS WARNING: Even the iterative solver failed :(. We will return NaN for this injection pattern (Sorry).', file=sys.stderr)
                qs = np.NaN
            return qs

        if expect_singular:
            self.solver = itsolver
        else:
            try:
                self.solver = sp.sparse.linalg.splu(M).solve
            except RuntimeError as e:
                if 'singular' not in e.args[0] or raise_on_singular:
                    raise

                self.__class__.count_singular += 1
                if self.__class__.count_singular % 5000 == 0:
                    print('WARNING: We encountered a singular flowtracing matrix M already for {} times. Is this really necessary?'.format(self.__class__.count_singular), file=sys.stderr)

                self.solver = itsolver

    def __call__(self, qin):
        if qin.ndim == 1:
            return self.solver(self.Pp * qin)
        else:
            qout = np.empty(qin.shape)
            Pqin = self.Pp * qin
            for i in range(qout.shape[0]):
                qout[i] = self.solver(Pqin[i])
            return qout
